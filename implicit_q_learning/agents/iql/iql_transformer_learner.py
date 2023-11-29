"""Implementations of algorithms for continuous control."""

from functools import partial
from typing import Dict, Optional, Sequence

import gym
import jax
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState

from networks import UnitStdNormalPolicy, MLP, Ensemble, SequenceMultiplexer
from networks.values import StateValue, StateActionValue
from networks.encoders import TransformerEncoder
from agents.iql.iql_learner import IQLLearner


def _share_encoder(source, target):
    replacers = {}

    for k, v in source.params.items():
        if "encoder" in k:
            replacers[k] = v

    # Use critic conv layers in actor:
    new_params = target.params.copy(add_or_replace=replacers)
    return target.replace(params=new_params)


class IQLTransformerLearner(IQLLearner):
    critic: TrainState
    target_critic: TrainState
    value: TrainState
    tau: float
    discount: float
    expectile: float
    critic_reduction: str
    A_scaling: float

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        actor_lr: float = 3e-4,
        value_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        decay_steps: Optional[int] = None,
        hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        expectile: float = 0.9,
        A_scaling: float = 10.0,
        critic_reduction: str = "min",
        apply_tanh: bool = False,
        dropout_rate: Optional[float] = None,
        encoder: str = "transformer",
        latent_dim: int = 512,
        depth: int = 2,
        num_heads: int = 8,
    ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1801.01290
        """

        action_dim = action_space.shape[-1]
        observations = observation_space.sample()
        actions = action_space.sample()

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)

        if encoder == "transformer":
            encoder_cls = partial(TransformerEncoder, emb_dim=latent_dim, depth=depth, num_heads=num_heads)
        else:
            raise NotImplementedError

        actor_base_cls = partial(MLP, hidden_dims=hidden_dims, activate_final=True)
        actor_cls = partial(UnitStdNormalPolicy, base_cls=actor_base_cls, action_dim=action_dim)
        actor_def = SequenceMultiplexer(encoder_cls=encoder_cls, network_cls=actor_cls, latent_dim=latent_dim)
        if decay_steps is not None:
            schedule_fn = optax.cosine_decay_schedule(-actor_lr, decay_steps)
            optimiser = optax.chain(optax.scale_by_adam(), optax.scale_by_schedule(schedule_fn))

        actor_params = actor_def.init(actor_key, observations)["params"]
        actor = TrainState.create(apply_fn=actor_def.apply, params=actor_params, tx=optimiser)
        critic_base_cls = partial(
            MLP,
            hidden_dims=hidden_dims,
            activate_final=True,
            dropout_rate=dropout_rate,
            use_layer_norm=False,
        )
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_cls = partial(Ensemble, net_cls=critic_cls, num=2)
        critic_def = SequenceMultiplexer(encoder_cls=encoder_cls, network_cls=critic_cls, latent_dim=latent_dim)
        critic_params = critic_def.init(critic_key, observations, actions)["params"]
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.adam(learning_rate=critic_lr),
        )

        target_critic_params = critic_def.init(critic_key, observations, actions)["params"]
        target_critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=target_critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        value_cls = partial(StateValue, base_cls=critic_base_cls)
        value_def = SequenceMultiplexer(encoder_cls=encoder_cls, network_cls=value_cls, latent_dim=latent_dim)
        value_params = value_def.init(value_key, observations)["params"]
        value = TrainState.create(
            apply_fn=value_def.apply,
            params=value_params,
            tx=optax.adam(learning_rate=value_lr),
        )

        return cls(
            rng=rng,
            actor=actor,
            critic=critic,
            target_critic=target_critic,
            value=value,
            expectile=expectile,
            tau=tau,
            discount=discount,
            critic_reduction=critic_reduction,
            A_scaling=A_scaling,
        )

    def update(self, batch: FrozenDict) -> Dict[str, float]:
        new_agent = self
        target_critic = _share_encoder(source=new_agent.critic, target=new_agent.target_critic)
        actor = _share_encoder(source=new_agent.critic, target=new_agent.actor)
        new_agent = new_agent.replace(target_critic=target_critic, actor=actor)
        return IQLLearner.update(new_agent, batch)
