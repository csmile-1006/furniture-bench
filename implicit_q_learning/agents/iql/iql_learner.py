"""Implementations of algorithms for continuous control."""

import functools
from functools import partial
from typing import Dict, Optional, Sequence, Tuple

import gym
import jax
import optax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState

from agents.agent import Agent
from agents.iql.actor_updater import update_actor
from agents.iql.critic_updater import update_q, update_v
from networks import MLP, Ensemble
from networks.distributions import UnitStdNormalPolicy
from networks.values import StateValue, StateActionValue
from data_types import Params, PRNGKey


@functools.partial(jax.jit, static_argnames="critic_reduction")
def _update_jit(
    rng: PRNGKey,
    actor: TrainState,
    critic: TrainState,
    target_critic: TrainState,
    value: TrainState,
    batch: TrainState,
    discount: float,
    tau: float,
    expectile: float,
    A_scaling: float,
    critic_reduction: str,
) -> Tuple[PRNGKey, TrainState, TrainState, Params, TrainState, Dict[str, float]]:
    new_value, value_info = update_v(target_critic, value, batch, expectile, critic_reduction)
    key, rng = jax.random.split(rng)
    new_actor, actor_info = update_actor(key, actor, target_critic, new_value, batch, A_scaling, critic_reduction)

    new_critic, critic_info = update_q(critic, new_value, batch, discount)

    new_target_critic_params = optax.incremental_update(new_critic.params, target_critic.params, tau)
    new_target_critic = target_critic.replace(params=new_target_critic_params)

    return (
        rng,
        new_actor,
        new_critic,
        new_target_critic,
        new_value,
        {**critic_info, **value_info, **actor_info},
    )


class IQLLearner(Agent):
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
    ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1801.01290
        """

        action_dim = action_space.shape[-1]
        observations = observation_space.sample()
        actions = action_space.sample()

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)

        actor_base_cls = partial(MLP, hidden_dims=hidden_dims, activate_final=True, dropout_rate=dropout_rate)
        actor_def = UnitStdNormalPolicy(base_cls=actor_base_cls, action_dim=action_dim)
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
        critic_def = Ensemble(critic_cls, num=2)
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

        value_def = StateValue(base_cls=critic_base_cls)
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
        (
            new_rng,
            new_actor,
            new_critic,
            new_target_critic,
            new_value,
            info,
        ) = _update_jit(
            self.rng,
            self.actor,
            self.critic,
            self.target_critic,
            self.value,
            batch,
            self.discount,
            self.tau,
            self.expectile,
            self.A_scaling,
            self.critic_reduction,
        )

        new_agent = self.replace(
            rng=new_rng, actor=new_actor, critic=new_critic, target_critic=new_target_critic, value=new_value
        )
        info["mse"] = jnp.mean((batch.actions - new_agent.eval_actions(batch.observations)) ** 2)
        return new_agent, info
