"""Implementations of algorithms for continuous control."""

from typing import Optional, Sequence, Tuple
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

import policy
import value_net
from actor import update as awr_update_actor
from common import Batch, InfoDict, Model, PRNGKey, TransformerEncoder, CrossAttnTransformerEncoder
from critic import update_q, update_v


def target_update(critic: Model, target_critic: Model, tau: float) -> Model:
    new_target_params = jax.tree_map(lambda p, tp: p * tau + tp * (1 - tau), critic.params, target_critic.params)

    return target_critic.replace(params=new_target_params)


def _share_encoder(source, target):
    replacers = {}

    for k, v in source.params.items():
        if "encoder" in k:
            replacers[k] = v

    # Use critic conv layers in actor:
    new_params = target.params.copy(add_or_replace=replacers)
    return target.replace(params=new_params)


@partial(jax.jit, static_argnames=("utd_ratio"))
def _update_jit(
    rng: PRNGKey,
    actor: Model,
    critic: Model,
    value: Model,
    target_critic: Model,
    batch: Batch,
    discount: float,
    tau: float,
    expectile: float,
    temperature: float,
    utd_ratio: int,
) -> Tuple[PRNGKey, Model, Model, Model, Model, Model, InfoDict]:
    # def slice(x, i):
    #     assert x.shape[0] % utd_ratio == 0
    #     batch_size = x.shape[0] // utd_ratio
    #     return x[batch_size * i : batch_size * (i + 1)]

    # for i in range(utd_ratio):
    #     mini_batch = Batch(
    #         observations={key: slice(batch.observations[key], i) for key in batch.observations},
    #         actions=slice(batch.actions, i),
    #         rewards=slice(batch.rewards, i),
    #         masks=slice(batch.masks, i),
    #         next_observations={key: slice(batch.next_observations[key], i) for key in batch.next_observations},
    #     )
    #     new_value, value_info = update_v(target_critic, value, mini_batch, expectile)
    #     new_critic, critic_info = update_q(critic, new_value, mini_batch, discount)
    # key, rng = jax.random.split(rng)
    # new_actor, actor_info = awr_update_actor(key, actor, target_critic, new_value, mini_batch, temperature)

    # new_target_critic = target_update(new_critic, target_critic, tau)
    actor = _share_encoder(source=critic, target=actor)
    value = _share_encoder(source=critic, target=value)
    new_value, value_info = update_v(target_critic, value, batch, expectile)
    key, rng = jax.random.split(rng)
    new_actor, actor_info = awr_update_actor(key, actor, target_critic, new_value, batch, temperature)

    new_critic, critic_info = update_q(critic, new_value, batch, discount)
    new_target_critic = target_update(new_critic, target_critic, tau)
    return (
        rng,
        new_actor,
        new_critic,
        new_value,
        new_target_critic,
        {**critic_info, **value_info, **actor_info},
    )


class Learner(object):
    def __init__(
        self,
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        actor_lr: float = 3e-4,
        value_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        emb_dim: int = 256,
        depth: int = 2,
        num_heads: int = 8,
        discount: float = 0.99,
        tau: float = 0.005,
        expectile: float = 0.8,
        temperature: float = 0.1,
        dropout_rate: Optional[float] = None,
        max_steps: Optional[int] = None,
        opt_decay_schedule: str = "cosine",
        critic_layer_norm: bool = False,
        obs_keys: Sequence[str] = ("image1", "image2"),
    ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1801.01290
        """

        self.expectile = expectile
        self.tau = tau
        self.discount = discount
        self.temperature = temperature

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)

        if len(observations["image1"].shape) == 3 or len(observations["image1"].shape) == 2:
            observations["image1"] = observations["image1"][np.newaxis]
            observations["image2"] = observations["image2"][np.newaxis]
        if len(observations["robot_state"].shape) == 2:
            observations["robot_state"] = observations["robot_state"][np.newaxis]
        if observations.get("text_feature") is not None and len(observations["text_feature"].shape) == 2:
            observations["text_feature"] = observations["text_feature"][np.newaxis]

        if "text_feature" in obs_keys:
            encoder_cls = partial(
                CrossAttnTransformerEncoder,
                emb_dim=emb_dim,
                depth=depth,
                num_heads=num_heads,
                att_drop=0.0 if dropout_rate is None else dropout_rate,
                drop=0.0 if dropout_rate is None else dropout_rate,
            )
        else:
            encoder_cls = partial(
                TransformerEncoder,
                emb_dim=emb_dim,
                depth=depth,
                num_heads=num_heads,
                att_drop=0.0 if dropout_rate is None else dropout_rate,
                drop=0.0 if dropout_rate is None else dropout_rate,
            )

        action_dim = actions.shape[-1]
        actor_def = policy.NormalTanhPolicy(
            hidden_dims,
            emb_dim,
            action_dim,
            log_std_scale=1e-3,
            log_std_min=-5.0,
            dropout_rate=dropout_rate,
            state_dependent_std=False,
            tanh_squash_distribution=False,
            encoder_cls=encoder_cls,
            obs_keys=obs_keys,
        )

        if opt_decay_schedule == "cosine":
            schedule_fn = optax.cosine_decay_schedule(-actor_lr, max_steps)
            optimiser = optax.chain(optax.scale_by_adam(), optax.scale_by_schedule(schedule_fn))
        else:
            optimiser = optax.adam(learning_rate=actor_lr)

        actor = Model.create(actor_def, inputs=[actor_key, observations], tx=optimiser)

        critic_def = value_net.DoubleCritic(
            hidden_dims, emb_dim, encoder_cls=encoder_cls, critic_layer_norm=critic_layer_norm, obs_keys=obs_keys
        )
        critic = Model.create(
            critic_def,
            inputs=[critic_key, observations, actions],
            tx=optax.adam(learning_rate=critic_lr),
        )

        value_def = value_net.ValueCritic(
            hidden_dims, emb_dim, encoder_cls=encoder_cls, critic_layer_norm=critic_layer_norm, obs_keys=obs_keys
        )
        value = Model.create(
            value_def,
            inputs=[value_key, observations],
            tx=optax.adam(learning_rate=value_lr),
        )

        target_critic = Model.create(critic_def, inputs=[critic_key, observations, actions])

        self.actor = actor
        self.critic = critic
        self.value = value
        self.target_critic = target_critic
        self.rng = rng

    def sample_actions(self, observations: np.ndarray, temperature: float = 1.0) -> jnp.ndarray:
        rng, actions = policy.sample_actions(
            self.rng, self.actor.apply_fn, self.actor.params, observations, temperature
        )
        self.rng = rng

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def update(self, batch: Batch, utd_ratio: int = 1) -> InfoDict:
        (
            new_rng,
            new_actor,
            new_critic,
            new_value,
            new_target_critic,
            info,
        ) = _update_jit(
            self.rng,
            self.actor,
            self.critic,
            self.value,
            self.target_critic,
            batch,
            self.discount,
            self.tau,
            self.expectile,
            self.temperature,
            utd_ratio,
        )

        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.value = new_value
        self.target_critic = new_target_critic

        info["mse"] = jnp.mean((batch.actions - self.sample_actions(batch.observations, temperature=0.0)) ** 2)
        return info

    def save(self, ckpt_dir, step):
        path = f"{ckpt_dir}/{step}_actor"
        self.actor.save(path)
        path = f"{ckpt_dir}/{step}_critic"
        self.critic.save(path)
        path = f"{ckpt_dir}/{step}_target_critic"
        self.target_critic.save(path)
        path = f"{ckpt_dir}/{step}_value"
        self.value.save(path)

    def load(self, ckpt_dir, step):
        path = f"{ckpt_dir}/{step}_actor"
        self.actor = self.actor.load(path)
        path = f"{ckpt_dir}/{step}_critic"
        self.critic = self.critic.load(path)
        path = f"{ckpt_dir}/{step}_target_critic"
        self.target_critic = self.target_critic.load(path)
        path = f"{ckpt_dir}/{step}_value"
        self.value = self.value.load(path)
