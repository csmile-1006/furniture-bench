from typing import Tuple

import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

import policy
from common import Batch, InfoDict, Model, Params, PRNGKey


def update(
    key: PRNGKey, actor: Model, critic: Model, value: Model, batch: Batch, temperature: float
) -> Tuple[Model, InfoDict]:
    v = value(batch.observations)

    qs = critic(batch.observations, batch.actions)
    q = qs.min(axis=0)
    exp_a = jnp.exp((q - v) * temperature)
    exp_a = jnp.minimum(exp_a, 100.0)

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply({"params": actor_params}, batch.observations, training=True, rngs={"dropout": key})
        log_probs = dist.log_prob(batch.actions)
        actor_loss = -(exp_a * log_probs).mean()

        return actor_loss, {"actor_loss": actor_loss, "adv": q - v}

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    return new_actor, info


def ddpg_bc_update(
    key: PRNGKey,
    actor: Model,
    critic: Model,
    batch: Batch,
    alpha: float
) -> Tuple[Model, InfoDict]:
    # Compute the critic value for the current policy action
    key, policy_actions = policy.sample_actions(key, actor.apply_fn, actor.params, batch.observations)
    qs = critic(batch.observations, policy_actions)
    q = qs.min(axis=0)

    def actor_loss_fn(actor_params: FrozenDict) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply({"params": actor_params}, batch.observations, training=True, rngs={"dropout": key})
        log_probs = dist.log_prob(batch.actions)
        actor_loss = -(q + alpha * log_probs).mean()

        return actor_loss, {"actor_loss": actor_loss, "Q_value": q.mean(), "log_probs": log_probs.mean()}

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    return new_actor, info
