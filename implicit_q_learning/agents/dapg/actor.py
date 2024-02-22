from typing import Tuple

import jax.numpy as jnp

from networks.common import Batch, InfoDict, Model, Params, PRNGKey


def get_value(
    key: PRNGKey, actor: Model, critic: Model, observations: jnp.ndarray, num_samples: int, temperature: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    dist = actor(observations, temperature)

    policy_actions = dist.sample(seed=key)

    n_observations = {}
    for k, v in observations.items():
        n_observations[k] = jnp.repeat(v[jnp.newaxis], num_samples, axis=0).reshape(-1, *v.shape[1:])
    q_pi1, q_pi2 = critic(n_observations, policy_actions)

    def get_v(q):
        return jnp.mean(q, axis=0)

    return get_v(q_pi1), get_v(q_pi2)


def dapg_update_actor(
    key: PRNGKey,
    actor: Model,
    critic: Model,
    batch: Batch,
    lambda_1: float,
    lambda_2: float,
    step: int,
    temperature: float,
) -> Tuple[Model, InfoDict]:
    v1, v2 = get_value(key, actor, critic, batch.observations, 1, temperature)
    v = jnp.minimum(v1, v2)

    dist = actor(batch.next_observations, temperature)
    next_actions = dist.sample(seed=key)
    next_q1, next_q2 = critic(batch.next_observations, next_actions)
    next_q = jnp.minimum(next_q1, next_q2)

    next_v1, next_v2 = get_value(key, actor, critic, batch.observations, 1, temperature)
    next_v = jnp.minimum(next_v1, next_v2)

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist, updated_states = actor.apply(
            actor_params,
            batch.observations,
            temperature,
            training=True,
            rngs={"dropout": key},
            mutable=actor.extra_variables.keys(),
        )
        lim = 1 - 1e-5
        actions = jnp.clip(batch.actions, -lim, lim)
        log_probs = dist.log_prob(actions)

        q1, q2 = critic(batch.observations, actions)
        q = jnp.minimum(q1, q2)
        a = q - v
        next_a = next_q - next_v

        offline_log_probs, online_log_probs = log_probs[::2], log_probs[1::2]
        online_a = a[1::2]
        online_next_a = next_a[1::2]

        offline_weight = lambda_1 * (lambda_2**step) * online_next_a.max()
        actor_q_loss = -online_a * online_log_probs
        bc_loss = -offline_log_probs
        actor_loss = actor_q_loss.sum() + offline_weight * bc_loss.sum()
        # a = q - v

        return (
            actor_loss,
            {
                "actor_loss_mean": actor_loss,
                "actor_loss_min": a.min(),
                "actor_loss_max": a.max(),
                "actor_loss_std": a.std(),
                "updated_states": updated_states,
            },
        )

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    return new_actor, info
