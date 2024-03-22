from typing import Tuple

import jax.numpy as jnp

from networks.common import Batch, InfoDict, Model, Params, PRNGKey


def get_value(
    key: PRNGKey, actor: Model, critic: Model, observations: jnp.ndarray, num_samples: int, expl_noise: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    dist = actor(observations, expl_noise)

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
    expl_noise: float,
    offline_batch_size: float,
) -> Tuple[Model, InfoDict]:
    v1, v2 = get_value(key, actor, critic, batch.observations, 1, expl_noise)
    v = jnp.minimum(v1, v2)

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist, updated_states = actor.apply(
            actor_params,
            batch.observations,
            expl_noise,
            training=True,
            rngs={"dropout": key},
            mutable=actor.extra_variables.keys(),
        )
        sampled_actions = dist.sample(seed=key)
        actions = batch.actions

        q1, q2 = critic(batch.observations, sampled_actions)
        q = jnp.minimum(q1, q2)
        a = q - v

        online_a = a[offline_batch_size:]
        actor_q_loss = -online_a

        log_probs = dist.log_prob(actions)
        offline_log_probs = log_probs[:offline_batch_size]
        offline_weight = lambda_1 * (lambda_2**step) * online_a.max()
        bc_loss = -offline_weight * offline_log_probs

        actor_loss = actor_q_loss.mean() + bc_loss.mean()

        return (
            actor_loss,
            {
                "bc_loss_mean": bc_loss.mean(),
                "bc_loss_min": bc_loss.min(),
                "bc_loss_max": bc_loss.max(),
                "bc_loss_std": bc_loss.std(),
                "actor_loss": actor_loss,
                "actor_q_loss_mean": actor_q_loss.mean(),
                "actor_q_loss_min": actor_q_loss.min(),
                "actor_q_loss_max": actor_q_loss.max(),
                "actor_q_loss_std": actor_q_loss.std(),
                "offline_weight": offline_weight,
                "updated_states": updated_states,
            },
        )

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    return new_actor, info
