from typing import Tuple

import jax
import jax.numpy as jnp

from networks.common import Batch, InfoDict, Model, Params, PRNGKey


def get_value(
    key: PRNGKey, actor: Model, critic: Model, observations: jnp.ndarray, num_samples: int, expl_noise: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    dist = actor(observations, expl_noise)

    key, rng = jax.random.split(key)
    policy_actions = dist.sample(seed=key)

    n_observations = {}
    for k, v in observations.items():
        n_observations[k] = jnp.repeat(v[jnp.newaxis], num_samples, axis=0).reshape(-1, *v.shape[1:])
    qs = critic(n_observations, policy_actions)

    def get_v(q):
        return jnp.mean(q, axis=0)

    return jnp.asarray([get_v(elem) for elem in qs])


def td3_update_actor(
    key: PRNGKey,
    actor: Model,
    critic: Model,
    batch: Batch,
    alpha: float,
    expl_noise: float,
    use_td3_bc: bool,
    bc_weight: float,
    offline_batch_size: int,
) -> Tuple[Model, InfoDict]:
    key, rng = jax.random.split(key)
    key2, rng = jax.random.split(rng)

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist, updated_states = actor.apply(
            actor_params,
            batch.observations,
            expl_noise,
            training=True,
            rngs={"dropout": key},
            mutable=actor.extra_variables.keys(),
        )
        sampled_actions = dist.sample(seed=key2)

        qs = critic(batch.observations, sampled_actions)
        q = qs.min(axis=0)
        actor_q_loss = -q.mean()

        if use_td3_bc:
            log_probs = dist.log_prob(batch.actions)
            offline_log_probs = log_probs[:offline_batch_size]
            bc_loss = -offline_log_probs.mean()
            actor_loss = actor_q_loss + bc_weight * bc_loss

            # lamb = alpha / abs(data_q.mean())
            # actor_loss = lamb * actor_q_loss + bc_loss

            return actor_loss, {
                "actor_loss": actor_loss,
                "actor_q_mean": -q.mean(),
                "actor_q_min": -q.min(),
                "actor_q_max": -q.max(),
                "actor_q_std": -q.std(),
                "bc_loss": -offline_log_probs.mean(),
                "bc_loss_min": -offline_log_probs.min(),
                "bc_loss_max": -offline_log_probs.max(),
                "bc_loss_std": -offline_log_probs.std(),
                "updated_states": updated_states,
            }
        else:
            actor_loss = actor_q_loss
            return actor_loss, {
                "actor_loss": actor_loss,
                "actor_q_mean": -q.mean(),
                "actor_q_min": -q.min(),
                "actor_q_max": -q.max(),
                "actor_q_std": -q.std(),
                "updated_states": updated_states,
            }

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    return new_actor, info
