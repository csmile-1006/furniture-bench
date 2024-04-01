from typing import Tuple

import jax
import jax.numpy as jnp

from networks.common import Batch, InfoDict, Model, Params, PRNGKey


def sac_update_actor(
    key: PRNGKey,
    actor: Model,
    critic: Model,
    temp: Model,
    batch: Batch,
    expl_noise: float,
    use_bc: bool,
    bc_weight: float,
    offline_batch_size: int,
) -> Tuple[Model, InfoDict]:
    key, rng = jax.random.split(key)
    key2, rng = jax.random.split(rng)
    key3, rng = jax.random.split(rng)

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
        online_log_probs = dist.log_prob(sampled_actions)

        qs = critic(batch.observations, sampled_actions)
        q = qs.mean(axis=0)
        actor_q_loss = (online_log_probs * temp() - q).mean()

        if use_bc:
            demo_dist, _ = actor.apply(
                actor_params,
                {key: value[:offline_batch_size] for key, value in batch.observations.items()},
                expl_noise,
                training=True,
                rngs={"dropout": key3},
                mutable=actor.extra_variables.keys(),
            )
            offline_log_probs = demo_dist.log_prob(batch.actions[:offline_batch_size])
            bc_loss = -offline_log_probs.mean()
            actor_loss = actor_q_loss + bc_weight * bc_loss

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
                "entropy": -online_log_probs.mean(),
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
                "entropy": -online_log_probs.mean(),
                "updated_states": updated_states,
            }

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    return new_actor, info
