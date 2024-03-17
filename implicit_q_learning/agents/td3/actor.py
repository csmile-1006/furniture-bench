from typing import Tuple

import jax.numpy as jnp

from networks.common import Batch, InfoDict, Model, Params, PRNGKey


def td3_update_actor(
    key: PRNGKey,
    actor: Model,
    critic: Model,
    batch: Batch,
    alpha: float,
    expl_noise: float,
    use_td3_bc: bool,
    temperature: float,
    bc_weight: float,
) -> Tuple[Model, InfoDict]:
    # data_q1, data_q2 = critic(batch.observations, batch.actions)
    # data_q = jnp.minimum(data_q1, data_q2)

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
        q1, q2 = critic(batch.observations, sampled_actions)
        q = jnp.minimum(q1, q2)
        actor_q_loss = -q.mean()

        if use_td3_bc:
            log_probs = dist.log_prob(jnp.clip(batch.actions, -1 + 1e-5, 1 - 1e-5))
            offline_log_probs = log_probs[::2]
            bc_loss = -offline_log_probs.mean()
            # bc_loss = -log_probs.mean()
            actor_loss = actor_q_loss + bc_weight * bc_loss

            # lamb = alpha / abs(data_q.mean())
            # actor_loss = lamb * actor_q_loss + bc_loss

            return actor_loss, {
                # "lamb": lamb,
                "actor_loss": actor_loss,
                "actor_q_loss": actor_q_loss,
                "actor_q_loss_min": -q.min(),
                "actor_q_loss_max": -q.max(),
                "actor_q_loss_std": -q.std(),
                "bc_loss": bc_loss,
                "bc_loss_min": -offline_log_probs.min(),
                "bc_loss_max": -offline_log_probs.max(),
                "bc_loss_std": -offline_log_probs.std(),
                "updated_states": updated_states,
            }
        else:
            actor_loss = actor_q_loss
            return actor_loss, {
                "actor_loss": actor_loss,
                "actor_loss_min": -q.min(),
                "actor_loss_max": -q.max(),
                "actor_loss_std": -q.std(),
                "updated_states": updated_states,
            }

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    return new_actor, info
