from typing import Tuple

import jax.numpy as jnp

from networks.common import Batch, InfoDict, Model, Params, PRNGKey


def bc_update_actor(key: PRNGKey, actor: Model, batch: Batch) -> Tuple[Model, InfoDict]:
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        actions, updated_states = actor.apply(
            actor_params, batch.observations, training=True, rngs={"dropout": key}, mutable=actor.extra_variables.keys()
        )
        a = (actions - batch.actions) ** 2
        actor_loss = a.mean()

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


def td3_update_actor(
    key: PRNGKey,
    actor: Model,
    critic: Model,
    batch: Batch,
    alpha: float,
    use_td3_bc: bool,
    bc_weight: float,
    offline_batch_size: int,
) -> Tuple[Model, InfoDict]:
    # data_q1, data_q2 = critic(batch.observations, batch.actions)
    # data_q = jnp.minimum(data_q1, data_q2)

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        predicted_actions, updated_states = actor.apply(
            actor_params,
            batch.observations,
            training=True,
            rngs={"dropout": key},
            mutable=actor.extra_variables.keys(),
        )
        q1, q2 = critic(batch.observations, predicted_actions)
        q = jnp.minimum(q1, q2)
        actor_q_loss = -q.mean()

        if use_td3_bc:
            offline_predicted_actions = predicted_actions[:offline_batch_size]
            offline_mse = (offline_predicted_actions - batch.actions[:offline_batch_size]) ** 2
            bc_loss = offline_mse.mean()
            actor_loss = actor_q_loss + bc_weight * bc_loss

            return actor_loss, {
                # "lamb": lamb,
                "actor_loss": actor_loss,
                "actor_q_loss": actor_q_loss,
                "actor_q_loss_min": -q.min(),
                "actor_q_loss_max": -q.max(),
                "actor_q_loss_std": -q.std(),
                "bc_loss": bc_loss,
                "bc_loss_min": offline_mse.min(),
                "bc_loss_max": offline_mse.max(),
                "bc_loss_std": offline_mse.std(),
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
