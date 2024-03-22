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


def get_value(
    key: PRNGKey, actor: Model, critic: Model, observations: jnp.ndarray, num_samples: int, expl_noise: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    policy_actions = actor(observations, expl_noise)

    n_observations = {}
    for k, v in observations.items():
        n_observations[k] = jnp.repeat(v[jnp.newaxis], num_samples, axis=0).reshape(-1, *v.shape[1:])
    q_pi1, q_pi2 = critic(n_observations, policy_actions)

    def get_v(q):
        return jnp.mean(q, axis=0)

    return get_v(q_pi1), get_v(q_pi2)


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
    v1, v2 = get_value(key, actor, critic, batch.observations, 1)
    v = jnp.minimum(v1, v2)

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
        a = q - v
        actor_q_loss = -a.mean()

        if use_td3_bc:
            offline_predicted_actions = predicted_actions[:offline_batch_size]
            offline_mse = (offline_predicted_actions - batch.actions[:offline_batch_size]) ** 2
            bc_loss = offline_mse.mean()
            actor_loss = actor_q_loss + bc_weight * bc_loss

            return actor_loss, {
                # "lamb": lamb,
                "actor_loss": actor_loss,
                "adv_mean": a.mean(),
                "adv_min": a.min(),
                "adv_max": a.max(),
                "adv_std": a.std(),
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
                "adv_mean": a.mean(),
                "adv_min": a.min(),
                "adv_max": a.max(),
                "adv_std": a.std(),
                "updated_states": updated_states,
            }

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    return new_actor, info
