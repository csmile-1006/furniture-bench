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
