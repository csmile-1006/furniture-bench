from typing import Tuple

import jax.numpy as jnp

from common import Batch, InfoDict, Model, Params, PRNGKey


def update(
    key: PRNGKey, actor: Model, critic: Model, value: Model, batch: Batch, temperature: float
) -> Tuple[Model, InfoDict]:
    v = value(batch.observations)

    q1, q2 = critic(batch.observations, batch.actions)
    q = jnp.minimum(q1, q2)
    exp_a = jnp.exp((q - v) * temperature)
    exp_a = jnp.clip(exp_a, -100.0, 100.0)

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist, updated_states = actor.apply(
            actor_params, batch.observations, training=True, rngs={"dropout": key}, mutable=actor.extra_variables.keys()
        )
        log_probs = dist.log_prob(batch.actions)
        a = -(exp_a * log_probs)
        actor_loss = a.mean()
        adv = q - v

        return (
            actor_loss,
            {
                "actor_loss_mean": actor_loss,
                "actor_loss_min": a.min(),
                "actor_loss_max": a.max(),
                "actor_loss_std": a.std(),
                "adv_mean": adv.mean(),
                "adv_min": adv.min(),
                "adv_max": adv.max(),
                "adv_std": adv.std(),
                "updated_states": updated_states,
                "log_probs_mean": log_probs.mean(),
            },
        )

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    return new_actor, info
