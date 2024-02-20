from typing import Tuple

import jax
import jax.numpy as jnp

from networks.common import Batch, InfoDict, Model, Params, PRNGKey
from agents.awac.critic import get_value


def awac_update_actor(
    key: PRNGKey, actor: Model, critic: Model, batch: Batch, num_samples: int, beta: float, temperature: float
) -> Tuple[Model, InfoDict]:
    v1, v2 = get_value(key, actor, critic, batch, num_samples, temperature)
    v = jnp.minimum(v1, v2)

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

        # we could have used exp(a / beta) here but
        # exp(a / beta) is unbiased but high variance,
        # softmax(a / beta) is biased but lower variance.
        # sum() instead of mean(), because it should be multiplied by batch size.
        actor_loss = -(jax.nn.softmax(a / beta) * log_probs).sum()

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
