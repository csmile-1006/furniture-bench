from typing import Tuple

import jax.numpy as jnp

from networks.common import Batch, InfoDict, Model, Params, PRNGKey


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
    # v1, v2 = get_value(key, actor, critic, batch, num_samples, temperature)
    # v = jnp.minimum(v1, v2)

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
        actor_q_loss = -q
        bc_loss = -log_probs
        actor_loss = actor_q_loss.mean() + lambda_1 * (lambda_2**step) * bc_loss.mean()
        # a = q - v

        # # we could have used exp(a / beta) here but
        # # exp(a / beta) is unbiased but high variance,
        # # softmax(a / beta) is biased but lower variance.
        # # sum() instead of mean(), because it should be multiplied by batch size.
        # actor_loss = -(jax.nn.softmax(a / beta) * log_probs).sum()

        return (
            actor_loss,
            {
                "actor_loss_mean": actor_loss,
                "actor_q_loss_mean": actor_q_loss.mean(),
                "actor_q_loss_min": actor_q_loss.min(),
                "actor_q_loss_max": actor_q_loss.max(),
                "actor_q_loss_std": actor_q_loss.std(),
                "bc_loss_mean": bc_loss.mean(),
                "bc_loss_min": bc_loss.min(),
                "bc_loss_max": bc_loss.max(),
                "bc_loss_std": bc_loss.std(),
                "updated_states": updated_states,
            },
        )

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    return new_actor, info
