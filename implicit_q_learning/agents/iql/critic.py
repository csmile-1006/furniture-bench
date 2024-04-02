from typing import Tuple

import jax.numpy as jnp

from networks.common import Batch, InfoDict, Model, Params, PRNGKey


def loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)


def update_v(key: PRNGKey, critic: Model, value: Model, batch: Batch, expectile: float) -> Tuple[Model, InfoDict]:
    actions = batch.actions
    q1, q2 = critic(batch.observations, actions)
    q = jnp.minimum(q1, q2)

    def value_loss_fn(value_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        v, updated_states = value.apply(
            value_params, batch.observations, training=True, rngs={"dropout": key}, mutable=value.extra_variables.keys()
        )
        vl = loss(q - v, expectile)
        value_loss = vl.mean()
        return (
            value_loss,
            {
                "value_loss_mean": vl.mean(),
                "value_loss_std": vl.std(),
                "value_loss_min": vl.min(),
                "value_loss_max": vl.max(),
                "v_mean": v.mean(),
                "v_std": v.std(),
                "v_min": v.min(),
                "v_max": v.max(),
                "updated_states": updated_states,
            },
        )

    new_value, info = value.apply_gradient(value_loss_fn)

    return new_value, info


def update_q(key: PRNGKey, critic: Model, target_value: Model, batch: Batch, discount: float) -> Tuple[Model, InfoDict]:
    next_v = target_value(batch.next_observations)

    target_q = batch.rewards + discount * batch.masks * next_v

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        qs, updated_states = critic.apply(
            critic_params,
            batch.observations,
            batch.actions,
            training=True,
            rngs={"dropout": key},
            mutable=critic.extra_variables.keys(),
        )
        critic_loss = ((qs - target_q) ** 2).mean()
        return (
            critic_loss,
            {
                "critic_loss": critic_loss,
                "q_mean": qs.mean(),
                "q_min": qs.min(),
                "q_max": qs.max(),
                "q_std": qs.std(),
                "target_q_mean": target_q.mean(),
                "target_q_std": target_q.std(),
                "target_q_min": target_q.min(),
                "target_q_max": target_q.max(),
                "updated_states": updated_states,
            },
        )

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info
