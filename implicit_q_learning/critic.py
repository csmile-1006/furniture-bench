from typing import Tuple

import jax.numpy as jnp

from common import Batch, InfoDict, Model, Params


def loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)


def update_v(critic: Model, value: Model, batch: Batch, expectile: float) -> Tuple[Model, InfoDict]:
    actions = batch.actions
    q1, q2 = critic(batch.observations, actions)
    q = jnp.minimum(q1, q2)

    def value_loss_fn(value_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        v = value.apply({"params": value_params}, batch.observations)
        value_loss = loss(q - v, expectile).mean()
        return value_loss, {
            "value_loss": value_loss,
            "v_mean": v.mean(),
            "v_std": v.std(),
            "v_min": v.min(),
            "v_max": v.max(),
        }

    new_value, info = value.apply_gradient(value_loss_fn)

    return new_value, info


def update_q(critic: Model, target_value: Model, batch: Batch, discount: float) -> Tuple[Model, InfoDict]:
    next_v = target_value(batch.next_observations)

    target_q = batch.rewards + discount * batch.masks * next_v

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q1, q2 = critic.apply({"params": critic_params}, batch.observations, batch.actions)
        critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
        return critic_loss, {
            "critic_loss": critic_loss,
            "q1_mean": q1.mean(),
            "q1_min": q1.min(),
            "q1_max": q1.max(),
            "q1_std": q1.std(),
            "q2_mean": q2.mean(),
            "q2_min": q2.min(),
            "q2_max": q2.max(),
            "q2_std": q2.std(),
            "target_q_mean": target_q.mean(),
            "target_q_std": target_q.std(),
            "target_q_min": target_q.min(),
            "target_q_max": target_q.max(),
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info
