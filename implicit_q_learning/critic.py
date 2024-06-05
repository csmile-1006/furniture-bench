from typing import Tuple

import jax
import jax.numpy as jnp
from common import Batch, InfoDict, Model, Params, PRNGKey


def subsample_ensemble(key: jax.random.PRNGKey, params, num_sample: int, num_qs: int):
    if num_sample is not None:
        all_indx = jnp.arange(0, num_qs)
        indx = jax.random.choice(key, a=all_indx, shape=(num_sample,), replace=False)

        if "VmapCritic_0" in params:
            ens_params = jax.tree_util.tree_map(lambda param: param[indx], params["VmapCritic_0"])
            params = params.copy(add_or_replace={"VmapCritic_0": ens_params})
        else:
            params = jax.tree_util.tree_map(lambda param: param[indx], params)
    return params


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
            "v": v.mean(),
        }

    new_value, info = value.apply_gradient(value_loss_fn)

    return new_value, info


def update_q(critic: Model, target_value: Model, batch: Batch, discount: float) -> Tuple[Model, InfoDict]:
    next_v = target_value(batch.next_observations)

    target_q = batch.rewards + discount * batch.masks * next_v

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q1, q2 = critic.apply({"params": critic_params}, batch.observations, batch.actions)
        critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
        return critic_loss, {"critic_loss": critic_loss, "q1": q1.mean(), "q2": q2.mean()}

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info


def update_value_critic(
    key: PRNGKey,
    critic: Model,
    value: Model,
    target_critic: Model,
    batch: Batch,
    discount: float,
    expectile: float,
    num_qs: 10,
    num_min_qs: 2,
) -> Tuple[Model, Model, InfoDict]:
    # critic loss function
    next_v = value(batch.next_observations)
    target_q = batch.rewards + discount * batch.masks * next_v

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        qs = critic.apply({"params": critic_params}, batch.observations, batch.actions)
        critic_loss = ((qs - target_q) ** 2).mean()
        return critic_loss, {"critic_loss": critic_loss, "qs": qs.mean()}

    # value loss function
    actions = batch.actions
    target_params = subsample_ensemble(key, target_critic.params, num_min_qs, num_qs)
    qs = target_critic.apply({"params": target_params}, batch.observations, actions)
    q = jnp.min(qs, axis=0)

    def value_loss_fn(value_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        v = value.apply({"params": value_params}, batch.observations)
        value_loss = loss(q - v, expectile).mean()
        return value_loss, {
            "value_loss": value_loss,
            "v": v.mean(),
        }

    new_critic, critic_info = critic.apply_gradient(critic_loss_fn)
    new_value, value_info = value.apply_gradient(value_loss_fn)

    return new_critic, new_value, {**critic_info, **value_info}
