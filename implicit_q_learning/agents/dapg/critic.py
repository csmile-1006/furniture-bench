from typing import Tuple

import jax
import jax.numpy as jnp
from networks.common import Batch, InfoDict, Model, Params, PRNGKey


def target_update(critic: Model, target_critic: Model, tau: float) -> Model:
    new_target_params = jax.tree_map(lambda p, tp: p * tau + tp * (1 - tau), critic.params, target_critic.params)

    return target_critic.replace(params=new_target_params)


def dapg_update_critic(
    key: PRNGKey,
    actor: Model,
    critic: Model,
    target_critic: Model,
    temp: Model,
    batch: Batch,
    discount: float,
    temperature: float,
    backup_entropy: bool,
) -> Tuple[Model, InfoDict]:
    dist = actor(batch.next_observations, temperature)
    next_actions = dist.sample(seed=key)
    next_q1, next_q2 = target_critic(batch.next_observations, next_actions)
    next_q = jnp.minimum(next_q1, next_q2)

    target_q = batch.rewards + discount * batch.masks * next_q

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        (q1, q2), updated_states = critic.apply(
            critic_params,
            batch.observations,
            batch.actions,
            training=True,
            rngs={"dropout": key},
            mutable=critic.extra_variables.keys(),
        )
        critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
        return (
            critic_loss,
            {
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
                "updated_states": updated_states,
            },
        )

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info
