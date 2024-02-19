from typing import Tuple

import jax
import jax.numpy as jnp
from networks.common import Batch, InfoDict, Model, Params, PRNGKey


def get_value(
    key: PRNGKey, actor: Model, critic: Model, batch: Batch, num_samples: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    dist = actor(batch.observations)

    policy_actions = dist.sample(seed=key)

    n_observations = {}
    for k, v in batch.observations.items():
        n_observations[k] = jnp.repeat(v[jnp.newaxis], num_samples, axis=0).reshape(-1, *v.shape[1:])
    q_pi1, q_pi2 = critic(n_observations, policy_actions)

    def get_v(q):
        return jnp.mean(q, axis=0)

    return get_v(q_pi1), get_v(q_pi2)


def target_update(critic: Model, target_critic: Model, tau: float) -> Model:
    new_target_params = jax.tree_map(lambda p, tp: p * tau + tp * (1 - tau), critic.params, target_critic.params)

    return target_critic.replace(params=new_target_params)


def awac_update_critic(
    key: PRNGKey,
    actor: Model,
    critic: Model,
    target_critic: Model,
    temp: Model,
    batch: Batch,
    discount: float,
    backup_entropy: bool,
) -> Tuple[Model, InfoDict]:
    dist = actor(batch.next_observations)
    next_actions = dist.sample(seed=key)
    next_log_probs = dist.log_prob(next_actions)
    next_q1, next_q2 = target_critic(batch.next_observations, next_actions)
    next_q = jnp.minimum(next_q1, next_q2)

    target_q = batch.rewards + discount * batch.masks * next_q

    if backup_entropy:
        target_q -= discount * batch.masks * temp() * next_log_probs

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
