from typing import Tuple

import jax
import jax.numpy as jnp
from networks.common import Batch, InfoDict, Model, Params, PRNGKey


def get_value(
    key: PRNGKey, actor: Model, critic: Model, batch: Batch, num_samples: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    dist = actor(batch.observations)

    policy_actions = dist.sample(seed=key, sample_shape=[num_samples])

    n_observations = jnp.repeat(batch.observations[jnp.newaxis], num_samples, axis=0)
    q_pi1, q_pi2 = critic(n_observations, policy_actions)

    def get_v(q):
        return jnp.mean(q, axis=0)

    return get_v(q_pi1), get_v(q_pi2)


def target_update(critic: Model, target_critic: Model, tau: float) -> Model:
    new_target_params = jax.tree_multimap(lambda p, tp: p * tau + tp * (1 - tau), critic.params, target_critic.params)

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
        q1, q2 = critic.apply_fn({"params": critic_params}, batch.observations, batch.actions)
        critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
        return critic_loss, {"critic_loss": critic_loss, "q1": q1.mean(), "q2": q2.mean()}

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info
