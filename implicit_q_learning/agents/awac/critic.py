from typing import Tuple

import jax
import jax.numpy as jnp
from networks.common import Batch, InfoDict, Model, Params, PRNGKey
from networks.value_net import subsample_critic_ensemble


def get_value(
    key: PRNGKey, actor: Model, critic: Model, batch: Batch, num_samples: int, expl_noise: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    dist = actor(batch.observations, expl_noise)

    rng = key
    key, rng = jax.random.split(rng)
    policy_actions = dist.sample(seed=key)

    n_observations = {}
    for k, v in batch.observations.items():
        n_observations[k] = jnp.repeat(v[jnp.newaxis], num_samples, axis=0).reshape(-1, *v.shape[1:])
    qs = critic(n_observations, policy_actions)

    def get_v(q):
        return jnp.mean(q, axis=0)

    return jnp.asarray([get_v(elem) for elem in qs])


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
    expl_noise: float,
    num_qs: int,
    num_min_qs: int,
    backup_entropy: bool,
) -> Tuple[Model, InfoDict]:
    dist = actor(batch.next_observations, expl_noise)

    rng = key

    key, rng = jax.random.split(rng)
    next_actions = dist.sample(seed=key)

    key, rng = jax.random.split(rng)
    target_params = subsample_critic_ensemble(key, target_critic.params, num_min_qs, num_qs)

    key, rng = jax.random.split(rng)
    next_qs = target_critic.apply(
        {"params": target_params, **target_critic.extra_variables},
        batch.next_observations,
        next_actions,
        rngs={"dropout": key},
    )
    next_q = next_qs.min(axis=0)

    target_q = batch.rewards + discount * batch.masks * next_q

    if backup_entropy:
        next_log_probs = dist.log_prob(next_actions)
        target_q -= discount * batch.masks * temp() * next_log_probs

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
