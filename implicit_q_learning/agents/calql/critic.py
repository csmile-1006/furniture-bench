from typing import Tuple

import jax
import jax.numpy as jnp
from networks.common import Batch, InfoDict, Model, Params, PRNGKey
from networks.value_net import subsample_critic_ensemble


def target_update(critic: Model, target_critic: Model, tau: float) -> Model:
    new_target_params = jax.tree_map(lambda p, tp: p * tau + tp * (1 - tau), critic.params, target_critic.params)

    return target_critic.replace(params=new_target_params)


def extend_and_repeat(tensor, axis, repeat):
    return jnp.repeat(jnp.expand_dims(tensor, axis), repeat, axis=axis)


def calql_update_critic(
    key: PRNGKey,
    actor: Model,
    critic: Model,
    target_critic: Model,
    temp: Model,
    batch: Batch,
    discount: float,
    num_qs: int,
    num_min_qs: int,
    expl_noise: float,
    backup_entropy: bool,
    cql_n_actions: int,
    cql_importance_sample: bool,
    cql_temp: float,
    cql_max_target_backup: bool,
    cql_min_q_weight: float,
    enable_calql: bool,
) -> Tuple[Model, InfoDict]:
    rng = key

    key, rng = jax.random.split(rng)
    target_params = subsample_critic_ensemble(key, target_critic.params, num_min_qs, num_qs)

    key, key2, key3, key4, key5, key6, key7, key8, key9, rng = jax.random.split(rng, 10)
    batch_size, action_dim = batch.actions.shape[0], batch.actions.shape[-1]

    def compute_action_and_log_prob(key, actor, obs, cql_n_actions):
        repeated_obs = {
            k: extend_and_repeat(val, 1, cql_n_actions).reshape(batch_size * cql_n_actions, *val.shape[1:])
            for k, val in obs.items()
        }
        dist = actor(repeated_obs, expl_noise)
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        return repeated_obs, actions, log_probs

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        if cql_max_target_backup:
            next_obs, next_actions, next_log_probs = compute_action_and_log_prob(
                key, actor, batch.next_observations, cql_n_actions
            )
            next_qs = target_critic.apply(
                {"params": target_params, **target_critic.extra_variables},
                next_obs,
                next_actions,
                rngs={"dropout": key2},
            )
            next_qs = next_qs.min(axis=0)
            next_qs = next_qs.reshape(batch_size, cql_n_actions)
            next_log_probs = next_log_probs.reshape(batch_size, cql_n_actions)
            max_target_indices = jnp.expand_dims(jnp.argmax(next_qs, axis=-1), axis=-1)
            target_q_values = jnp.take_along_axis(next_qs, max_target_indices, axis=-1).squeeze(-1)
            next_log_probs = jnp.take_along_axis(next_log_probs, max_target_indices, axis=-1).squeeze(-1)
        else:
            dist = actor(batch.next_observations, expl_noise)
            next_actions = dist.sample(seed=key)
            next_qs = target_critic.apply(
                {"params": target_params, **target_critic.extra_variables},
                batch.next_observations,
                next_actions,
                rngs={"dropout": key2},
            )
            target_q_values = next_qs.min(axis=0)
            next_log_probs = dist.log_prob(next_actions)

        if backup_entropy:
            target_q_values -= temp() * next_log_probs

        target_q = jax.lax.stop_gradient(batch.rewards + discount * batch.masks * target_q_values)

        qs, updated_states = critic.apply(
            critic_params,
            batch.observations,
            batch.actions,
            training=True,
            rngs={"dropout": key3},
            mutable=critic.extra_variables.keys(),
        )
        bellman_loss = (qs - target_q) ** 2

        # CQL
        cql_random_actions = jax.random.uniform(
            key4, shape=(batch_size * cql_n_actions, action_dim), minval=-1.0, maxval=1.0
        )
        cql_obs, cql_current_actions, cql_current_log_probs = compute_action_and_log_prob(
            key5, actor, batch.observations, cql_n_actions
        )
        cql_next_obs, cql_next_actions, cql_next_log_probs = compute_action_and_log_prob(
            key6, actor, batch.next_observations, cql_n_actions
        )

        cql_qs_rand, _ = critic.apply(
            critic_params,
            cql_obs,
            cql_random_actions,
            training=True,
            rngs={"dropout": key7},
            mutable=critic.extra_variables.keys(),
        )

        cql_qs_current_actions, _ = critic.apply(
            critic_params,
            cql_obs,
            cql_current_actions,
            training=True,
            rngs={"dropout": key8},
            mutable=critic.extra_variables.keys(),
        )

        cql_qs_next_actions, _ = critic.apply(
            critic_params,
            cql_obs,
            cql_next_actions,
            training=True,
            rngs={"dropout": key9},
            mutable=critic.extra_variables.keys(),
        )

        cql_qs_rand = cql_qs_rand.reshape(num_qs, batch_size, cql_n_actions)
        cql_qs_current_actions = cql_qs_current_actions.reshape(num_qs, batch_size, cql_n_actions)
        cql_qs_next_actions = cql_qs_next_actions.reshape(num_qs, batch_size, cql_n_actions)

        cql_current_log_probs = cql_current_log_probs.reshape(batch_size, cql_n_actions)
        cql_next_log_probs = cql_next_log_probs.reshape(batch_size, cql_n_actions)

        cql_cat_qs = jnp.concatenate(
            [cql_qs_rand, jnp.expand_dims(qs, -1), cql_qs_current_actions, cql_qs_next_actions], axis=-1
        )
        cql_std_qs = jnp.std(cql_cat_qs, axis=-1)

        # CalQL
        if enable_calql:
            lower_bounds = jnp.repeat(batch.mc_returns.reshape(-1, 1), cql_qs_current_actions.shape[2], axis=1)
            num_vals = jnp.sum(lower_bounds == lower_bounds)
            bound_rate_cql_qs_current_actions = jnp.sum(cql_qs_current_actions < lower_bounds) / num_vals
            bound_rate_cql_qs_next_actions = jnp.sum(cql_qs_next_actions < lower_bounds) / num_vals

            lower_bounds = jnp.repeat(jnp.expand_dims(lower_bounds, 0), num_qs, axis=0)
            cql_qs_current_actions = jnp.maximum(cql_qs_current_actions, lower_bounds)
            cql_qs_next_actions = jnp.maximum(cql_qs_next_actions, lower_bounds)

        if cql_importance_sample:
            random_density = jnp.log(0.5**action_dim)
            cql_cat_qs = jnp.concatenate(
                [
                    cql_qs_rand - random_density,
                    cql_qs_next_actions - cql_next_log_probs,
                    cql_qs_current_actions - cql_current_log_probs,
                ],
                axis=-1,
            )

        cql_qs_ood = jax.scipy.special.logsumexp(cql_cat_qs / cql_temp, axis=-1) * cql_temp
        cql_qs_diff = jnp.clip(cql_qs_ood - qs, -jnp.inf, jnp.inf)

        cql_min_qs_loss = cql_qs_diff * cql_min_q_weight
        critic_loss = bellman_loss.mean() + cql_min_qs_loss.mean()

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
                "cql_min_qs_loss_mean": cql_min_qs_loss.mean(),
                "cql_min_qs_loss_min": cql_min_qs_loss.min(),
                "cql_min_qs_loss_max": cql_min_qs_loss.max(),
                "cql_min_qs_loss_std": cql_min_qs_loss.std(),
                "cql_std_qs": cql_std_qs.mean(),
                "bound_rate_cql_qs_current_actions": bound_rate_cql_qs_current_actions,
                "bound_rate_cql_qs_next_actions": bound_rate_cql_qs_next_actions,
            },
        )

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info
