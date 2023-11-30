from functools import partial
from typing import Callable, Tuple

import distrax
import jax
import jax.numpy as jnp
import numpy as np

from data_types import Params, PRNGKey


@partial(jax.jit, static_argnames="actor_apply_fn")
def eval_actions_jit(
    actor_apply_fn: Callable[..., distrax.Distribution],
    actor_params: Params,
    observations: np.ndarray,
) -> jnp.ndarray:
    dist = actor_apply_fn({"params": actor_params}, observations)
    return dist.mode()


@partial(jax.jit, static_argnames="actor_apply_fn")
def iql_eval_actions_jit(
    rng: PRNGKey, actor_apply_fn: Callable[..., distrax.Distribution], actor_params: Params, observations: np.ndarray
) -> jnp.ndarray:
    dist = actor_apply_fn({"params": actor_params}, observations, temperature=0.0)
    new_rng, key = jax.random.split(rng)
    return new_rng, dist.sample(seed=key)


@partial(jax.jit, static_argnames="actor_apply_fn")
def sample_actions_jit(
    rng: PRNGKey,
    actor_apply_fn: Callable[..., distrax.Distribution],
    actor_params: Params,
    observations: np.ndarray,
) -> Tuple[PRNGKey, jnp.ndarray]:
    dist = actor_apply_fn({"params": actor_params}, observations)
    rng, key = jax.random.split(rng)
    return rng, dist.sample(seed=key)
