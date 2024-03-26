from typing import Callable, Dict, Sequence, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

from networks.common import MLP


class ValueCritic(nn.Module):
    hidden_dims: Sequence[int]
    emb_dim: int
    critic_layer_norm: bool = False

    @nn.compact
    def __call__(self, features: Dict[str, jnp.ndarray], training=False) -> jnp.ndarray:
        # obs = self.encoder_cls(name="encoder")(observations, deterministic=not training)[:, -1]
        critic = MLP((*self.hidden_dims, 1), use_layer_norm=self.critic_layer_norm)(features)
        return jnp.squeeze(critic, -1)


class Critic(nn.Module):
    hidden_dims: Sequence[int]
    emb_dim: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    training: bool = (False,)
    critic_layer_norm: bool = False

    @nn.compact
    def __call__(self, features: Dict[str, jnp.ndarray], actions: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        # obs = self.encoder_cls(name="encoder")(observations, deterministic=not training)[:, -1]
        if len(actions.shape) == 3:
            # Reduce the redundant dimension
            actions = jnp.squeeze(actions, 1)

        inputs = jnp.concatenate([features, actions], -1)
        critic = MLP((*self.hidden_dims, 1), activations=self.activations, use_layer_norm=self.critic_layer_norm)(
            inputs
        )
        return jnp.squeeze(critic, -1)


class DoubleCritic(nn.Module):
    hidden_dims: Sequence[int]
    emb_dim: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    critic_layer_norm: bool = False

    @nn.compact
    def __call__(
        self, features: jnp.ndarray, actions: jnp.ndarray, training: bool = False
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        critic1 = Critic(
            self.hidden_dims,
            self.emb_dim,
            activations=self.activations,
            critic_layer_norm=self.critic_layer_norm,
        )(features, actions, training=training)
        critic2 = Critic(
            self.hidden_dims,
            self.emb_dim,
            activations=self.activations,
            critic_layer_norm=self.critic_layer_norm,
        )(features, actions, training=training)
        return critic1, critic2


class CriticEnsemble(nn.Module):
    hidden_dims: Sequence[int]
    emb_dim: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    critic_layer_norm: bool = False
    num_qs: int = 2

    @nn.compact
    def __call__(self, states, actions, training: bool = False):
        VmapCritic = nn.vmap(
            Critic,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num_qs,
        )
        qs = VmapCritic(
            self.hidden_dims,
            self.emb_dim,
            activations=self.activations,
            critic_layer_norm=self.critic_layer_norm,
        )(states, actions, training)
        return qs


def subsample_critic_ensemble(key: jax.random.PRNGKey, params, num_sample: int, num_qs: int):
    if num_sample is not None:
        all_indx = jnp.arange(0, num_qs)
        indx = jax.random.choice(key, a=all_indx, shape=(num_sample,), replace=False)

        if "CriticEnsemble_0" in params:
            ens_params = jax.tree_util.tree_map(lambda param: param[indx], params["CriticEnsemble_0"])
            params = params.copy(add_or_replace={"CriticEnsemble_0": ens_params})
        else:
            params = jax.tree_util.tree_map(lambda param: param[indx], params)
    return params
