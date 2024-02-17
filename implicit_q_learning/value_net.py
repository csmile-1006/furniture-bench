from typing import Callable, Sequence, Tuple, Dict

import jax.numpy as jnp
from flax import linen as nn

from common import MLP


class ValueCritic(nn.Module):
    hidden_dims: Sequence[int]
    emb_dim: int
    critic_layer_norm: bool = False
    obs_keys: Sequence[str] = ("image1", "image2")

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
    obs_keys: Sequence[str] = ("image1", "image2")

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
    obs_keys: Sequence[str] = ("image1", "image2")

    @nn.compact
    def __call__(
        self, features: jnp.ndarray, actions: jnp.ndarray, training: bool = False
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        critic1 = Critic(
            self.hidden_dims,
            self.emb_dim,
            activations=self.activations,
            critic_layer_norm=self.critic_layer_norm,
            obs_keys=self.obs_keys,
        )(features, actions, training=training)
        critic2 = Critic(
            self.hidden_dims,
            self.emb_dim,
            activations=self.activations,
            critic_layer_norm=self.critic_layer_norm,
            obs_keys=self.obs_keys,
        )(features, actions, training=training)
        return critic1, critic2
