from typing import Callable, Sequence, Tuple, Dict

import jax.numpy as jnp
from flax import linen as nn

from common import MLP


class ValueCritic(nn.Module):
    hidden_dims: Sequence[int]
    emb_dim: int
    encoder_cls: nn.Module = None
    critic_layer_norm: bool = False
    obs_keys: Sequence[str] = ("image1", "image2")

    @nn.compact
    def __call__(self, observations: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        obs = self.encoder_cls(name="encoder")(observations)[:, -1]
        critic = MLP((*self.hidden_dims, 1), use_layer_norm=self.critic_layer_norm)(obs)
        return jnp.squeeze(critic, -1)


class Critic(nn.Module):
    hidden_dims: Sequence[int]
    emb_dim: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    encoder_cls: nn.Module = None
    training: bool = (False,)
    critic_layer_norm: bool = False
    obs_keys: Sequence[str] = ("image1", "image2")

    @nn.compact
    def __call__(
        self, observations: Dict[str, jnp.ndarray], actions: jnp.ndarray, training: bool = False
    ) -> jnp.ndarray:
        obs = self.encoder_cls(name="encoder")(observations, deterministic=training)[:, -1]
        if len(actions.shape) == 3:
            # Reduce the redundant dimension
            actions = jnp.squeeze(actions, 1)

        inputs = jnp.concatenate([obs, actions], -1)
        critic = MLP((*self.hidden_dims, 1), activations=self.activations, use_layer_norm=self.critic_layer_norm)(
            inputs
        )
        return jnp.squeeze(critic, -1)


class DoubleCritic(nn.Module):
    hidden_dims: Sequence[int]
    emb_dim: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    encoder_cls: nn.Module = None
    critic_layer_norm: bool = False
    obs_keys: Sequence[str] = ("image1", "image2")

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        critic1 = Critic(
            self.hidden_dims,
            self.emb_dim,
            activations=self.activations,
            encoder_cls=self.encoder_cls,
            critic_layer_norm=self.critic_layer_norm,
            obs_keys=self.obs_keys,
        )(observations, actions)
        critic2 = Critic(
            self.hidden_dims,
            self.emb_dim,
            activations=self.activations,
            encoder_cls=self.encoder_cls,
            critic_layer_norm=self.critic_layer_norm,
            obs_keys=self.obs_keys,
        )(observations, actions)
        return critic1, critic2
