from typing import Callable, Sequence, Tuple, Dict

import jax.numpy as jnp
from flax import linen as nn

from common import MLP, concat_multiple_image_emb, get_1d_sincos_pos_embed


class ValueCritic(nn.Module):
    hidden_dims: Sequence[int]
    emb_dim: int
    encoder_cls: nn.Module = None
    critic_layer_norm: bool = False
    obs_keys: Sequence[str] = ("image1", "image2")

    @nn.compact
    def __call__(self, observations: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        image_features = {}
        for k, v in observations.items():
            if v.ndim == 2:
                v = v[jnp.newaxis]
            if k in self.obs_keys:
                image_features[k] = v
        image_features = jnp.array(list(image_features.values()))
        num_image, batch_size, num_timestep, _ = image_features.shape
        image_features = concat_multiple_image_emb(image_features)
        image_features = MLP([self.emb_dim])(image_features)
        image_embed = image_features + get_1d_sincos_pos_embed(self.emb_dim, num_timestep)
        token_embed = jnp.concatenate(
            [image_embed],
            axis=-1,
        )
        token_embed = jnp.reshape(
            token_embed,
            [batch_size, 1 * num_timestep, self.emb_dim],
        )
        obs = self.encoder_cls(name="encoder")(token_embed)[:, -1]
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
        image_features = {}
        for k, v in observations.items():
            if k in self.obs_keys:
                image_features[k] = v
        image_features = jnp.array(list(image_features.values()))
        num_image, batch_size, num_timestep, _ = image_features.shape
        image_features = concat_multiple_image_emb(image_features)
        image_features = MLP([self.emb_dim])(image_features)
        image_embed = image_features + get_1d_sincos_pos_embed(self.emb_dim, num_timestep)
        token_embed = jnp.concatenate([image_embed], axis=-1)
        token_embed = jnp.reshape(
            token_embed,
            [batch_size, 1 * num_timestep, self.emb_dim],
        )
        obs = self.encoder_cls(name="encoder")(token_embed, deterministic=training)[:, -1]
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
