from typing import Callable, Sequence, Tuple, Dict

import jax.numpy as jnp
from flax import linen as nn

from common import MLP, Transformer, concat_multiple_image_emb, get_1d_sincos_pos_embed


class ValueCritic(nn.Module):
    hidden_dims: Sequence[int]
    emb_dim: int
    use_encoder: bool = False
    encoder: nn.Module = None

    @nn.compact
    def __call__(self, observations: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        image_features = {}
        for k, v in observations.items():
            if v.ndim == 2:
                v = v[jnp.newaxis]
            if self.use_encoder and (k == 'image1' or k == 'image2'):
                image_features[k] = v
            else:
                state_embed = MLP([self.emb_dim, self.emb_dim, self.emb_dim])(v)
        if self.use_encoder:
            image_features = jnp.array(list(image_features.values()))
            num_image, batch_size, num_timestep, _ = image_features.shape
            image_features = concat_multiple_image_emb(image_features)
            # Image features: (batch_size, num_timestep, num_images * embd_dim)
            image_features = MLP([self.emb_dim, self.emb_dim, self.emb_dim])(image_features)
            image_embed = image_features + get_1d_sincos_pos_embed(self.emb_dim, num_timestep)
            token_embed = jnp.concatenate(
                [image_embed, state_embed], axis=-1
            )
            token_embed = jnp.reshape(
                token_embed,
                [batch_size, 2 * num_timestep, self.emb_dim],
            )
            obs = self.encoder(token_embed)[:, -1]
        critic = MLP((*self.hidden_dims, 1))(obs)
        return jnp.squeeze(critic, -1)


class Critic(nn.Module):
    hidden_dims: Sequence[int]
    emb_dim: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    use_encoder: bool = False
    encoder: nn.Module = None
    training: bool = False,

    @nn.compact
    def __call__(self, observations: Dict[str, jnp.ndarray], actions: jnp.ndarray, training:bool=False) -> jnp.ndarray:
        image_features = {}
        for k, v in observations.items():
            if self.use_encoder and (k == 'image1' or k == 'image2'):
                image_features[k] = v
            else:
                state_embed = MLP([self.emb_dim, self.emb_dim, self.emb_dim])(v)
        for k, v in observations.items():
            if v.ndim == 2:
                v = v[jnp.newaxis]
            if self.use_encoder and (k == 'image1' or k == 'image2'):
                image_features[k] = v
            else:
                state_embed = MLP([self.emb_dim, self.emb_dim, self.emb_dim])(v)
        if self.use_encoder:
            image_features = jnp.array(list(image_features.values()))
            num_image, batch_size, num_timestep, _ = image_features.shape
            image_features = concat_multiple_image_emb(image_features)
            # Image features: (batch_size, num_timestep, num_images * embd_dim)
            image_features = MLP([self.emb_dim, self.emb_dim, self.emb_dim])(image_features)
            image_embed = image_features + get_1d_sincos_pos_embed(self.emb_dim, num_timestep)
            token_embed = jnp.concatenate(
                [image_embed, state_embed], axis=-1
            )
            token_embed = jnp.reshape(
                token_embed,
                [batch_size, 2 * num_timestep, self.emb_dim],
            )
            obs = self.encoder(token_embed, deterministic=training)[:, -1]
        if len(actions.shape) == 3:
            # Reduce the redundant dimension
            actions = jnp.squeeze(actions, 1)

        inputs = jnp.concatenate([obs, actions], -1)
        critic = MLP((*self.hidden_dims, 1), activations=self.activations)(inputs)
        return jnp.squeeze(critic, -1)


class DoubleCritic(nn.Module):
    hidden_dims: Sequence[int]
    emb_dim: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    use_encoder: bool = False
    encoder: nn.Module = None

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        critic1 = Critic(self.hidden_dims,
                         self.emb_dim,
                         activations=self.activations,
                         use_encoder=self.use_encoder,
                         encoder=self.encoder)(observations, actions)
        critic2 = Critic(self.hidden_dims,
                         self.emb_dim,
                         activations=self.activations,
                         use_encoder=self.use_encoder,
                         encoder=self.encoder)(observations, actions)
        return critic1, critic2
