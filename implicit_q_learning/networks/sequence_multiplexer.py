from typing import Dict, Optional, Type, Union

import flax.linen as nn
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from networks import MLP
from networks.encoders.transformer_encoder import get_1d_sincos_pos_embed


class SequenceMultiplexer(nn.Module):
    encoder_cls: Type[nn.Module]
    network_cls: Type[nn.Module]
    latent_dim: int

    @nn.compact
    def __call__(
        self,
        observations: Union[FrozenDict, Dict],
        actions: Optional[jnp.ndarray] = None,
        training: bool = False,
    ) -> jnp.ndarray:
        # observations: (batch_size, seq_length, observation_dim (color_image1 + color_image2 + robot_state))
        num_timestep = observations.shape[-2]
        x = MLP([self.latent_dim, self.latent_dim, self.latent_dim])(observations)
        x = x + get_1d_sincos_pos_embed(self.latent_dim, num_timestep)
        x = self.encoder_cls(name="encoder")(x, deterministic=training)[:, -1]
        x = x.squeeze()

        if actions is None:
            return self.network_cls()(x, training)
        else:
            return self.network_cls()(x, actions, training)
