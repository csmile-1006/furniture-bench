from typing import Dict, Optional, Type, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict


class SequentialMultiplexer(nn.Module):
    encoder_cls: Type[nn.Module]
    network_cls: Type[nn.Module]
    stop_gradient: bool = False

    @nn.compact
    def __call__(
        self,
        observations: Union[FrozenDict, Dict],
        actions: Optional[jnp.ndarray] = None,
        training: bool = False,
    ) -> jnp.ndarray:
        observations = FrozenDict(observations)
        x = self.encoder_cls(name="encoder")(observations, deterministic=not training)[:, -1]
        if self.stop_gradient:
            x = jax.lax.stop_gradient(x)

        if actions is None:
            return self.network_cls()(x, training)
        else:
            return self.network_cls()(x, actions, training)

    def disable_gradient(self):
        self.stop_gradient = True


class ConcatMultiPlexer(nn.Module):
    encoder_cls: Type[nn.Module]
    network_cls: Type[nn.Module]
    stop_gradient: bool = False

    @nn.compact
    def __call__(
        self,
        observations: Union[FrozenDict, Dict],
        actions: Optional[jnp.ndarray] = None,
        training: bool = False,
    ) -> jnp.ndarray:
        observations = FrozenDict(observations)
        x = self.encoder_cls(name="encoder")(observations, deterministic=not training)
        if self.stop_gradient:
            x = jax.lax.stop_gradient(x)

        if actions is None:
            return self.network_cls()(x, training)
        else:
            return self.network_cls()(x, actions, training)

    def disable_gradient(self):
        self.stop_gradient = True
