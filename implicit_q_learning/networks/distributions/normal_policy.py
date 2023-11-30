from typing import Optional, Type, Any

import tensorflow_probability


tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions
tfb = tfp.bijectors

import flax.linen as nn
import jax.numpy as jnp

from networks.constants import default_init


class TanhTransformedDistribution(tfd.TransformedDistribution):
    def __init__(self, distribution: tfd.Distribution, validate_args: bool = False):
        super().__init__(distribution=distribution, bijector=tfb.Tanh(), validate_args=validate_args)

    def mode(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.mode())

    @classmethod
    def _parameter_properties(cls, dtype: Optional[Any], num_classes=None):
        td_properties = super()._parameter_properties(dtype, num_classes=num_classes)
        del td_properties["bijector"]
        return td_properties


class UnitStdNormalPolicy(nn.Module):
    base_cls: Type[nn.Module]
    action_dim: int
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    state_dependent_std: bool = True
    squash_tanh: bool = False

    @nn.compact
    def __call__(self, inputs, *args, **kwargs) -> tfd.Distribution:
        x = self.base_cls()(inputs, *args, **kwargs)

        means = nn.Dense(self.action_dim, kernel_init=default_init(), name="OutputDenseMean")(x)
        if self.state_dependent_std:
            log_stds = nn.Dense(self.action_dim, kernel_init=default_init(), name="OutputDenseLogStd")(x)
        else:
            log_stds = self.param("OutpuLogStd", nn.initializers.zeros, (self.action_dim,), jnp.float32)

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)
        distribution = tfd.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds))

        if self.squash_tanh:
            return TanhTransformedDistribution(distribution)
        else:
            return distribution


class NormalTanhPolicy(nn.Module):
    base_cls: Type[nn.Module]
    action_dim: int
    log_std_scale: Optional[float] = 1.0
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    state_dependent_std: bool = True
    squash_tanh: bool = False

    @nn.compact
    def __call__(self, inputs, *args, temperature=1.0, **kwargs) -> tfd.Distribution:
        x = self.base_cls()(inputs, *args, **kwargs)

        means = nn.Dense(self.action_dim, kernel_init=default_init(), name="OutputDenseMean")(x)
        if self.state_dependent_std:
            log_stds = nn.Dense(
                self.action_dim, kernel_init=default_init(self.log_std_scale), name="OutputDenseLogStd"
            )(x)
        else:
            log_stds = self.param("OutpuLogStd", nn.initializers.zeros, (self.action_dim,), jnp.float32)

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)
        if not self.squash_tanh:
            means = nn.tanh(means)

        distribution = tfd.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)

        if self.squash_tanh:
            return TanhTransformedDistribution(distribution)
        else:
            return distribution
