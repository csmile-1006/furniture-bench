import functools
from typing import Optional, Sequence, Tuple, Dict

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import distrax
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

from networks.common import MLP  # noqa: E402
from networks.common import Params  # noqa: E402
from networks.common import PRNGKey  # noqa: E402
from networks.common import default_init  # noqa: E402

LOG_STD_MIN = -10.0
LOG_STD_MAX = 2.0


class NormalTanhPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    state_dependent_std: bool = True
    dropout_rate: Optional[float] = None
    log_std_scale: float = 1.0
    std_min: Optional[float] = None
    std_max: Optional[float] = None
    tanh_squash_distribution: bool = True

    @nn.compact
    def __call__(
        self,
        features: Dict[str, jnp.ndarray],
        expl_noise: float = 1.0,
        training: bool = False,
    ) -> tfd.Distribution:
        outputs = MLP(self.hidden_dims, activate_final=True, dropout_rate=0.0, name="OutputMLP")(
            features, training=training
        )

        means = nn.Dense(self.action_dim, kernel_init=default_init(1.0), name="OutputDenseMean")(outputs)

        if self.state_dependent_std:
            log_stds = nn.Dense(self.action_dim, kernel_init=default_init(1.0), name="OutputDenseLogStd")(outputs)
        else:
            log_stds = self.param("OutputLogStd", nn.initializers.zeros, (self.action_dim,))

        # log_stds = nn.tanh(log_stds)
        # log_stds = (self.log_std_max - self.log_std_min) * 0.5 * (log_stds + 1.0) + self.log_std_min
        # log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)
        stds = (self.std_max - self.std_min) * nn.sigmoid(log_stds) + self.std_min

        if not self.tanh_squash_distribution:
            means = nn.tanh(means)

        if self.tanh_squash_distribution:
            return distrax.Transformed(
                distrax.MultivariateNormalDiag(means, jnp.exp(log_stds) * expl_noise),
                distrax.Block(distrax.Tanh(), ndims=1),
            )
        else:
            # base_dist = tfd.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * expl_noise)
            base_dist = tfd.MultivariateNormalDiag(loc=means, scale_diag=stds * expl_noise)
            return base_dist


class NormalTanhMixturePolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    num_modes: int = 5
    dropout_rate: Optional[float] = None
    std_min: Optional[float] = None
    std_max: Optional[float] = None
    use_tanh: bool = False

    @nn.compact
    def __call__(self, features: jnp.ndarray, expl_noise: float = 1.0, training: bool = False) -> tfd.Distribution:
        # obs = self.encoder_cls(name="encoder")(observations, deterministic=not training)[:, -1]
        outputs = MLP(self.hidden_dims, activate_final=True, dropout_rate=self.dropout_rate, name="OutputMLP")(
            features, training=training
        )

        logits = nn.Dense(self.action_dim * self.num_modes, kernel_init=default_init(1.0), name="OutputDenseLogit")(
            outputs
        )
        means = nn.Dense(
            self.action_dim * self.num_modes,
            kernel_init=default_init(1.0),
            bias_init=nn.initializers.normal(stddev=1.0),
            name="OutputDenseMean",
        )(outputs)
        scales = nn.Dense(self.action_dim * self.num_modes, kernel_init=default_init(1.0), name="OutputDenseScales")(
            outputs
        )
        scales = (self.std_max - self.std_min) * nn.sigmoid(scales) + self.std_min

        if not self.use_tanh:
            means = nn.tanh(means)

        shape = list(means.shape[:-1]) + [-1, self.num_modes]

        logits = jnp.reshape(logits, shape)
        mu = jnp.reshape(means, shape)
        scales = jnp.reshape(scales, shape)

        components_distribution = tfd.Normal(loc=mu, scale=scales * expl_noise)

        dist = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=logits), components_distribution=components_distribution
        )

        if self.use_tanh:
            dist = tfd.TransformedDistribution(distribution=dist, bijector=tfb.Tanh())

        return tfd.Independent(dist, 1)


@functools.partial(jax.jit, static_argnames=("actor_def", "distribution"))
def _sample_actions(
    rng: PRNGKey,
    actor_def: nn.Module,
    actor_params: Params,
    observations: np.ndarray,
    expl_noise: float = 1.0,
) -> Tuple[PRNGKey, jnp.ndarray]:
    dist = actor_def.apply(actor_params, observations, expl_noise)
    rng, key = jax.random.split(rng)
    return rng, dist.sample(seed=key)


def sample_actions(
    rng: PRNGKey,
    actor_def: nn.Module,
    actor_params: Params,
    observations: np.ndarray,
    expl_noise: float = 1.0,
) -> Tuple[PRNGKey, jnp.ndarray]:
    return _sample_actions(rng, actor_def, actor_params, observations, expl_noise)
