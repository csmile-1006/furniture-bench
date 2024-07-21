import functools
from typing import Optional, Sequence, Tuple, Dict

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

from common import MLP, Encoder, ResNet18
from common import Params
from common import PRNGKey
from common import default_init

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class NormalTanhPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    state_dependent_std: bool = True
    dropout_rate: Optional[float] = None
    log_std_scale: float = 1.0
    log_std_min: Optional[float] = None
    log_std_max: Optional[float] = None
    det_policy: bool = False
    tanh_squash_distribution: bool = True
    use_encoder: bool = False
    use_layer_norm: bool = False

    @nn.compact
    def __call__(
        self,
        observations: Dict[str, jnp.ndarray],
        temperature: float = 1.0,
        training: bool = False,
    ) -> tfd.Distribution:
        features = []
        for k, v in observations.items():
            if self.use_encoder and (k == 'image1' or k == 'image2'):
                features.append(Encoder()(v))
            else:
                features.append(v)
        # obs = jnp.concatenate([image_feature1, image_feature2, observations['robot_state']], axis=-1)
        obs = jnp.concatenate(features, axis=-1)
        outputs = MLP(self.hidden_dims, activate_final=True,
                      dropout_rate=self.dropout_rate,
                      use_layer_norm=self.use_layer_norm)(obs, training=training)

        means = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)

        if self.state_dependent_std:
            log_stds = nn.Dense(self.action_dim,
                                kernel_init=default_init(self.log_std_scale))(outputs)
        else:
            log_stds = self.param("log_stds", nn.initializers.zeros, (self.action_dim, ))

        if not self.det_policy:
            log_std_min = self.log_std_min or LOG_STD_MIN
            log_std_max = self.log_std_max or LOG_STD_MAX
            log_stds = jnp.clip(log_stds, log_std_min, log_std_max)

            if not self.tanh_squash_distribution:
                means = nn.tanh(means)

            base_dist = tfd.MultivariateNormalDiag(loc=means,
                                                   scale_diag=jnp.exp(log_stds) * temperature)
        else:
            # Convert quaternion to euler angle.
            if not self.tanh_squash_distribution:
                means = nn.tanh(means)
            std = jnp.ones((self.action_dim,)) * 0.0001 # Small value to prevent NaN.

            base_dist = tfd.MultivariateNormalDiag(loc=means, scale_diag=std)
            
        if self.tanh_squash_distribution:
            return tfd.TransformedDistribution(distribution=base_dist, bijector=tfb.Tanh())
        else:
            return base_dist


@functools.partial(jax.jit, static_argnames=("actor_def", "distribution"))
def _sample_actions(
    rng: PRNGKey,
    actor_def: nn.Module,
    actor_params: Params,
    observations: np.ndarray,
    temperature: float = 1.0,
) -> Tuple[PRNGKey, jnp.ndarray]:
    # dist = actor_def.apply({"params": actor_params}, observations, temperature)
    # rng, key = jax.random.split(rng)
    # return rng, dist.sample(seed=key)
    rng, key = jax.random.split(rng)
    dist = _dist_actions(actor_def, actor_params, observations, temperature)
    return rng, dist.sample(seed=key)


@functools.partial(jax.jit, static_argnames=("actor_def", "distribution"))
def _dist_actions(
    actor_def: nn.Module,
    actor_params: Params,
    observations: np.ndarray,
    temperature: float = 1.0,
) -> jnp.ndarray:
    dist = actor_def.apply({"params": actor_params}, observations, temperature)
    return dist


def sample_actions(
    rng: PRNGKey,
    actor_def: nn.Module,
    actor_params: Params,
    observations: np.ndarray,
    temperature: float = 1.0,
) -> Tuple[PRNGKey, jnp.ndarray]:
    return _sample_actions(rng, actor_def, actor_params, observations, temperature)


def dist_actions(
    actor_def: nn.Module,
    actor_params: Params,
    observations: np.ndarray,
    temperature: float = 1.0,
) -> jnp.ndarray:
    return _dist_actions(actor_def, actor_params, observations, temperature)


def logprob(
    actor_def: nn.Module,
    actor_params: Params,
    observations: np.ndarray,
    actions: np.ndarray,
    temperature: float = 1.0,
) -> jnp.ndarray:
    """Compute the log probability of the actions under the current policy."""
    dist = actor_def.apply({"params": actor_params}, observations, temperature)
    return dist.log_prob(actions)
