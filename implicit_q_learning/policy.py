import functools
from typing import Optional, Sequence, Tuple, Dict

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

from common import MLP, ResNet18, Transformer, concat_multiple_image_emb, get_1d_sincos_pos_embed
from common import Params
from common import PRNGKey
from common import default_init

LOG_STD_MIN = -10.0
LOG_STD_MAX = 2.0


class NormalTanhPolicy(nn.Module):
    hidden_dims: Sequence[int]
    emb_dim: int
    action_dim: int
    state_dependent_std: bool = True
    dropout_rate: Optional[float] = None
    log_std_scale: float = 1.0
    log_std_min: Optional[float] = None
    log_std_max: Optional[float] = None
    tanh_squash_distribution: bool = True
    use_encoder: bool = False

    @nn.compact
    def __call__(
        self,
        observations: Dict[str, jnp.ndarray],
        temperature: float = 1.0,
        training: bool = False,
    ) -> tfd.Distribution:
        image_features = {}
        for k, v in observations.items():
            if v.ndim == 2:
                v = v[jnp.newaxis]
            if self.use_encoder and (k == 'image1' or k == 'image2'):
                image_features[k] = v
            else:
                state_embed = MLP([self.emb_dim])(v)
        if self.use_encoder:
            image_features = jnp.array(list(image_features.values()))
            num_image, batch_size, num_timestep, _ = image_features.shape
            image_features = concat_multiple_image_emb(image_features)
            # Image features: (batch_size, num_timestep, num_images * embd_dim)
            image_features = nn.tanh(MLP([self.emb_dim])(image_features))
            image_embed = image_features + get_1d_sincos_pos_embed(self.emb_dim, num_timestep)
            token_embed = jnp.concatenate(
                [image_embed, state_embed], axis=-1
            )
            token_embed = jnp.reshape(
                token_embed,
                [batch_size, 2 * num_timestep, self.emb_dim],
            )
            obs = Transformer(emb_dim=self.emb_dim)(token_embed)[:, -1]
        # obs = jnp.concatenate([image_feature1, image_feature2, observations['robot_state']], axis=-1)
        # obs = jnp.concatenate(features, axis=-1)
        outputs = MLP(self.hidden_dims, activate_final=True,
                      dropout_rate=self.dropout_rate)(obs, training=training)

        means = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)

        if self.state_dependent_std:
            log_stds = nn.Dense(self.action_dim,
                                kernel_init=default_init(self.log_std_scale))(outputs)
        else:
            log_stds = self.param("log_stds", nn.initializers.zeros, (self.action_dim, ))

        log_std_min = self.log_std_min or LOG_STD_MIN
        log_std_max = self.log_std_max or LOG_STD_MAX
        log_stds = jnp.clip(log_stds, log_std_min, log_std_max)

        if not self.tanh_squash_distribution:
            means = nn.tanh(means)

        base_dist = tfd.MultivariateNormalDiag(loc=means,
                                               scale_diag=jnp.exp(log_stds) * temperature)
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
    dist = actor_def.apply({"params": actor_params}, observations, temperature)
    rng, key = jax.random.split(rng)
    return rng, dist.sample(seed=key)


def sample_actions(
    rng: PRNGKey,
    actor_def: nn.Module,
    actor_params: Params,
    observations: np.ndarray,
    temperature: float = 1.0,
) -> Tuple[PRNGKey, jnp.ndarray]:
    return _sample_actions(rng, actor_def, actor_params, observations, temperature)
