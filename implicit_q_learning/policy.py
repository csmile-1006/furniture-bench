import functools
from typing import Optional, Sequence, Tuple, Dict

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

from common import MLP, concat_multiple_image_emb, get_1d_sincos_pos_embed  # noqa: E402
from common import Params  # noqa: E402
from common import PRNGKey  # noqa: E402
from common import default_init  # noqa: E402

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
    encoder_cls: nn.Module = None

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
            if self.use_encoder and (k == "image1" or k == "image2" or k == "text_feature"):
                image_features[k] = v
            # else:
            #     state_embed = MLP([self.emb_dim, self.emb_dim, self.emb_dim])(v)
        if self.use_encoder:
            # for key, val in image_features.items():
            #     print(f"[INFO] {key}: {val.shape}")
            image_features = jnp.array(list(image_features.values()))
            num_image, batch_size, num_timestep, _ = image_features.shape
            image_features = concat_multiple_image_emb(image_features)
            # Image features: (batch_size, num_timestep, num_images * embd_dim)
            # if observations["robot_state"].ndim == 2:
            #     image_features = jnp.concatenate([image_features, observations["robot_state"][jnp.newaxis]], axis=-1)
            # else:
            #     image_features = jnp.concatenate([image_features, observations["robot_state"]], axis=-1)
            image_features = MLP([self.emb_dim], dropout_rate=self.dropout_rate, name="FeatureMLP")(image_features)
            image_embed = image_features + get_1d_sincos_pos_embed(self.emb_dim, num_timestep)
            token_embed = jnp.concatenate(
                [image_embed],
                axis=-1,
            )
            token_embed = jnp.reshape(
                token_embed,
                [batch_size, 1 * num_timestep, self.emb_dim],
            )
            obs = self.encoder_cls(name="encoder")(token_embed, deterministic=training)[:, -1]
        outputs = MLP(self.hidden_dims, activate_final=True, dropout_rate=self.dropout_rate, name="OutputMLP")(
            obs, training=training
        )

        means = nn.Dense(self.action_dim, kernel_init=default_init(), name="OutputDenseMean")(outputs)

        if self.state_dependent_std:
            log_stds = nn.Dense(
                self.action_dim, kernel_init=default_init(self.log_std_scale), name="OutputDenseLogStd"
            )(outputs)
        else:
            log_stds = self.param("OutputLogStd", nn.initializers.zeros, (self.action_dim,))

        log_std_min = self.log_std_min or LOG_STD_MIN
        log_std_max = self.log_std_max or LOG_STD_MAX
        log_stds = jnp.clip(log_stds, log_std_min, log_std_max)

        if not self.tanh_squash_distribution:
            means = nn.tanh(means)

        base_dist = tfd.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
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
