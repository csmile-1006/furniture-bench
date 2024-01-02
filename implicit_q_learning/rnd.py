from typing import Sequence, Dict, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from common import (
    MLP,
    default_init,
    Batch,
    InfoDict,
    Model,
    Params,
    PRNGKey,
)


class RND(nn.Module):
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, observations: Dict[str, jnp.ndarray], training: bool = False):
        features = []
        for k, v in observations.items():
            if v.ndim == 2:
                v = v[jnp.newaxis]
            features.append(v[:, -1])
        # else:
        #     state_embed = MLP([self.emb_dim, self.emb_dim, self.emb_dim])(v)
        obs = jnp.concatenate(features, axis=-1)
        predict_feature = MLP(
            self.hidden_dims,
        )(obs, training=training)

        target_feature = nn.Dense(self.hidden_dims[-1], kernel_init=default_init())(obs)
        target_feature = jax.lax.stop_gradient(target_feature)

        return predict_feature, target_feature


def update_rnd(key: PRNGKey, rnd: Model, batch: Batch, update_proportion: float = 0.25) -> Tuple[Model, InfoDict]:
    predict_next_feature, target_next_feature = rnd(batch.next_observations)

    def rnd_loss_fn(rnd_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        rnd_loss = jnp.mean((predict_next_feature - target_next_feature) ** 2)
        mask = jax.random.normal(key, (len(predict_next_feature),))
        mask = (mask < update_proportion).astype(jnp.float32)
        rnd_loss = (rnd_loss * mask).sum() / jnp.maximum(mask.sum(), 1.0)
        intrinsic_reward = jnp.mean((target_next_feature - predict_next_feature) ** 2, axis=1) / 2
        return rnd_loss, {"rnd_loss": rnd_loss, "expl_reward": intrinsic_reward}

    new_rnd, info = rnd.apply_gradient(rnd_loss_fn)
    return new_rnd, info
