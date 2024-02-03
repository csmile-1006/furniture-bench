import collections
from functools import partial
import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import einops
import pickle

Batch = collections.namedtuple("Batch", ["observations", "actions", "rewards", "masks", "next_observations"])


def default_init(scale: Optional[float] = None):
    if scale is None:
        scale = jnp.sqrt(2)
    return nn.initializers.orthogonal(scale)


PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
PRNGKey = Any
Shape = Sequence[int]
Dtype = Any  # this could be a real type?
InfoDict = Dict[str, float]


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    dropout_rate: Optional[float] = None
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init())(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.dropout_rate is not None:
                    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
                if self.use_layer_norm:
                    x = nn.LayerNorm()(x)
                x = self.activations(x)
        return x


class Encoder(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=32, kernel_size=(3, 3), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=32, kernel_size=(3, 3), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=32, kernel_size=(3, 3), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=50)(x)
        x = nn.LayerNorm()(x)
        return x


# class Encoder(nn.Module):
#   @nn.compact
#   def __call__(self, x):
#     x = nn.Conv(features=32, kernel_size=(3, 3))(x)
#     x = nn.relu(x)
#     x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
#     x = nn.Conv(features=64, kernel_size=(3, 3))(x)
#     x = nn.relu(x)
#     x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
#     x = x.reshape((x.shape[0], -1))  # flatten
#     x = nn.Dense(features=256)(x)
#     x = nn.relu(x)
#     x = nn.Dense(features=10)(x)
#     return x


class FeedForward(nn.Module):
    dim: int = 256
    out_dim: int = 256
    dropout: float = 0.0
    use_bias: bool = False
    kernel_init: Callable = nn.initializers.xavier_uniform()
    bias_init: Callable = nn.initializers.zeros
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, batch, deterministic=None):
        deterministic = nn.merge_param("deterministic", self.deterministic, deterministic)
        x = nn.Dense(
            self.dim,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name="fc1",
        )(batch)

        x = nn.gelu(x)
        x = nn.Dropout(self.dropout)(x, deterministic)
        x = nn.Dense(
            self.out_dim,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name="fc2",
        )(x)
        x = nn.Dropout(self.dropout)(x, deterministic)

        return x


class Attention(nn.Module):
    dim: int
    num_heads: int = 8
    use_bias: bool = False
    att_drop: float = 0
    proj_drop: float = 0
    kernel_init: Callable = nn.linear.default_kernel_init
    bias_init: Callable = nn.initializers.zeros
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, batch, deterministic=None, custom_mask=None):
        deterministic = nn.merge_param("deterministic", self.deterministic, deterministic)
        qkv = nn.Dense(
            self.dim * 3,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(batch)
        qkv = jnp.split(qkv, 3, axis=-1)

        mh_fn = lambda x: einops.rearrange(x, "b n (h d) -> b h n d", h=self.num_heads)  # noqa: E731
        q, k, v = jax.tree_map(mh_fn, qkv)

        scale = (self.dim // self.num_heads) ** -0.5
        attention = (q @ jnp.swapaxes(k, -2, -1)) * scale

        n = attention.shape[-1]
        mask = custom_mask
        if mask is None:
            # mask = jnp.tril(jnp.ones((n, n)))[None, None, ...]
            mask = jnp.ones((n, n))[None, None, ...]
            mask = jnp.broadcast_to(mask, attention.shape)

        big_neg = jnp.finfo(attention.dtype).min
        attention = jnp.where(mask == 0, big_neg, attention)
        attention = nn.softmax(attention, axis=-1)
        attention = nn.Dropout(self.att_drop)(attention, deterministic)

        x = einops.rearrange(attention @ v, "b h n d -> b n (h d)")
        x = nn.Dense(
            self.dim,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(x)
        x = nn.Dropout(self.proj_drop)(x, deterministic)

        return x


class Block(nn.Module):
    dim: int = 256
    num_heads: int = 8
    mlp_ratio: int = 4
    att_drop: float = 0.0
    drop: float = 0.0

    @nn.compact
    def __call__(self, batch, deterministic=False, custom_mask=None):
        x = nn.LayerNorm()(batch)
        x = Attention(
            self.dim,
            self.num_heads,
            True,
            self.att_drop,
            self.drop,
        )(x, deterministic, custom_mask)
        batch = batch + x

        x = nn.LayerNorm()(batch)
        x = FeedForward(self.dim * self.mlp_ratio, self.dim, self.drop)(x, deterministic)
        return batch + x


class TransformerEncoder(nn.Module):
    emb_dim: int = 1024
    depth: int = 2
    att_drop: float = 0.0
    drop: float = 0.0
    num_heads: int = 8
    mlp_ratio: int = 4
    obs_keys: Sequence[str] = ("image1", "image2", "text_feature")

    @nn.compact
    def __call__(self, observations: Dict[str, jnp.ndarray], deterministic=False, custom_mask=None):
        features = {}
        for k, v in observations.items():
            if v.ndim == 2:
                v = v[jnp.newaxis]
            if k in self.obs_keys:
                features[k] = v

        features = jnp.array(list(features.values()))
        num_image, batch_size, num_timestep, _ = features.shape
        features = concat_multiple_image_emb(features)
        features = MLP([self.emb_dim], dropout_rate=self.drop, name="FeatureMLP")(features)
        embed = features + get_1d_sincos_pos_embed(features.shape[-1], num_timestep)

        x = embed
        for _ in range(self.depth):
            x = Block(
                dim=self.emb_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                att_drop=self.att_drop,
                drop=self.drop,
            )(x, deterministic, custom_mask)

        x = nn.LayerNorm()(x)
        return x


class PrenormPixelLangBlock(nn.Module):
    dim: int = 256
    num_heads: int = 8
    mlp_ratio: int = 4
    att_drop: float = 0.0
    drop: float = 0.0

    @nn.compact
    def __call__(self, pixel, lang, deterministic=False, custom_mask=None):
        residual_pixel = pixel
        pixel = nn.LayerNorm()(pixel)
        lang = nn.LayerNorm()(lang)
        x2 = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.dim,
            dropout_rate=self.att_drop,
        )(inputs_q=pixel, inputs_kv=lang, mask=custom_mask, deterministic=deterministic)
        x2 = nn.Dropout(self.drop)(x2, deterministic)
        x3 = residual_pixel + x2

        x4 = nn.LayerNorm()(x3)
        x5 = FeedForward(self.dim * self.mlp_ratio, self.dim, self.drop)(x4, deterministic)
        x = x5 + x3
        return x


class CrossAttnTransformerEncoder(nn.Module):
    emb_dim: int = 1024
    depth: int = 2
    att_drop: float = 0.0
    drop: float = 0.0
    num_heads: int = 8
    mlp_ratio: int = 4
    obs_keys: Sequence[str] = ("image1", "image2", "text_feature")

    @nn.compact
    def __call__(self, observations: Dict[str, jnp.ndarray], deterministic=False, custom_mask=None):
        assert "text_feature" in observations, "text_feature must be in observations"
        image_features = {}
        for k, v in observations.items():
            if v.ndim == 2:
                v = v[jnp.newaxis]
            if k in ["image1", "image2"]:
                image_features[k] = v
        image_features = jnp.array(list(image_features.values()))
        num_image, batch_size, num_timestep, _ = image_features.shape
        image_features = concat_multiple_image_emb(image_features)
        image_features = MLP([self.emb_dim], dropout_rate=self.drop, name="ImageFeatureMLP")(image_features)
        image_embed = image_features + get_1d_sincos_pos_embed(image_features.shape[-1], num_timestep)

        text_feature = observations["text_feature"]
        text_feature = MLP([self.emb_dim], dropout_rate=self.drop, name="TextFeatureMLP")(text_feature)
        text_embed = text_feature + get_1d_sincos_pos_embed(text_feature.shape[-1], num_timestep)

        fused_feature = image_embed
        for _ in range(self.depth):
            fused_feature = PrenormPixelLangBlock(
                dim=self.emb_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                att_drop=self.att_drop,
                drop=self.drop,
            )(lang=text_embed, pixel=image_embed, deterministic=deterministic)
        fused_feature = nn.LayerNorm()(fused_feature)

        x = fused_feature
        for _ in range(self.depth):
            x = Block(
                dim=self.emb_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                att_drop=self.att_drop,
                drop=self.drop,
            )(fused_feature, deterministic, custom_mask)

        x = nn.LayerNorm()(x)
        return x


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = jnp.arange(embed_dim // 2, dtype=jnp.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = jnp.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = jnp.sin(out)  # (M, D/2)
    emb_cos = jnp.cos(out)  # (M, D/2)

    emb = jnp.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, length):
    return jnp.expand_dims(
        get_1d_sincos_pos_embed_from_grid(embed_dim, jnp.arange(length, dtype=jnp.float32)),
        0,
    )


def concat_multiple_image_emb(img_emb):
    num_image, batch_size, num_timestep = img_emb.shape[:3]
    img_emb = jnp.reshape(img_emb, (batch_size * num_image, num_timestep, -1))
    img_emb = jnp.concatenate(jnp.split(img_emb, num_image, axis=0), -1)  # (batch_size, num_timestep, emb_dim)
    return img_emb


class ResNetBlock(nn.Module):
    """ResNet block."""

    filters: int
    conv: Any
    norm: Any
    act: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(
        self,
        x,
    ):
        residual = x
        y = self.conv(self.filters, (3, 3), self.strides)(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3))(y)
        y = self.norm(scale_init=nn.initializers.zeros)(y)

        if residual.shape != y.shape:
            residual = self.conv(self.filters, (1, 1), self.strides, name="conv_proj")(residual)
            residual = self.norm(name="norm_proj")(residual)

        return self.act(residual + y)


class ResNet(nn.Module):
    """ResNetV1."""

    stage_sizes: Sequence[int]
    block_cls: Any
    num_filters: int = 64
    dtype: Any = jnp.float32
    act: Callable = nn.relu

    @nn.compact
    def __call__(self, x, train: bool = True):
        conv = partial(nn.Conv, use_bias=False, dtype=self.dtype)
        norm = partial(nn.BatchNorm, use_running_average=not train, momentum=0.9, epsilon=1e-5, dtype=self.dtype)

        x = conv(self.num_filters, (7, 7), (2, 2), padding=[(3, 3), (3, 3)], name="conv_init")(x)
        x = norm(name="bn_init")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")
        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = self.block_cls(self.num_filters * 2**i, strides=strides, conv=conv, norm=norm, act=self.act)(x)
        x = jnp.mean(x, axis=(1, 2))
        # x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
        x = jnp.asarray(x, self.dtype)
        return x


ResNet18 = partial(ResNet, stage_sizes=[1, 1], block_cls=ResNetBlock)


@flax.struct.dataclass
class Model:
    step: int
    apply_fn: nn.Module = flax.struct.field(pytree_node=False)
    params: Params
    tx: Optional[optax.GradientTransformation] = flax.struct.field(pytree_node=False)
    opt_state: Optional[optax.OptState] = None

    @classmethod
    def create(
        cls, model_def: nn.Module, inputs: Sequence[jnp.ndarray], tx: Optional[optax.GradientTransformation] = None
    ) -> "Model":
        variables = model_def.init(*inputs)

        _, params = variables.pop("params")

        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(step=1, apply_fn=model_def, params=params, tx=tx, opt_state=opt_state)

    def __call__(self, *args, **kwargs):
        return self.apply_fn.apply({"params": self.params}, *args, **kwargs)

    def apply(self, *args, **kwargs):
        return self.apply_fn.apply(*args, **kwargs)

    def apply_gradient(self, loss_fn) -> Tuple[Any, "Model"]:
        grad_fn = jax.grad(loss_fn, has_aux=True)
        grads, info = grad_fn(self.params)

        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(step=self.step + 1, params=new_params, opt_state=new_opt_state), info

    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(self.params, f)
            # f.write(flax.serialization.to_bytes(self.params))

    def load(self, load_path: str) -> "Model":
        with open(load_path, "rb") as f:
            params = pickle.load(f)
            # params = flax.serialization.from_bytes(self.params, f.read())
        return self.replace(params=params)
