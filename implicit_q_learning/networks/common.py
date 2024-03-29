import collections
from functools import partial
import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import flax
import flax.linen as nn
import jax
from jax import lax
import jax.nn.initializers as initjax
from flax.linen.dtypes import promote_dtype
import jax.numpy as jnp
import optax
import einops
import pickle

Batch = collections.namedtuple("Batch", ["observations", "actions", "rewards", "masks", "next_observations"])
PrecisionLike = Union[None, str, lax.Precision, Tuple[str, str], Tuple[lax.Precision, lax.Precision]]


def default_init(scale: Optional[float] = None):
    if scale is None:
        scale = jnp.sqrt(2)
    return nn.initializers.orthogonal(scale)


PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
Variables = flax.core.FrozenDict[str, Any]
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


class SNDense(nn.Module):
    """A linear transformation applied over the last dimension of the input with sigmaReparam.
    Attributes:
        features: the number of output features.
        use_bias: whether to add a bias to the output (default: True).
        dtype: the dtype of the computation (default: infer from input and params).
        param_dtype: the dtype passed to parameter initializers (default: float32).
        precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
        kernel_init: initializer function for the weight matrix.
        bias_init: initializer function for the bias.
    """

    features: int
    use_bias: bool = True
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    precision: PrecisionLike = None
    bias_init: Callable[[PRNGKey, Shape, Dtype], jax.Array] = initjax.zeros
    std_init: float = 0.1

    @nn.compact
    def __call__(self, inputs: Any) -> Any:
        """Applies a linear transformation to the inputs along the last dimension.
        Args:
            inputs: The nd-array to be transformed.
        Returns:
            The transformed input.
        """
        initializing = self.is_mutable_collection("params")

        kernel = self.param(
            "kernel",
            initjax.normal(self.std_init),
            (jnp.shape(inputs)[-1], self.features),
            self.param_dtype,
        )

        if self.use_bias:
            bias = self.param("bias", self.bias_init, (self.features,), self.param_dtype)
        else:
            bias = None

        # fake init
        s = jnp.ones((1, 1))
        vh = jnp.ones((1))
        if initializing:
            _, s, vh = lax.stop_gradient(jnp.linalg.svd(kernel, full_matrices=False))
        sigma_param = self.param("sigma", initjax.ones, (1,), self.param_dtype)
        spectral_u_var = self.variable("spectral", "u", lambda shape: jnp.ones(shape) * vh[0], vh[0].shape)
        spectral_norm_var = self.variable("spectral", "norm", lambda shape: jnp.ones(shape) * s[0], (1,))
        inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)
        # power method to compute spectral norm
        u = spectral_u_var.value
        v = lax.stop_gradient(jnp.matmul(kernel, u))
        # l2 norm
        v = lax.stop_gradient(v / jnp.linalg.norm(v, ord=2))
        u = lax.stop_gradient(jnp.matmul(jnp.transpose(kernel), v))
        # l2 norm
        u = lax.stop_gradient(u / jnp.linalg.norm(u, ord=2))
        if spectral_u_var.is_mutable() and not initializing:
            spectral_u_var.value = u
        sigma = jnp.einsum("c,cd,d->", v, kernel, u)

        if spectral_norm_var.is_mutable() and not initializing:
            spectral_norm_var.value = sigma

        inputs, sigma_param, sigma = promote_dtype(inputs, sigma_param, sigma, dtype=self.dtype)
        y = lax.dot_general(
            inputs,
            (sigma_param / sigma) * kernel,
            (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
        )
        if bias is not None:
            y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
        return y


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


class FeedForward(nn.Module):
    dim: int = 256
    out_dim: int = 256
    dropout: float = 0.0
    use_bias: bool = False
    kernel_init: Callable = nn.initializers.xavier_uniform()
    bias_init: Callable = nn.initializers.zeros
    deterministic: Optional[bool] = None
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.leaky_relu
    use_sigmareparam: bool = True

    @nn.compact
    def __call__(self, batch, deterministic=None):
        deterministic = nn.merge_param("deterministic", self.deterministic, deterministic)
        if self.use_sigmareparam:
            x = SNDense(
                features=self.out_dim,
                use_bias=self.use_bias,
                bias_init=self.bias_init,
                name="fc1",
            )(batch)
        else:
            x = nn.Dense(
                self.dim,
                use_bias=self.use_bias,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                name="fc1",
            )(batch)

        x = self.activations(x)
        x = nn.Dropout(self.dropout)(x, deterministic)
        if self.use_sigmareparam:
            x = SNDense(
                features=self.out_dim,
                use_bias=self.use_bias,
                bias_init=self.bias_init,
                name="fc2",
            )(x)
        else:
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
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    use_sigmareparam: bool = True

    @nn.compact
    def __call__(self, batch, deterministic=None, custom_mask=None):
        deterministic = nn.merge_param("deterministic", self.deterministic, deterministic)
        if self.use_sigmareparam:
            qkv = SNDense(
                features=self.dim * 3,
                use_bias=self.use_bias,
                bias_init=self.bias_init,
            )(batch)
        else:
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
            mask = jnp.tril(jnp.ones((n, n)))[None, None, ...]
            mask = jnp.broadcast_to(mask, attention.shape)

        big_neg = jnp.finfo(attention.dtype).min
        attention = jnp.where(mask == 0, big_neg, attention)
        attention = nn.softmax(attention, axis=-1)
        attention = nn.Dropout(self.att_drop)(attention, deterministic)

        x = einops.rearrange(attention @ v, "b h n d -> b n (h d)")
        if self.use_sigmareparam:
            x = SNDense(
                features=self.dim,
                use_bias=self.use_bias,
                bias_init=self.bias_init,
            )(x)
        else:
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
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.leaky_relu
    use_sigmareparam: bool = True

    @nn.compact
    def __call__(self, batch, deterministic=False, custom_mask=None):
        x = nn.LayerNorm()(batch)
        x = Attention(
            dim=self.dim,
            num_heads=self.num_heads,
            use_bias=True,
            att_drop=self.att_drop,
            proj_drop=self.drop,
            use_sigmareparam=self.use_sigmareparam,
        )(x, deterministic, custom_mask)
        x = nn.LayerNorm()(x)
        batch = batch + x

        x = nn.LayerNorm()(batch)
        x = FeedForward(
            dim=self.dim * self.mlp_ratio,
            out_dim=self.dim,
            dropout=self.drop,
            use_bias=True,
            activations=self.activations,
            use_sigmareparam=self.use_sigmareparam,
        )(x, deterministic)
        return batch + x


class ConcatEncoder(nn.Module):
    obs_keys: Sequence[str] = ("image1", "image2", "text_feature")

    @nn.compact
    def __call__(self, observations: Dict[str, jnp.ndarray], deterministic=False):
        features = {}
        for k, v in observations.items():
            if v.ndim == 2:
                v = v[jnp.newaxis]
            if k in self.obs_keys:
                features[k] = v
        batch_size, num_timestep, _ = features[self.obs_keys[0]].shape
        features = concat_multiple_emb(features)
        features = jnp.reshape(features, (batch_size, -1))
        return features


class TransformerEncoder(nn.Module):
    emb_dim: int = 1024
    depth: int = 2
    att_drop: float = 0.0
    drop: float = 0.0
    num_heads: int = 8
    mlp_ratio: int = 4
    obs_keys: Sequence[str] = ("image1", "image2", "text_feature")
    stop_gradient: bool = False
    normalize_inputs: bool = True
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.leaky_relu
    use_sigmareparam: bool = True
    return_intermeidate: bool = False

    @nn.compact
    def __call__(self, observations: Dict[str, jnp.ndarray], deterministic=False, custom_mask=None):
        features = {}
        for k, v in observations.items():
            if v.ndim == 2:
                v = v[jnp.newaxis]
            if k in self.obs_keys:
                features[k] = v
        batch_size, num_timestep, _ = features[self.obs_keys[0]].shape

        features = concat_multiple_emb(features)
        features = InputNorm(features.shape[-1], skip=not self.normalize_inputs)(features, deterministic=deterministic)
        embed = MLP(
            [self.emb_dim, self.emb_dim, self.emb_dim],
            dropout_rate=self.drop,
            activations=self.activations,
            name="FeatureMLP",
        )(features, training=not deterministic)
        embed = embed + get_1d_sincos_pos_embed(embed.shape[-1], num_timestep)
        embed = nn.LayerNorm()(embed)

        x = embed
        intermediate_values = []
        for _ in range(self.depth):
            x = Block(
                dim=self.emb_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                att_drop=self.att_drop,
                drop=self.drop,
                activations=self.activations,
                use_sigmareparam=self.use_sigmareparam,
            )(x, deterministic, custom_mask)
            intermediate_values.append(x)

        x = nn.LayerNorm()(x)
        if self.stop_gradient:
            x = lax.stop_gradient(x)
        if self.return_intermeidate:
            return x, intermediate_values
        else:
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
    stop_gradient: bool = False

    @nn.compact
    def __call__(self, observations: Dict[str, jnp.ndarray], deterministic=False, custom_mask=None):
        assert "text_feature" in observations, "text_feature must be in observations"
        image_features = {}
        for k, v in observations.items():
            if v.ndim == 2:
                v = v[jnp.newaxis]
            if k in self.obs_keys:
                image_features[k] = v
        batch_size, num_timestep, _ = image_features[self.obs_keys[0]].shape
        image_features = concat_multiple_emb(image_features)
        image_features = MLP([self.emb_dim], dropout_rate=self.drop, name="ImageFeatureMLP")(image_features)
        # image_embed = image_features + get_1d_sincos_pos_embed(image_features.shape[-1], num_timestep)

        timesteps = jnp.tile(jnp.arange(num_timestep, dtype=jnp.int32), (batch_size, 1))
        embed_timestep = nn.Embed(num_timestep, features=self.emb_dim, name="TimestepEmbed")(timesteps)
        image_embed = image_features + embed_timestep

        text_feature = observations["text_feature"]
        text_feature = MLP([self.emb_dim], dropout_rate=self.drop, name="TextFeatureMLP")(text_feature)
        # text_embed = text_feature + get_1d_sincos_pos_embed(text_feature.shape[-1], num_timestep)

        text_embed_timestep = nn.Embed(num_timestep, features=self.emb_dim, name="TextTimestepEmbed")(timesteps)
        text_embed = text_feature + text_embed_timestep

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
        if self.stop_gradient:
            x = lax.stop_gradient(x)
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


def concat_multiple_emb(img_emb: dict):
    return jnp.concatenate([jnp.asarray(elem) for elem in img_emb.values()], axis=-1)
    # num_features, batch_size, num_timestep = img_emb.shape[:3]
    # img_emb = jnp.reshape(img_emb, (batch_size * num_features, num_timestep, -1))
    # img_emb = jnp.concatenate(jnp.split(img_emb, num_features, axis=0), -1)  # (batch_size, num_timestep, emb_dim)
    # return img_emb


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
    extra_variables: Params = None

    @classmethod
    def create(
        cls, model_def: nn.Module, inputs: Sequence[jnp.ndarray], tx: Optional[optax.GradientTransformation] = None
    ) -> "Model":
        variables = model_def.init(*inputs)

        _, params = variables.pop("params")
        if len(variables) > 1:
            extra_variables = {}
            for key in variables.keys():
                if key != "params":
                    extra_variables[key] = variables[key]
        else:
            extra_variables = {}

        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(
            step=1, apply_fn=model_def, params=params, tx=tx, opt_state=opt_state, extra_variables=extra_variables
        )

    def __call__(self, *args, **kwargs):
        variables = {"params": self.params}
        variables.update(self.extra_variables)
        return self.apply_fn.apply(variables, *args, **kwargs)

    def apply(self, *args, **kwargs):
        return self.apply_fn.apply(*args, **kwargs)

    def apply_gradient(self, loss_fn) -> Tuple[Any, "Model"]:
        variables = {"params": self.params}
        variables.update(self.extra_variables)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        # grads, info = grad_fn(variables)
        (_, info), grads = grad_fn(variables)

        updates, new_opt_state = self.tx.update(grads["params"], self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        new_variables = info["updated_states"]
        info.pop("updated_states")

        return self.replace(
            step=self.step + 1, params=new_params, opt_state=new_opt_state, extra_variables=new_variables
        ), info

    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        data = {
            "params": self.params,
            "extra_variables": self.extra_variables,
        }
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
            # f.write(flax.serialization.to_bytes(self.params))

    def load(self, load_path: str) -> "Model":
        with open(load_path, "rb") as f:
            data = pickle.load(f)
            # params = flax.serialization.from_bytes(self.params, f.read())
        return self.replace(params=data["params"], extra_variables=data["extra_variables"])


MAGIC_PAD_VAL = -1  # Define this according to your needs


class InputNorm(nn.Module):
    dim: int
    beta: float = 1e-4
    init_nu: float = 1.0
    skip: bool = False
    pad_val: float = MAGIC_PAD_VAL

    def setup(self):
        self.mu = self.variable("batch_stats", "mu", lambda shape: jnp.zeros(shape), (self.dim,))
        self.nu = self.variable("batch_stats", "nu", lambda shape: jnp.ones(shape) * self.init_nu, (self.dim,))
        self.t = self.variable("batch_stats", "t", lambda shape: jnp.ones(shape), ())

    def sigma(self):
        mu, nu = self.mu.value, self.nu.value
        sigma_ = jnp.sqrt(nu - mu**2 + 1e-5)
        return jnp.clip(jnp.nan_to_num(sigma_), 1e-3, 1e6)

    def normalize_values(self, val):
        if self.skip:
            return val
        sigma = self.sigma()
        normalized = jnp.clip((val - self.mu.value) / sigma, -1e4, 1e4)
        not_nan = ~jnp.isnan(normalized)
        stable = sigma > 0.01
        use_norm = jnp.logical_and(stable, not_nan)
        output = jnp.where(use_norm, normalized, val - jnp.nan_to_num(self.mu.value))
        return output

    def denormalize_values(self, val):
        if self.skip:
            return val
        sigma = self.sigma()
        denormalized = (val * sigma) + self.mu.value
        stable = sigma > 0.01
        output = jnp.where(stable, denormalized, val + jnp.nan_to_num(self.mu.value))
        return output

    def masked_stats(self, val):
        mask = ~(val == self.pad_val)
        sum_ = (val * mask).sum((0, 1))
        square_sum = ((val * mask) ** 2).sum((0, 1))
        total = mask.sum((0, 1))
        mean = sum_ / total
        square_mean = square_sum / total
        return mean, square_mean

    def update_stats(self, val, mu, nu, t):
        new_t = t + 1
        beta_t = self.beta / (1.0 - (1.0 - self.beta) ** new_t)
        mean, square_mean = self.masked_stats(val)
        new_mu = (1.0 - beta_t) * mu + (beta_t * mean)
        new_nu = (1.0 - beta_t) * nu + (beta_t * square_mean)
        return new_mu, new_nu, new_t

    def __call__(self, x, denormalize=False, deterministic=False):
        if denormalize:
            val = self.denormalize_values(x)
        else:
            val = self.normalize_values(x)

        if not deterministic:
            mu, nu, t = self.update_stats(val, self.mu.value, self.nu.value, self.t.value)
            self.mu.value = mu
            self.nu.value = nu
            self.t.value = t

        return val
