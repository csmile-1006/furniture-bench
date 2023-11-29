from typing import Callable, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import einops


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
    alibi_bias: bool = False

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

        mh_fn = lambda x: einops.rearrange(x, "b n (h d) -> b h n d", h=self.num_heads)
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
    alibi_bias: bool = False

    @nn.compact
    def __call__(self, batch, deterministic=False, custom_mask=None):
        x = nn.LayerNorm()(batch)
        x = Attention(
            self.dim,
            self.num_heads,
            True,
            self.att_drop,
            self.drop,
            alibi_bias=self.alibi_bias,
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
    alibi_bias: bool = False

    @nn.compact
    def __call__(self, x, deterministic=False, custom_mask=None):
        for _ in range(self.depth):
            x = Block(
                self.emb_dim,
                self.num_heads,
                self.mlp_ratio,
                self.att_drop,
                self.drop,
                self.alibi_bias,
            )(x, deterministic, custom_mask)

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
