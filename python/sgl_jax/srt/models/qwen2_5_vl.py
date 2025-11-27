import math
from functools import partial
from typing import (Callable, NamedTuple, Optional, Any)

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from flax import nnx
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig

from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.embeddings import RotaryEmbedding
from sgl_jax.srt.layers.attention.native_backend import NativeAttention

import time


class SegmentIds(NamedTuple):
    """SegmentIds for Q and KV sequences.

  SegmentIds are used to generate segment mask, which prevents attention between
  different segments in the input sequence. Each array is a list of ids
  (integers).
  Only the token with the same id can attend to each other.

  Attributes:
    q: segment ids along the Q sequence.
    kv: segment ids along the KV sequence.
  """

    q: jax.Array  # [batch_size, q_seq_len]
    kv: jax.Array  # [batch_size, kv_seq_len]


def apply_rotary_pos_emb_vision(x: jax.Array,
                                rotary_pos_emb: jax.Array) -> jax.Array:
    # x: [B, T, N, H]
    # rotary_pos_emb: [T, H//2]
    _, _, _, H = x.shape
    half_dim = H // 2

    # [B, T, N, H//2]
    x_real = x[..., :half_dim]
    x_imag = x[..., half_dim:]

    # [T, H//2]
    cos_emb = jnp.cos(rotary_pos_emb)
    sin_emb = jnp.sin(rotary_pos_emb)

    # [1, T, 1, H//2]
    cos_emb = cos_emb[None, :, None, :]
    sin_emb = sin_emb[None, :, None, :]

    # [B, T, N, H//2]
    x_rotated_real = x_real * cos_emb - x_imag * sin_emb
    x_rotated_imag = x_real * sin_emb + x_imag * cos_emb

    # [B, T, N, H]
    x_rotated = jnp.concatenate([x_rotated_real, x_rotated_imag], axis=-1)

    return x_rotated


def generate_window_segment_ids(cu_seqlens: jax.Array, seq_len: int,
                                padded_seq_len: int) -> SegmentIds:
    """Generates segment IDs for windowed attention

    Args:
        cu_seqlens: A 1D array of cumulative sequence lengths for each window.
            e.g., [0, len_win0, len_win0+len_win1, ...]

    Returns:
        A SegmentIds object for flash_attention.
    """
    indices = jnp.arange(seq_len, dtype=jnp.int32)
    segment_ids = jnp.searchsorted(cu_seqlens[1:], indices, side='right') + 1
    padding_segment_ids = jnp.zeros(padded_seq_len - seq_len, dtype=jnp.int32)
    segment_ids = jnp.concatenate([segment_ids, padding_segment_ids])
    segment_ids = segment_ids.reshape(1, -1)

    return SegmentIds(q=segment_ids, kv=segment_ids)


def get_padded_num_heads(num_heads: int, sharding_size: int) -> int:
    if num_heads >= sharding_size:
        assert num_heads % sharding_size == 0
    else:
        assert sharding_size % num_heads == 0
        num_heads = sharding_size
    return num_heads


class Qwen2_5_VisionMLP(nnx.Module):

    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            dtype: jnp.dtype = jnp.bfloat16,
            rngs: nnx.Rngs = None,
            mesh: Mesh = None
    ):
        self.gate_proj = LinearBase(
            hidden_size,
            intermediate_size,
            kernel_axes=(None, "tensor"),
            use_bias=True,
            params_dtype=dtype,
            rngs=rngs,
            mesh=mesh,
        )
        self.up_proj = LinearBase(
            hidden_size,
            intermediate_size,
            kernel_axes=(None, "tensor"),
            use_bias=True,
            params_dtype=dtype,
            rngs=rngs,
            mesh=mesh,
        )
        self.down_proj = LinearBase(
            intermediate_size,
            hidden_size,
            kernel_axes=("tensor", None),
            use_bias=True,
            params_dtype=dtype,
            rngs=rngs,
            mesh=mesh,
        )
        self.act_fn = jax.nn.swish

    def __call__(self, x: jax.Array) -> jax.Array:
        gate = self.act_fn(self.gate_proj(x)[0])
        up = self.up_proj(x)[0]
        fuse = gate * up
        result = self.down_proj(fuse)[0]
        return result


class Attention(nnx.Module):

    def __init__(
            self,
            mesh: Mesh = None,
            scale: Optional[float] = None,
    ):
        self.mesh = mesh
        self.scale = scale

    def __call__(self, q, k, v, mask=None):
        attn_logits = jnp.einsum("bnth,bnsh->bnts", q, k) * self.scale

        # Apply appropriate masking
        if mask is not None:
            mask_value = jnp.finfo(attn_logits.dtype).min
            attn_logits = jnp.where(mask, attn_logits, mask_value)

        # Softmax
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)

        attn_output = jnp.matmul(attn_weights, v)
        return attn_output


class Qwen2_5_VisionAttention(nnx.Module):

    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            rope_theta: float = 1000000,
            rope_scaling: dict[str, Any] | None = None,
            head_dim: int | None = None,
            dtype: jnp.dtype = jnp.bfloat16,
            rngs: nnx.Rngs = None,
            mesh: Mesh = None,
    ):
        self.num_heads = num_heads
        self.num_kv_heads = self.num_heads
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

        sharding_size = mesh.shape["tensor"]
        self.num_heads = get_padded_num_heads(self.num_heads,
                                              sharding_size)
        self.num_kv_heads = get_padded_num_heads(self.num_kv_heads,
                                                 sharding_size)

        # TODO: Consider padding in future
        self.head_dim = head_dim or hidden_size // self.num_heads

        self.mesh = mesh

        self.qkv_proj = LinearBase(
            hidden_size,
            3 * hidden_size,
            kernel_axes=(None, "tensor"),
            use_bias=True,
            params_dtype=dtype,
            rngs=rngs,
            mesh=mesh,
        )

        self.o_proj = LinearBase(
            hidden_size,
            hidden_size,
            kernel_axes=("tensor", None),
            use_bias=True,
            params_dtype=dtype,
            rngs=rngs,
            mesh=mesh,
        )

        self.attention = Attention(
            mesh=mesh,
            scale=1.0 / math.sqrt(self.head_dim),
        )

    def __call__(
        self,
        x: jax.Array,
        rotary_pos_emb: jax.Array,
        cu_window_seqlens: Optional[jax.Array] = None,
        use_fullattn: bool = True,
    ) -> jax.Array:
        T, B, D = x.shape
        assert B == 1, "Vision attention currently only supports batch size 1"
        # [T, B, D] -> [T, B, 3 * D]
        qkv, _ = self.qkv_proj(x)

        # Split into Q, K, V.
        # NOTE: simplified from vLLM's split_qkv,
        # may need to revisit for tp>1
        # [T, B, 3 * D] -> 3 *[T, B, D]
        q, k, v = jnp.split(qkv, 3, axis=-1)

        # [T, B, N, H]
        q = q.reshape(T, B, self.num_heads, self.head_dim)
        k = k.reshape(T, B, self.num_heads, self.head_dim)
        v = v.reshape(T, B, self.num_heads, self.head_dim)

        # [T, B, N, H] -> [B, T, N, H]
        q = jnp.transpose(q, (1, 0, 2, 3))
        k = jnp.transpose(k, (1, 0, 2, 3))
        v = jnp.transpose(v, (1, 0, 2, 3))

        # rotary_pos_emb shape: (T, H)
        q = apply_rotary_pos_emb_vision(q, rotary_pos_emb)
        k = apply_rotary_pos_emb_vision(k, rotary_pos_emb)

        # NOTE: an extra transpose because we need to
        # align the correctness with vLLM's design.
        # Might be able to remove one once implemented.
        # [B, T, N, H] -> [B, N, T, H]
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # Pad the sequence length to be a multiple of 128 for flash_attention
        block_k_major = 128
        T_attn = q.shape[2]
        padded_T = (T_attn + block_k_major -
                    1) // block_k_major * block_k_major
        pad_width = ((0, 0), (0, 0), (0, padded_T - T_attn), (0, 0))

        q = jnp.pad(q, pad_width, 'constant')
        k = jnp.pad(k, pad_width, 'constant')
        v = jnp.pad(v, pad_width, 'constant')

        mask_shape = (q.shape[-2], k.shape[-2])
        row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
        col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
        mask = col_ids <= row_ids

        if cu_window_seqlens is not None:
            segment_ids = generate_window_segment_ids(cu_window_seqlens, T_attn,
                                                      padded_T)
            window_mask = jnp.equal(jnp.repeat(segment_ids.q, segment_ids.kv.shape[-1], 0),
                             segment_ids.kv.transpose(1, 0)).astype(jnp.bool_)
            mask = jnp.logical_and(mask, window_mask)
        mask = mask[None, None, :, :]

        # TODO: add support for quantized KV cache?
        output = self.attention(q, k, v, mask)

        # Unpad the output
        output = output[:, :, :T_attn, :]

        # [B, N, T, H] -> [T, B, N, H]
        output = jnp.transpose(output, (2, 0, 1, 3))

        output = output.reshape(T, B, D)

        output = self.o_proj(output)

        return output[0]


class Qwen2_5_VisionBlock(nnx.Module):

    def __init__(
            self,
            config: Qwen2_5_VLVisionConfig,
            dtype: jnp.dtype = jnp.bfloat16,
            rngs: nnx.Rngs = None,
            mesh: Mesh = None,
    ):
        dim = config.hidden_size
        norm_layer = partial(nnx.RMSNorm,
                             epsilon=config.rms_norm_eps,
                             scale_init=nnx.with_partitioning(
                                 nnx.initializers.uniform(), (None, )))

        self.norm1 = norm_layer(dim, dtype=dtype, rngs=rngs)
        self.norm2 = norm_layer(dim, dtype=dtype, rngs=rngs)
        self.attn = Qwen2_5_VisionAttention(hidden_size=config.hidden_size,
                                            num_heads=config.num_heads,
                                            rope_theta=config.rope_theta,
                                            rope_scaling=config.rope_scaling,
                                            head_dim=config.head_dim,
                                            dtype=dtype,
                                            rngs=rngs,
                                            mesh=mesh)
        self.mlp = Qwen2_5_VisionMLP(hidden_size=config.hidden_size,
                                     intermediate_size=config.intermediate_size,
                                     dtype=dtype,
                                     rngs=rngs,
                                     mesh=mesh)

    def __call__(self,
                 x: jax.Array,
                 rotary_pos_emb: jax.Array,
                 cu_window_seqlens: Optional[jax.Array] = None,
                 use_fullattn: bool = True) -> jax.Array:
        x = x + self.attn(self.norm1(x), rotary_pos_emb, cu_window_seqlens,
                          use_fullattn)
        x = x + self.mlp(self.norm2(x))

        return x


class Qwen2_5_VisionPatchEmbed(nnx.Module):

    def __init__(
        self,
        rngs: nnx.Rngs,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        hidden_size: int = 1152,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.hidden_size = hidden_size
        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.proj = nnx.Conv(in_features=in_channels,
                             out_features=hidden_size,
                             kernel_size=kernel_size,
                             strides=kernel_size,
                             use_bias=False,
                             param_dtype=dtype,
                             kernel_init=nnx.with_partitioning(
                                 nnx.initializers.uniform(),
                                 (None, None, None, None, "tensor")
                             ),
                             rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        # x is (L, C * T * H * W)
        L, dim = x.shape
        C = dim // (self.temporal_patch_size * self.patch_size *
                    self.patch_size)
        # Reshape to (L, T, H, W, C) for Conv3D with channels_last
        x = x.reshape(L, C, self.temporal_patch_size, self.patch_size,
                      self.patch_size)
        # L,T,H,W,C
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        x = self.proj(x)
        # After conv, shape is (L, T_out, H_out, W_out, C_out)
        # With stride=kernel_size, T_out=H_out=W_out=1.
        # So shape is (L, 1, 1, 1, hidden_size)
        x = x.reshape(L, self.hidden_size)
        return x


class Qwen2_5_VisionPatchMerger(nnx.Module):

    def __init__(
            self,
            d_model: int,
            context_dim: int,
            norm_layer: Callable,
            spatial_merge_size: int,
            dtype: jnp.dtype = jnp.bfloat16,
            rngs: nnx.Rngs = None,
            mesh: Mesh = None
    ):
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = norm_layer(context_dim,
                               dtype=dtype,
                               rngs=rngs,
                               scale_init=nnx.with_partitioning(
                                   nnx.initializers.uniform(),
                                   (None, )
                               ))
        self.mlp_fc1 = LinearBase(
            self.hidden_size,
            self.hidden_size,
            kernel_axes=(None, "tensor"),
            use_bias=True,
            params_dtype=dtype,
            rngs=rngs,
            mesh=mesh,
        )
        self.mlp_act = jax.nn.gelu
        self.mlp_fc2 = LinearBase(
            self.hidden_size,
            d_model,
            kernel_axes=("tensor", None),
            use_bias=True,
            params_dtype=dtype,
            rngs=rngs,
            mesh=mesh,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.ln_q(x)
        x = x.reshape(-1, self.hidden_size)
        x = self.mlp_fc1(x)[0]
        x = self.mlp_act(x)
        x = self.mlp_fc2(x)[0]
        return x


class Qwen2_5_VisionRotaryEmbedding(nnx.Module):

    def __init__(self, dim: int, theta: float = 10000.0):
        self.dim = dim
        self.theta = theta

    def __call__(self, seqlen: int) -> jax.Array:
        inv_freq = 1.0 / (self.theta**(
            jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim))
        seq = jnp.arange(seqlen, dtype=jnp.float32)
        freqs = jnp.outer(seq, inv_freq)
        return freqs.astype(jnp.bfloat16)


class Qwen2_5_VisionTransformer(nnx.Module):

    def __init__(self,
                 config: Qwen2_5_VLVisionConfig,
                 norm_eps: float = 1e-6,
                 dtype: jnp.dtype = jnp.bfloat16,
                 rngs: nnx.Rngs = None,
                 mesh: Mesh = None):
        # args for get_window_index_thw
        self.window_size = config.window_size
        self.patch_size = config.patch_size
        self.spatial_merge_size = config.spatial_merge_size
        self.fullatt_block_indexes = config.fullatt_block_indexes
        self.spatial_merge_unit = config.spatial_merge_size**2

        self.patch_embed = Qwen2_5_VisionPatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            hidden_size=config.hidden_size,
            dtype=dtype,
            rngs=rngs)

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)

        self.blocks = [
            Qwen2_5_VisionBlock(
                config=config,
                dtype=dtype,
                rngs=rngs,
                mesh=mesh,
            ) for _ in range(config.depth)
        ]
        self.merger = Qwen2_5_VisionPatchMerger(
            d_model=config.out_hidden_size,
            context_dim=config.hidden_size,
            norm_layer=partial(nnx.RMSNorm, epsilon=norm_eps),
            spatial_merge_size=config.spatial_merge_size,
            dtype=dtype,
            rngs=rngs,
            mesh=mesh,
        )

    def rotary_pos_emb_thw(self, t, h, w):
        # hpos_ids: [h, w], wpos_ids: [h, w]
        hpos_ids, wpos_ids = jnp.indices((h, w))
        # hpos_ids: [h, w] -> [(h / spatial_merge_size) *
        #                      (w / spatial_merge_size) *
        #                      spatial_merge_size       *
        #                      spatial_merge_size]
        hpos_ids = hpos_ids.reshape(
            h // self.spatial_merge_size,
            self.spatial_merge_size,
            w // self.spatial_merge_size,
            self.spatial_merge_size,
        ).transpose(0, 2, 1, 3).flatten()
        # wpos_ids: [h, w] -> [(h / spatial_merge_size) *
        #                      (w / spatial_merge_size) *
        #                      spatial_merge_size       *
        #                      spatial_merge_size]
        wpos_ids = wpos_ids.reshape(
            h // self.spatial_merge_size,
            self.spatial_merge_size,
            w // self.spatial_merge_size,
            self.spatial_merge_size,
        ).transpose(0, 2, 1, 3).flatten()
        # pos_ids: [(h / spatial_merge_size) *
        #           (w / spatial_merge_size) *
        #           spatial_merge_size       *
        #           spatial_merge_size, 2]
        pos_ids = jnp.stack([hpos_ids, wpos_ids], axis=-1)
        # pos_ids: [t * (h / spatial_merge_size) *
        #           (w / spatial_merge_size) *
        #           spatial_merge_size       *
        #           spatial_merge_size, 2]
        pos_ids = jnp.tile(pos_ids, (t, 1))

        max_size = max(h, w)
        # rotary_pos_emb_full: [max_size, head_dim // 4]
        rotary_pos_emb_full = self.rotary_pos_emb(max_size)
        # rotary_pos_emb: [t * h * w, head_dim // 2]
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].reshape(
            pos_ids.shape[0], -1)
        # rotary_pos_emb: [t * h * w / (spatial_merge_size*spatial_merge_size),
        #                  spatial_merge_size*spatial_merge_size,
        #                  head_dim // 2]
        rotary_pos_emb = rotary_pos_emb.reshape(
            rotary_pos_emb.shape[0] // self.spatial_merge_unit,
            self.spatial_merge_unit, -1)

        return rotary_pos_emb

    def get_window_index_thw(self, grid_t, grid_h, grid_w):
        vit_merger_window_size = (self.window_size //
                                  self.spatial_merge_size // self.patch_size)

        llm_grid_h = grid_h // self.spatial_merge_size
        llm_grid_w = grid_w // self.spatial_merge_size

        index = jnp.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
            grid_t, llm_grid_h, llm_grid_w)

        pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
        pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
        num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
        num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size

        index_padded = jnp.pad(index, ((0, 0), (0, pad_h), (0, pad_w)),
                               constant_values=-100)
        index_padded = index_padded.reshape(grid_t, num_windows_h,
                                            vit_merger_window_size,
                                            num_windows_w,
                                            vit_merger_window_size)
        index_padded = jnp.transpose(index_padded, (0, 1, 3, 2, 4)).reshape(
            grid_t, num_windows_h * num_windows_w, vit_merger_window_size,
            vit_merger_window_size)
        seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
        index_padded = index_padded.reshape(-1)
        # The number of valid indices is static because grid_t, grid_h, grid_w
        # are static.
        num_valid_indices = grid_t * llm_grid_h * llm_grid_w
        valid_indices = jnp.nonzero(index_padded != -100,
                                    size=num_valid_indices)[0]
        index_new = index_padded[valid_indices]
        cu_seqlens_tmp = jnp.cumsum(seqlens) * self.spatial_merge_unit
        cu_seqlens_tmp = cu_seqlens_tmp.astype(jnp.int32)

        # NOTE (wenlong): Pytorch code uses this to reduce replication,
        # but I don't think there is a need here, plus it would cause problem in JIT
        # Please refer here if there is a problem down-stream
        # cu_seqlens_tmp = jnp.unique(cu_seqlens_tmp)

        return index_new, cu_seqlens_tmp

    def get_rope_by_thw(self, t, h, w):
        window_index_thw, cu_seqlens_window_thw = self.get_window_index_thw(
            t, h, w)

        rotary_pos_emb_thw = self.rotary_pos_emb_thw(t, h, w)

        rotary_pos_emb_thw = rotary_pos_emb_thw[window_index_thw, :, :]
        rotary_pos_emb_thw = rotary_pos_emb_thw.reshape(
            -1, rotary_pos_emb_thw.shape[-1])
        cu_seqlens_thw = jnp.full(t, h * w, dtype=jnp.int32)

        return (rotary_pos_emb_thw, window_index_thw, cu_seqlens_window_thw,
                cu_seqlens_thw)

    def __call__(self, x: jax.Array, grid_thw: tuple[tuple[int, int,
                                                           int]]) -> jax.Array:
        # x: pixel_values: jax.Array
        # """Shape:
        # `(num_patches, num_channels * patch_size * patch_size)`
        # """

        # grid_thw: image_grid_thw: jax.Array
        # """Shape: `(num_images, 3)`
        # This should be in `(grid_t, grid_h, grid_w)` format.
        # """
        hidden_states = self.patch_embed(x)

        # num of patches
        seq_len = x.shape[0]
        # num of images/videoes
        num_grids = len(grid_thw)

        rotary_pos_emb = []
        window_index = []
        cu_window_seqlens = [jnp.array([0], dtype=jnp.int32)]
        cu_seqlens = []

        window_index_id = 0
        cu_window_seqlens_last = 0
        for t, h, w in grid_thw:

            llm_h = h // self.spatial_merge_size
            llm_w = w // self.spatial_merge_size

            (
                rotary_pos_emb_thw,
                window_index_thw,
                cu_seqlens_window_thw,
                cu_seqlens_thw,
            ) = self.get_rope_by_thw(t, h, w)

            window_index.append(window_index_thw + window_index_id)
            window_index_id += (t * llm_h * llm_w)

            cu_seqlens_window_thw += cu_window_seqlens_last
            cu_window_seqlens_last = cu_seqlens_window_thw[-1]
            cu_window_seqlens.append(cu_seqlens_window_thw)

            rotary_pos_emb.append(rotary_pos_emb_thw)

            cu_seqlens.append(cu_seqlens_thw)

        rotary_pos_emb = jnp.concatenate(rotary_pos_emb, axis=0)
        window_index = jnp.concatenate(window_index, axis=0)
        cu_window_seqlens = jnp.concatenate(cu_window_seqlens, axis=0)

        cu_seqlens = jnp.concatenate(cu_seqlens, axis=0)
        cu_seqlens = jnp.cumsum(cu_seqlens, axis=0, dtype=jnp.int32)
        cu_seqlens = jnp.pad(cu_seqlens, ((1, 0), ),
                             mode='constant',
                             constant_values=0)

        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        hidden_states = jnp.expand_dims(hidden_states, axis=1)

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                hidden_states = blk(hidden_states,
                                    rotary_pos_emb=rotary_pos_emb,
                                    cu_window_seqlens=cu_seqlens,
                                    use_fullattn=True)
            else:
                hidden_states = blk(hidden_states,
                                    rotary_pos_emb=rotary_pos_emb,
                                    cu_window_seqlens=cu_window_seqlens,
                                    use_fullattn=False)

        # adapter
        hidden_states = self.merger(hidden_states)
        reverse_indices = jnp.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]
        return hidden_states
