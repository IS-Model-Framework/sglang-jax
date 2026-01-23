import math
from functools import partial
from typing import List, Callable, NamedTuple, Optional, Any
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig,Qwen3VLVisionConfig,Qwen3VLTextConfig
from transformers import modeling_flax_utils
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.experimental import shard_map
from flax import nnx
import jax.image as jimage
from sgl_jax.srt.managers.schedule_batch import (
    MultimodalDataItem,
    MultimodalInputs,
)
from sgl_jax.srt.multimodal.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sgl_jax.srt.models.qwen2 import Qwen2Model
from sgl_jax.srt.layers.embeddings import ParallelLMHead
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping
from sgl_jax.utils import logger
from sgl_jax.srt.kernels.flash_attention import flash_attention
from sgl_jax.srt.models.qwen2_5_vl import Qwen2_5_VisionAttention
from sgl_jax.srt.configs.model_config import ModelConfig


#---VisionEncoder

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


class Qwen3_VisionMLP(nnx.Module):

    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            dtype: jnp.dtype = jnp.bfloat16,
            mesh: Mesh = None
    ):

        self.up_proj = LinearBase(
            hidden_size,
            intermediate_size,
            kernel_axes=(None, "tensor"),
            use_bias=True,
            params_dtype=dtype,
            mesh=mesh,
        )
        self.down_proj = LinearBase(
            intermediate_size,
            hidden_size,
            kernel_axes=("tensor", None),
            use_bias=True,
            params_dtype=dtype,
            mesh=mesh,
        )
        self.act_fn = modeling_flax_utils.ACT2FN["gelu_pytorch_tanh"]

    def __call__(self, x: jax.Array) -> jax.Array:
        up = self.up_proj(x)[0]
        output = self.down_proj(self.act_fn(up))[0]
        return output
    
class Qwen3_VLVisionPatchEmbed(nnx.Module):

    def __init__(
        self,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        hidden_size: int = 1152,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs = None,
        mesh = None
    ):
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.proj = nnx.Conv(
            in_features=in_channels,
            out_features=hidden_size,
            kernel_size=kernel_size,
            strides=kernel_size,
            use_bias=True,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.lecun_normal(),
                (None, None, None, None, "tensor")
            ),
            bias_init=nnx.with_partitioning(
                nnx.initializers.zeros_init(),
                ("tensor",)
            ),
            rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        #NOTE：SGLang显式转换了dtype
        if x.dtype != self.proj.param_dtype:
            x = x.astype(self.proj.param_dtype)
        # x is (L, C * T * H * W)
        L, dim = x.shape
        C = dim // (self.temporal_patch_size * self.patch_size *
                    self.patch_size)
        # Reshape to (L, C, T, H, W) first
        x = x.reshape(L, C, self.temporal_patch_size, self.patch_size,
                      self.patch_size)
        # Transpose to (L, T, H, W, C) for Conv3D with channels_last format
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        x = self.proj(x)
        # After conv, shape is (L, T_out, H_out, W_out, C_out)
        # With stride=kernel_size, T_out=H_out=W_out=1.
        # So shape is (L, 1, 1, 1, hidden_size)
        x = x.reshape(L, self.hidden_size)
        return x
    
def get_padded_num_heads(num_heads: int, sharding_size: int) -> int:
    if num_heads >= sharding_size:
        assert num_heads % sharding_size == 0
    else:
        assert sharding_size % num_heads == 0
        num_heads = sharding_size
    return num_heads

def sharded_flash_attention(
    mesh: Mesh,
    causal: bool = True,
    sm_scale: Optional[float] = None,
    vmem_limit_bytes: int | None = None,
) -> Callable[..., Any]:
    in_specs = (
        P("data", "tensor", None, None),  # q
        P("data", "tensor", None, None),  # k
        P("data", "tensor", None, None),  # v
        P(),  # segment_ids
    )
    out_specs = P("data", "tensor", None, None)

    def _flash_attention(q, k, v, segment_ids):
        return flash_attention(q,
                               k,
                               v,
                               segment_ids=segment_ids,
                               sm_scale=sm_scale,
                               causal=causal,
                               vmem_limit_bytes=vmem_limit_bytes)

    return jax.jit(
        shard_map.shard_map(_flash_attention,
                            mesh=mesh,
                            in_specs=in_specs,
                            out_specs=out_specs,
                            check_rep=False))

class Qwen3_VisionAttention(nnx.Module):

    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            rope_theta: float = 5000000,
            rope_scaling: dict[str, Any] | None = None,
            head_dim: int | None = None,
            dtype: jnp.dtype = jnp.bfloat16,
            mesh: Mesh = None,
    ):
        self.num_heads = num_heads
        self.num_kv_heads = self.num_heads
        self.num_heads_original = num_heads
        self.num_kv_heads_original = self.num_kv_heads
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        if mesh is None:
            sharding_size = 1
        else:
            sharding_size = mesh.shape["tensor"]
        self.num_heads = get_padded_num_heads(self.num_heads,
                                              sharding_size)
        self.num_kv_heads = get_padded_num_heads(self.num_kv_heads,
                                                 sharding_size)

        self.head_dim = head_dim or hidden_size // self.num_heads_original
        self.heads_pad = self.num_heads - self.num_heads_original

        self.mesh = mesh

        self.qkv_proj = LinearBase(
            hidden_size,
            3 * hidden_size,
            kernel_axes=(None, "tensor"),
            use_bias=True,
            params_dtype=dtype,
            mesh=mesh,
        )

        self.o_proj = LinearBase(
            hidden_size,
            hidden_size,
            kernel_axes=("tensor", None),
            use_bias=True,
            params_dtype=dtype,
            mesh=mesh,
        )

        self.flash_attention = sharded_flash_attention(
            mesh=mesh,
            causal=False,
            sm_scale=1.0 / math.sqrt(self.head_dim),
            vmem_limit_bytes=128 * 1024 * 1024,
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
        q = q.reshape(T, B, self.num_heads_original, self.head_dim)
        k = k.reshape(T, B, self.num_kv_heads_original, self.head_dim)
        v = v.reshape(T, B, self.num_kv_heads_original, self.head_dim)

        if self.heads_pad:
            pad_width = ((0, 0), (0, 0), (0, self.heads_pad), (0, 0))
            q = jnp.pad(q, pad_width, "constant")
            k = jnp.pad(k, pad_width, "constant")
            v = jnp.pad(v, pad_width, "constant")

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

        segment_ids = generate_window_segment_ids(cu_window_seqlens, T_attn,
                                                  padded_T)

        # TODO (jacobplatin): add support for quantized KV cache?
        output = self.flash_attention(q, k, v, segment_ids)

        # Unpad the output
        output = output[:, :, :T_attn, :]

        if self.heads_pad:
            output = output[:, :self.num_heads_original, :, :]

        # [B, N, T, H] -> [T, B, N, H]
        output = jnp.transpose(output, (2, 0, 1, 3))

        output = output.reshape(T, B, D)

        output = self.o_proj(output)

        return output[0]


class Qwen3_VisionBlock(nnx.Module):

    def __init__(
            self,
            config: Qwen3VLConfig,
            norm_eps: float = 1e-6,
            dtype: jnp.dtype = jnp.bfloat16,
            rngs: nnx.Rngs = None,
            mesh: Mesh = None,
    ):
        dim = config.vision_config.hidden_size
        norm_layer = partial(nnx.LayerNorm,
                             epsilon=norm_eps,
                             scale_init=nnx.with_partitioning(
                                 nnx.initializers.uniform(), (None, )))

        self.norm1 = norm_layer(dim, dtype=dtype, rngs=rngs)
        self.norm2 = norm_layer(dim, dtype=dtype, rngs=rngs)
        self.attn = Qwen3_VisionAttention(hidden_size=config.vision_config.hidden_size,
                                            num_heads=config.vision_config.num_heads,
                                            rope_theta=config.text_config.rope_theta,
                                            rope_scaling=config.text_config.rope_scaling,
                                            head_dim=config.vision_config.hidden_size // config.vision_config.num_heads,
                                            dtype=dtype,
                                            mesh=mesh)
        self.mlp = Qwen3_VisionMLP(hidden_size=config.vision_config.hidden_size,
                                     intermediate_size=config.vision_config.intermediate_size,
                                     dtype=dtype,
                                     mesh=mesh)

    def __call__(self,
                 x: jax.Array,
                 rotary_pos_emb: jax.Array,
                 cu_seqlens: Optional[jax.Array] = None,
                 use_fullattn: bool = True) -> jax.Array:
        x = x + self.attn(self.norm1(x), rotary_pos_emb, cu_seqlens,
                          use_fullattn)
        x = x + self.mlp(self.norm2(x))

        return x
    

class Qwen3_VisionPatchMerger(nnx.Module):

    def __init__(
            self,
            config: Qwen3VLVisionConfig,
            use_postshuffle_norm: bool = False,
            dtype: jnp.dtype = jnp.bfloat16,
            rngs: nnx.Rngs = None,
            mesh: Mesh = None
    ):
        self.hidden_size = config.hidden_size * (config.spatial_merge_size ** 2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.ln_q = nnx.LayerNorm(
            self.hidden_size if self.use_postshuffle_norm else config.hidden_size,
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
            mesh=mesh,
        )
        self.mlp_act = modeling_flax_utils.ACT2FN["gelu"]
        self.mlp_fc2 = LinearBase(
            self.hidden_size,
            config.out_hidden_size,
            kernel_axes=("tensor", None),
            use_bias=True,
            params_dtype=dtype,
            mesh=mesh,
        )
        self.mesh = mesh

    def __call__(self, x: jax.Array) -> jax.Array:
        if self.use_postshuffle_norm:
            x = self.ln_q(
                    jax.lax.reshape(
                        x, 
                        (x.shape[0] * x.shape[1] * x.shape[2] // self.hidden_size, self.hidden_size),
                        out_sharding = P(None, "tensor")
                    )
                ) 
        else:
            x = self.ln_q(x)
            x = jax.lax.reshape(
                        x, 
                        (x.shape[0] * x.shape[1] * x.shape[2] // self.hidden_size, self.hidden_size),
                        out_sharding = P(None, "tensor")
                    )
        x = self.mlp_fc1(x)[0]
        x = self.mlp_act(x)
        x = self.mlp_fc2(x)[0]
        return x
    

class Qwen3_VisionRotaryEmbedding(nnx.Module):

    def __init__(self, dim: int, theta: float = 10000.0):
        self.dim = dim
        self.theta = theta

    def __call__(self, seqlen: int) -> jax.Array:
        inv_freq = 1.0 / (self.theta**(
            jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim))
        seq = jnp.arange(seqlen, dtype=jnp.float32)
        freqs = jnp.outer(seq, inv_freq)
        return freqs.astype(jnp.bfloat16)
    
# TODO:used for abs position embedding
class TokenEmbedding(nnx.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, rngs: nnx.Rngs):
        self.weight = nnx.Param(jax.random.normal(rngs.params(), (num_embeddings, embedding_dim)))
    def __call__(self, token_ids):
        return jnp.take(self.weight, token_ids, axis=0)

def _resize_bilinear_nchw(x, out_h, out_w, align_corners: bool):
    # x: [N, C, H, W]
    in_h, in_w = x.shape[2], x.shape[3]

    def _coords(in_size, out_size):
        if out_size == 1:
            return jnp.zeros((1,), dtype=jnp.float32)
        if align_corners:
            scale = (in_size - 1) / (out_size - 1)
            return jnp.arange(out_size, dtype=jnp.float32) * scale
        else:
            scale = in_size / out_size
            return (jnp.arange(out_size, dtype=jnp.float32) + 0.5) * scale - 0.5

    ys = _coords(in_h, out_h)
    xs = _coords(in_w, out_w)

    y0 = jnp.floor(ys).astype(jnp.int32)
    x0 = jnp.floor(xs).astype(jnp.int32)
    y1 = jnp.minimum(y0 + 1, in_h - 1)
    x1 = jnp.minimum(x0 + 1, in_w - 1)

    wy = (ys - y0).reshape(1, 1, out_h, 1)
    wx = (xs - x0).reshape(1, 1, 1, out_w)

    # 先按 H 做插值，再按 W 做插值
    v0 = jnp.take(x, y0, axis=2)  # [N,C,out_h,W]
    v1 = jnp.take(x, y1, axis=2)
    v = v0 * (1.0 - wy) + v1 * wy  # [N,C,out_h,W]

    v0 = jnp.take(v, x0, axis=3)  # [N,C,out_h,out_w]
    v1 = jnp.take(v, x1, axis=3)
    out = v0 * (1.0 - wx) + v1 * wx
    return out


class Qwen3_VisionModel(nnx.Module):
    def __init__(self,
                 config: Qwen3VLConfig,
                 norm_eps: float = 1e-6,
                 dtype: jnp.dtype = jnp.bfloat16,
                 rngs: nnx.Rngs = None,
                 mesh: Mesh = None):
        self.dtype = dtype
        self.hidden_size = config.vision_config.hidden_size
        self.num_heads = config.vision_config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_position_embeddings = config.vision_config.num_position_embeddings
        self.num_grid_per_side = int(self.num_position_embeddings ** 0.5)
        self.num_grid = self.num_grid_per_side ** 2
        self.align_corners = False
        self.patch_size = config.vision_config.patch_size
        self.spatial_merge_size = config.vision_config.spatial_merge_size
        self.spatial_merge_unit = self.spatial_merge_size ** 2
        self.temporal_patch_size = config.vision_config.temporal_patch_size
        # DeepStack
        self.deepstack_visual_indexes = config.vision_config.deepstack_visual_indexes
        self.out_hidden_size = self.hidden_size * self.spatial_merge_unit
        
        self.patch_embed = Qwen3_VLVisionPatchEmbed(
            patch_size=config.vision_config.patch_size,
            temporal_patch_size=config.vision_config.temporal_patch_size,
            in_channels=config.vision_config.in_channels,
            hidden_size=config.vision_config.hidden_size,
            dtype=dtype,
            rngs=rngs)
        
        # TODO (qihang) Simple version:2026.1.20 pos_embed (be used to abs pos embed)-------------------------------
        self.pos_embed = nnx.Embed(
                            num_embeddings=self.num_position_embeddings,
                            features=self.hidden_size,
                            rngs=rngs,
                        )
        # NOTE(qihang) 实际上这只是freqs，不是完整的rope_emb
        self.rotary_pos_emb = Qwen3_VisionRotaryEmbedding(self.head_dim // 2)

        self.blocks = nnx.data([
            Qwen3_VisionBlock(
                config=config,
                norm_eps=norm_eps,
                dtype=dtype,
                rngs=rngs,
                mesh=mesh,
            ) for _ in range(config.vision_config.depth)
        ])
        self.merger = Qwen3_VisionPatchMerger(
            config = config.vision_config,
            use_postshuffle_norm=False,
            dtype=dtype,
            rngs=rngs,
            mesh=mesh,
        )
        # TODO(qihang) 所有的prefix都还没有加上去
        self.deepstack_merger_list = nnx.data([
                Qwen3_VisionPatchMerger(
                    config = config.vision_config,
                    use_postshuffle_norm = True,
                    dtype = dtype,
                    rngs = rngs,
                    mesh = mesh,
                )
                for layer_idx in range(len(self.deepstack_visual_indexes))
            ])
     

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

    def fast_pos_embed_interpolate(self, grid_thw):
        patch_pos_embeds_permute = []
        m_size = self.spatial_merge_size

        embeds = jnp.arange(self.num_grid, dtype=jnp.int32)
        embeds = (
            self.pos_embed(embeds)          # [num_grid, dim]
            .transpose(1, 0)                # [dim, num_grid]
            .reshape(1, -1, self.num_grid_per_side, self.num_grid_per_side)
        )  # [1, dim, Gh, Gw]

        for t, h, w in grid_thw:
            # TODO (qihang) JAX的插值使用自己的坐标映射规则，需要检查和Torch是否一致。
            pos_embed = jimage.resize(
                embeds, (1, embeds.shape[1], h, w), method="bilinear"
            )
            pos_embed = pos_embed.reshape(
                -1,
                h // m_size,
                m_size,
                w // m_size,
                m_size,
            )
            pos_embed = pos_embed.transpose(1, 3, 2, 4, 0)
            pos_embed = pos_embed.reshape(-1, pos_embed.shape[-1])
            pos_embed = jnp.repeat(pos_embed, t, axis=0)
            patch_pos_embeds_permute.append(pos_embed)

        return jnp.concatenate(patch_pos_embeds_permute, axis=0)

    def rot_pos_ids(self, h: int, w: int, spatial_merge_size: int) -> jnp.ndarray:
        """
        生成旋转位置编码 ID
        
        Args:
            h: 高度 (int)
            w: 宽度 (int)
            spatial_merge_size: 空间合并大小 (int)
        """
        # 1. 生成网格坐标
        # jnp.indices((h, w)) 返回 shape (2, h, w)
        # grid[0] 是行索引 (hpos), grid[1] 是列索引 (wpos)
        grid = jnp.indices((h, w), dtype=jnp.int32)
        hpos_ids, wpos_ids = grid[0], grid[1]

        # 2. 计算 reshape 的维度
        # 注意：在 JAX 的 JIT 编译中，reshape 的维度必须是静态已知的整数
        h_div = h // spatial_merge_size
        w_div = w // spatial_merge_size
        
        # 定义变换逻辑
        def process_ids(ids):
            # Reshape: [h_div, s, w_div, s]
            ids = ids.reshape(h_div, spatial_merge_size, w_div, spatial_merge_size)
            # Transpose: [h_div, w_div, s, s] (交换轴 1 和 2)
            ids = ids.transpose(0, 2, 1, 3)
            # Flatten
            return ids.flatten()

        # 3. 应用变换
        hpos_ids = process_ids(hpos_ids)
        wpos_ids = process_ids(wpos_ids)

        # 4. 堆叠: [flattened_len, 2]
        return jnp.stack([hpos_ids, wpos_ids], axis=-1)

    def rot_pos_emb(self, grid_thw: list[list[int]]):
        pos_ids = []
        # TODO (qihang) 检查多图像batch，base是否需要添加offset
        for t, h, w in grid_thw:
            base = self.rot_pos_ids(h, w, self.spatial_merge_size)  # [hw', 1] or [hw', ?]
            pos_ids.append(base if t == 1 else jnp.tile(base, (t, 1)))

        pos_ids = jnp.concatenate(pos_ids, axis=0)
        max_grid_size = max(max(h, w) for _, h, w in grid_thw)

        freqfreq_tables = self.rotary_pos_emb(max_grid_size)
        embeddings = freqfreq_tables[pos_ids]
        embeddings = embeddings.reshape(embeddings.shape[0], -1)
        return embeddings

    def compute_cu_seqlens_from_grid(self, grid_thw: tuple[tuple[int, int,
                                                           int]]) -> jax.Array:
        seqlens_list = []
        for t, h, w in grid_thw:
            spatial_len = h * w
            
            if t > 1:
                chunk = jnp.full((t,), spatial_len, dtype=jnp.int32)
            else:
                chunk = jnp.array([spatial_len], dtype=jnp.int32)
                
            seqlens_list.append(chunk)

        all_lengths = jnp.concatenate(seqlens_list, axis=0)

        cu_seqlens = jnp.cumsum(all_lengths, dtype=jnp.int32)

        cu_seqlens = jnp.pad(cu_seqlens, (1, 0), constant_values=0)

        return cu_seqlens
    
    
    def __call__(self, x: jax.Array, grid_thw: tuple[tuple[int, int,
                                                           int]]) -> jax.Array:
        # 1. patchembed
        hidden_states = self.patch_embed(x)
        # 2. abs pos embed
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states += pos_embeds
        # 3. rope
        # NOTE (qihang) 重构 rotary pos emb | cu_seqlens，参考SGLang、Transformers
        seq_len = x.shape[0]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = self.rot_pos_emb(jnp.array(grid_thw).tolist())
        cu_seqlens = self.compute_cu_seqlens_from_grid(grid_thw)# check if cu_seqlens is on cpu and int32
        # 4. deepstack
        num_deepstack_captured = 0
        deepstack_feature_lists = []
        # 5. blk forward
        hidden_states = jnp.expand_dims(hidden_states, axis=1)
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(hidden_states,
                                rotary_pos_emb=rotary_pos_emb,
                                cu_seqlens=cu_seqlens,
                                use_fullattn=True)
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[num_deepstack_captured](hidden_states)
                deepstack_feature_lists.append(deepstack_feature)
                num_deepstack_captured += 1
        hidden_states = self.merger(hidden_states)
        results = jnp.concatenate([hidden_states] + deepstack_feature_lists, axis=0)
        return results

def test_qwen3_vision_model():
    import numpy as np
    config = Qwen3VLConfig()
    devices = jax.devices() 
    mesh = Mesh(np.array(devices).reshape((1, 1)), ("data", "tensor"), axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit))
    jax.set_mesh(mesh)
    rngs = nnx.Rngs(0)
    
    model = Qwen3_VisionModel(
        config=config,
        dtype=jnp.bfloat16,
        rngs=rngs,
        mesh=mesh
    )
    # 模拟某一视频输入的shape
    dummy_x = jnp.ones((14080, 1536), dtype=jnp.bfloat16)
    
    grid_thw = ((4, 80, 44),) 
    
    print("Forwarding...")
    output = model(dummy_x, grid_thw)
    print(f"✅ Forward is done! 输出形状: {output.shape}")
#---LLMDecoder

#---Model
class Qwen3_VLForConditionalGeneration(nnx.Module):

    def __init__(
        self,
        config: Qwen3VLConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        mesh: Mesh = None,
    ) -> None:

        self.config = config
        self.rng = nnx.Rngs(params=0)
        self.dtype = dtype
        self.mesh = mesh

        self.visual = Qwen3_VisionModel(
            config=config,
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            dtype=dtype,
            rngs=self.rng,
            mesh=mesh,
        )
        # TODO (qihang) LLM Model
        # self.model = Qwen2Model(
        #     config=config,
        #     dtype=dtype,
        #     mesh=mesh,
        # )

        # self.lm_head = ParallelLMHead(
        #     config.vocab_size,
        #     config.hidden_size,
        #     dtype=dtype,
        #     param_dtype=dtype,
        #     kernel_axes=("tensor", None),
        # )

        # self.is_mrope_enabled = "mrope_section" in config.rope_scaling

        # self.logits_processor = LogitsProcessor(config.vocab_size, mesh=mesh)

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_image_feature(self, items: List[MultimodalDataItem]) -> jax.Array:
        # in qwen-vl, last dim is the same
        pixel_values = jnp.concatenate([item.feature for item in items], axis=0).astype(
            self.visual.dtype
        )
        image_grid_thw = jnp.concatenate([item.image_grid_thw for item in items], axis=0)
        assert pixel_values.ndim == 2, pixel_values.ndim
        assert image_grid_thw.ndim == 2, image_grid_thw.ndim
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        return image_embeds

    def get_video_feature(self, items: List[MultimodalDataItem]) -> jax.Array:
        # in qwen-vl, last dim is the same
        pixel_values = jnp.concatenate([item.feature for item in items], axis=0).astype(
            self.visual.dtype
        )
        video_grid_thw = jnp.concatenate([item.video_grid_thw for item in items], axis=0)
        assert pixel_values.ndim == 2, pixel_values.ndim
        assert video_grid_thw.ndim == 2, video_grid_thw.ndim
        video_embeds = self.visual(pixel_values, grid_thw=video_grid_thw)
        return video_embeds

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        logits_metadata: LogitsMetadata,
    ):
        """Run forward pass for Qwen2_5-VL.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            positions: Flattened (concatenated) position ids corresponding to a
                batch.
                **NOTE**: If mrope is enabled (default setting for Qwen2-VL
                opensource models), the shape will be `(3, seq_len)`,
                otherwise it will be `(seq_len,).
                (Use input_metadata.mrope_positions to replace it)
        """
        if self.is_mrope_enabled:
            positions = forward_batch.mrope_positions

        if not (
            forward_batch.forward_mode.is_decode()
            or not forward_batch.contains_image_inputs()
        ):
            if self.is_mrope_enabled:
                assert positions.ndim == 2 and positions.shape[0] == 3, (
                    "multimodal section rotary embedding requires "
                    f"(3, seq_len) positions, but got {positions.shape}"
                )

        hidden_states, layers_kv_fused, layers_callback_flag = general_mm_embed_routine(
            forward_batch=forward_batch,
            language_model=self.model,
            token_to_kv_pool=token_to_kv_pool,
            multimodal_model=self,
            positions=positions
        )
        
        return self.logits_processor(hidden_states, self.lm_head, logits_metadata), layers_kv_fused, layers_callback_flag

    def load_weights(self, model_config):
        """Load weights for Qwen3-VL model.
        
        Args:
            model_config: Model configuration containing model path and settings
        """
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        
        weight_mappings = self._create_qwen3_vl_weight_mappings()
        
        loader.load_weights_from_safetensors(weight_mappings)
        
        if getattr(self.config, "tie_word_embeddings", False):
            self.lm_head.embedding = self.model.embed_tokens.embedding
            logger.info("Tied word embeddings: lm_head's weights are now tied to embed_tokens'.")
        
        logger.info("Qwen3-VL weights loaded successfully!")

    def _create_qwen3_vl_weight_mappings(self) -> dict:
        """Create weight mappings for Qwen3-VL model.
        
        Returns:
            Dictionary mapping HuggingFace weight names to model parameter paths
        """        
        mappings = {}
        
        # Vision transformer weights
        mappings.update(self._create_vision_transformer_mappings())
        
        # # Language model embeddings
        # mappings["model.embed_tokens.weight"] = WeightMapping(
        #     target_path="model.embed_tokens.embedding",
        #     sharding=("tensor", None),
        #     transpose=False,
        # )
        
        # # Language model norm
        # mappings["model.norm.weight"] = WeightMapping(
        #     target_path="model.norm.scale",
        #     sharding=(None,),
        #     transpose=False,
        # )
        
        # # LM head
        # if not getattr(self.config, "tie_word_embeddings", False):
        #     mappings["lm_head.weight"] = WeightMapping(
        #         target_path="lm_head.embedding",
        #         sharding=("tensor", None),
        #         transpose=False,
        #     )
        
        # # Language model layers
        # num_layers = self.config.num_hidden_layers
        # for layer_idx in range(num_layers):
        #     layer_mappings = self._create_layer_mappings(layer_idx)
        #     mappings.update(layer_mappings)
        
        return mappings
    
    def _create_vision_transformer_mappings(self) -> dict:
        """Create weight mappings for the vision transformer.
        
        Returns:
            Dictionary mapping vision transformer weight names to model paths
        """        
        mappings = {}
        
        # Vision embeddings
        mappings["visual.patch_embed.proj.weight"] = WeightMapping(
            target_path="visual.patch_embed.proj.kernel",
            sharding=(None, None, None, None, "tensor"),
            transpose=False,
            transpose_dims=(2, 3, 4, 1, 0),
        )
        
        # Note: In the model definition, use_bias=False is set for the proj Conv layer
        # So we don't need to map the bias parameter
        
        # Add merger mappings
        mappings["visual.merger.ln_q.weight"] = WeightMapping(
            target_path="visual.merger.ln_q.scale",
            sharding=(None,),
            transpose=False,
        )
        mappings["visual.merger.mlp.0.weight"] = WeightMapping(
            target_path="visual.merger.mlp_fc1.weight",
            sharding=(None, "tensor"),
            transpose=True,
        )
        mappings["visual.merger.mlp.0.bias"] = WeightMapping(
            target_path="visual.merger.mlp_fc1.bias",
            sharding=("tensor",),
            transpose=False,
        )
        mappings["visual.merger.mlp.2.weight"] = WeightMapping(
            target_path="visual.merger.mlp_fc2.weight",
            sharding=("tensor", None),
            transpose=True,
        )
        mappings["visual.merger.mlp.2.bias"] = WeightMapping(
            target_path="visual.merger.mlp_fc2.bias",
            sharding=(None,),
            transpose=False,
        )
        
        # Vision transformer layers
        if hasattr(self.config, "vision_config"):
            num_vision_layers = getattr(self.config.vision_config, "depth", 0)
            for layer_idx in range(num_vision_layers):
                vision_layer_mappings = self._create_vision_layer_mappings(layer_idx)
                mappings.update(vision_layer_mappings)
        
        return mappings
    
    def _create_vision_layer_mappings(self, layer_idx: int) -> dict:
        """Create weight mappings for a single vision transformer layer.
        
        Args:
            layer_idx: Index of the vision layer
            
        Returns:
            Dictionary mapping vision layer weight names to model paths
        """
        from sgl_jax.srt.utils.weight_utils import WeightMapping
        
        prefix = f"visual.blocks.{layer_idx}"
        target_prefix = f"visual.blocks.{layer_idx}"
        
        mappings = {
            # Attention norm
            f"{prefix}.norm1.weight": WeightMapping(
                target_path=f"{target_prefix}.norm1.scale",
                sharding=(None,),
                transpose=False,
            ),
            # Attention QKV projection
            f"{prefix}.attn.qkv.weight": WeightMapping(
                target_path=f"{target_prefix}.attn.qkv_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
            f"{prefix}.attn.qkv.bias": WeightMapping(
                target_path=f"{target_prefix}.attn.qkv_proj.bias",
                sharding=("tensor",),
                transpose=False,
            ),
            # Attention output projection
            f"{prefix}.attn.proj.weight": WeightMapping(
                target_path=f"{target_prefix}.attn.o_proj.weight",
                sharding=("tensor", None),
                transpose=True,
            ),
            f"{prefix}.attn.proj.bias": WeightMapping(
                target_path=f"{target_prefix}.attn.o_proj.bias",
                sharding=(None,),
                transpose=False,
            ),
            # MLP norm
            f"{prefix}.norm2.weight": WeightMapping(
                target_path=f"{target_prefix}.norm2.scale",
                sharding=(None,),
                transpose=False,
            ),
            # MLP gate projection
            f"{prefix}.mlp.gate_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.mlp.gate_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
            f"{prefix}.mlp.gate_proj.bias": WeightMapping(
                target_path=f"{target_prefix}.mlp.gate_proj.bias",
                sharding=("tensor",),
                transpose=False,
            ),
            # MLP up projection
            f"{prefix}.mlp.up_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.mlp.up_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
            f"{prefix}.mlp.up_proj.bias": WeightMapping(
                target_path=f"{target_prefix}.mlp.up_proj.bias",
                sharding=("tensor",),
                transpose=False,
            ),
            # MLP down projection
            f"{prefix}.mlp.down_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.mlp.down_proj.weight",
                sharding=("tensor", None),
                transpose=True,
            ),
            f"{prefix}.mlp.down_proj.bias": WeightMapping(
                target_path=f"{target_prefix}.mlp.down_proj.bias",
                sharding=(None,),
                transpose=False,
            ),
        }
        
        return mappings
    
    def _create_layer_mappings(self, layer_idx: int) -> dict:
        """Create weight mappings for a single language model layer.
        
        Args:
            layer_idx: Index of the layer
            
        Returns:
            Dictionary mapping layer weight names to model paths
        """        
        prefix = f"model.layers.{layer_idx}"
        target_prefix = f"model.layers.{layer_idx}"
        
        mappings = {
            f"{prefix}.input_layernorm.weight": WeightMapping(
                target_path=f"{target_prefix}.input_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.post_attention_layernorm.weight": WeightMapping(
                target_path=f"{target_prefix}.post_attention_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.self_attn.q_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.q_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
                head_dim_padding=True,
                kv_head_padding=False,
            ),
            f"{prefix}.self_attn.k_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.k_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
                head_dim_padding=True,
                kv_head_padding=True,
            ),
            f"{prefix}.self_attn.v_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.v_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
                head_dim_padding=True,
                kv_head_padding=True,
            ),
            f"{prefix}.self_attn.o_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.o_proj.weight",
                sharding=("tensor", None),
                transpose=True,
                head_dim_padding=True,
                kv_head_padding=False,
            ),
            f"{prefix}.mlp.gate_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.mlp.gate_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
            f"{prefix}.mlp.up_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.mlp.up_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
            f"{prefix}.mlp.down_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.mlp.down_proj.weight",
                sharding=("tensor", None),
                transpose=True,
            ),
        }
        
        # Add bias mappings if attention_bias is enabled
        if getattr(self.config, "attention_bias", True):
            bias_mappings = {
                f"{prefix}.self_attn.q_proj.bias": WeightMapping(
                    target_path=f"{target_prefix}.self_attn.q_proj.bias",
                    sharding=("tensor",),
                    transpose=False,
                    head_dim_padding=True,
                    kv_head_padding=False,
                ),
                f"{prefix}.self_attn.k_proj.bias": WeightMapping(
                    target_path=f"{target_prefix}.self_attn.k_proj.bias",
                    sharding=("tensor",),
                    transpose=False,
                    head_dim_padding=True,
                    kv_head_padding=True,
                ),
                f"{prefix}.self_attn.v_proj.bias": WeightMapping(
                    target_path=f"{target_prefix}.self_attn.v_proj.bias",
                    sharding=("tensor",),
                    transpose=False,
                    head_dim_padding=True,
                    kv_head_padding=True,
                ),
            }
            mappings.update(bias_mappings)
        
        return mappings


EntryClass = [Qwen3_VLForConditionalGeneration]

#--- Encoder Test code


if __name__ == "__main__":
    # Test Vision Model
    # test_qwen3_vision_model()
    import numpy as np
    config = Qwen3VLConfig()
    devices = jax.devices() 
    mesh = Mesh(np.array(devices).reshape((1, 1)), ("data", "tensor"), axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit))
    jax.set_mesh(mesh)
    model = Qwen3_VLForConditionalGeneration(Qwen3VLConfig(), dtype=jnp.bfloat16, mesh=mesh)
    print("Model initialized successfully.")
    '''
    Model_path:
    (qihang)local_dir: /home/wqh/projects/Qwen3-VL/model_dir
    TPU remote_dir: /models/Qwen3-VL/Qwen3-VL-8B-Thinking      ---if use Qwen3-VL-8B-Thinking   
    '''
    model.load_weights(model_config=ModelConfig(model_path="/models/Qwen3-VL/Qwen3-VL-8B-Thinking"))
    print("Load Weight successfully.")

