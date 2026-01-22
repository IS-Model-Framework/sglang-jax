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
                        out_sharding = P("data", "tensor")
                    )
                ) 
        else:
            x = self.ln_q(x)
            x = jax.lax.reshape(
                        x, 
                        (x.shape[0] * x.shape[1] * x.shape[2] // self.hidden_size, self.hidden_size),
                        out_sharding = P("data", "tensor")
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
#---LLMDecoder

#---Model
def test_qwen3_vision_model():
    from jax.experimental import mesh_utils
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
    # 强制将数据按照该策略放置到设备上
    dummy_x = jnp.ones((14080, 1536), dtype=jnp.bfloat16)
    
    grid_thw = ((4, 80, 44),) 
    
    print("正在运行 Forward...")
    output = model(dummy_x, grid_thw)
    print(f"✅ Forward 运行成功! 输出形状: {output.shape}")

if __name__ == "__main__":
    test_qwen3_vision_model()
