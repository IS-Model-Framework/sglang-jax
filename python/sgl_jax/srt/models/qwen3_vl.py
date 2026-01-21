import math
from functools import partial
from typing import List, Callable, NamedTuple, Optional, Any
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig,Qwen3VLVisionConfig,Qwen3VLTextConfig
from transformers import modeling_flax_utils
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jax.experimental import shard_map
from flax import nnx

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
        self.attn = Qwen2_5_VisionAttention(hidden_size=config.vision_config.hidden_size,
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

    def __call__(self, x: jax.Array) -> jax.Array:
        if self.use_postshuffle_norm:
            x = self.ln_q(x.reshape(-1, self.hidden_size))
        else:
            x = self.ln_q(x).reshape(-1, self.hidden_size)
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
        self.pos_embed = TokenEmbedding(
            num_embeddings=self.num_position_embeddings,
            embedding_dim=self.hidden_size,
            rngs=rngs
        )
        # NOTE(qihang) 实际上这只是freqs，不是完整的rope_emb
        self.rotary_pos_emb = Qwen3_VisionRotaryEmbedding(self.head_dim // 2)

        self.blocks = [
            Qwen3_VisionBlock(
                config=config,
                norm_eps=norm_eps,
                dtype=dtype,
                rngs=rngs,
                mesh=mesh,
            ) for _ in range(config.vision_config.depth)
        ]
        self.merger = Qwen3_VisionPatchMerger(
            config = config.vision_config,
            use_postshuffle_norm=False,
            dtype=dtype,
            rngs=rngs,
            mesh=mesh,
        )
        # TODO(qihang) 所有的prefix都还没有加上去
        self.deepstack_merger_list = [
                Qwen3_VisionPatchMerger(
                    config = config.vision_config,
                    use_postshuffle_norm = True,
                    dtype = dtype,
                    rngs = rngs,
                    mesh = mesh,
                )
                for layer_idx in range(len(self.deepstack_visual_indexes))
            ]
     

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
            pos_embed = _resize_bilinear_nchw(
                embeds, h, w, align_corners=self.align_corners
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

        # 1. patchembed
        hidden_states = self.patch_embed(x)
        # 2. abs pos embed
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states += pos_embeds
        # 3. rope
        # TODO (qihang) rotary_pos_emb, 

        seq_len = x.shape[0]
        hidden_states = hidden_states.reshape(seq_len, -1)
        
        # 4. deepstack
        # 5. blk forward
        # num of patches
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
            hidden_states = blk(hidden_states,
                                rotary_pos_emb=rotary_pos_emb,
                                cu_seqlens=cu_seqlens,
                                use_fullattn=True)
          

        # adapter
        hidden_states = rotary_pos_embmerger(hidden_states)
        reverse_indices = jnp.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]
        return hidden_states
#---LLMDecoder

#---Model
