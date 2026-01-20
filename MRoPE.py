from python.sgl_jax.srt.layers.embeddings import RotaryEmbedding, _apply_rotary_emb
import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax.nnx.nn import dtypes
from flax.typing import PromoteDtypeFn
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from typing import Optional

class RotaryEmbedding:
    """Rotary Position Embedding (safe to initialize inside JIT if needed)."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: jnp.dtype,
    ):
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype

        inv_freq_np = 1.0 / (base ** (np.arange(0, rotary_dim, 2, dtype=np.float32) / rotary_dim))
        self._inv_freq_np = inv_freq_np  # shape: (rotary_dim // 2,)

    def __call__(
        self,
        positions: jax.Array,
        query: jax.Array,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        positions = positions.flatten()  # [num_tokens]
        num_tokens = positions.shape[0]

        cache = self._compute_cos_sin_cache()
        cos_sin = jnp.take(cache, positions, axis=0)  
        cos, sin = jnp.split(cos_sin, 2, axis=-1)  # Each: [num_tokens, rotary_dim // 2]

        query_shape = query.shape
        query = query.reshape(num_tokens, -1, self.head_size)
        query_rot = query[..., : self.rotary_dim]
        query_pass = query[..., self.rotary_dim :]
        query_rot = _apply_rotary_emb(query_rot, cos, sin, self.is_neox_style)
        query = jnp.concatenate((query_rot, query_pass), axis=-1).reshape(query_shape)

        key_shape = key.shape
        key = key.reshape(num_tokens, -1, self.head_size)
        key_rot = key[..., : self.rotary_dim]
        key_pass = key[..., self.rotary_dim :]
        key_rot = _apply_rotary_emb(key_rot, cos, sin, self.is_neox_style)
        key = jnp.concatenate((key_rot, key_pass), axis=-1).reshape(key_shape)

        return query, key

    def _compute_inv_freq(self, base: int | float) -> jax.Array:
        """Compute the inverse frequency."""
        inv_freq = 1.0 / (
            base ** (jnp.arange(0, self.rotary_dim, 2, dtype=jnp.float32) / self.rotary_dim)
        )
        return inv_freq

    def _compute_cos_sin_cache(self) -> jax.Array:
        """Compute the cos and sin cache."""
        inv_freq = self._compute_inv_freq(self.base)
        t = jnp.arange(self.max_position_embeddings, dtype=jnp.float32)
        freqs = jnp.outer(t, inv_freq)
        sin, cos = jnp.sin(freqs), jnp.cos(freqs)
        cache = jnp.concatenate((cos, sin), axis=-1)
        return cache



class MRotaryEmbedding(RotaryEmbedding):
    """Rotary Embedding with Multimodal Sections for JAX."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: jnp.dtype,
        mrope_section: list[int] | None = None,
        mrope_interleaved: bool = False,
    ) -> None:
        super().__init__(head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype)

        self.mrope_section = mrope_section
        self.mrope_interleaved = mrope_interleaved

        # Validation and Auto-correction Logic adapted from PyTorch implementation
        if self.mrope_section:
            expected_sum = rotary_dim // 2
            actual_sum = sum(self.mrope_section)
            if actual_sum != expected_sum:
                print(
                    f"MRoPE section sum mismatch: expected {expected_sum}, got {actual_sum}. "
                    f"Adjusting mrope_section to match rotary_dim // 2 = {expected_sum}"
                )
                # Auto-correct by scaling the mrope_section proportionally
                if actual_sum > 0:
                    scale_factor = expected_sum / actual_sum
                    self.mrope_section = [
                        max(1, int(section * scale_factor)) for section in self.mrope_section
                    ]
                    # Ensure the sum exactly matches by adjusting the last element
                    current_sum = sum(self.mrope_section)
                    if current_sum != expected_sum:
                        self.mrope_section[-1] += expected_sum - current_sum
                else:
                    # Fallback for zero sum
                    self.mrope_section = [expected_sum // len(self.mrope_section)] * len(
                        self.mrope_section
                    )
                    # Handle remainder
                    remainder = expected_sum % len(self.mrope_section)
                    for i in range(remainder):
                        self.mrope_section[i] += 1
                print(
                    f"Corrected mrope_section: {self.mrope_section} (sum={sum(self.mrope_section)})"
                )
            # Pre-calculate split indices for jnp.split
            # mrope_section is like [16, 24, 24], split indices should be [16, 40]
            self.split_indices = np.cumsum(self.mrope_section)[:-1].tolist()

    def __call__(
        self,
        positions: jax.Array,
        query: jax.Array,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """
        Args:
            positions: [num_tokens] (Text only) or
                       [3, num_tokens] (Multimodal T/H/W positions)
            query: [num_tokens, num_heads * head_size]
            key: [num_tokens, num_kv_heads * head_size]
        """
        # Handle Multimodal 3D Positions
        if positions.ndim == 2 and positions.shape[0] == 3:
            return self._forward_mrope(positions, query, key)

        # Fallback to standard RoPE for 1D positions
        return super().__call__(positions, query, key)

    def _forward_mrope(
        self,
        positions: jax.Array,
        query: jax.Array,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        # positions: [3, num_tokens]
        num_tokens = positions.shape[-1]

        # 1. Compute Cos/Sin for all 3 dimensions
        inv_freq = jnp.asarray(self._inv_freq_np, dtype=self.dtype)

        # freqs: [3, num_tokens, rotary_dim // 2]
        freqs = jnp.einsum("cn,d->cnd", positions.astype(jnp.float32), inv_freq)

        cos_all = jnp.cos(freqs).astype(self.dtype)
        sin_all = jnp.sin(freqs).astype(self.dtype)

        if self.mrope_interleaved:
            # --- Interleaved Mode ---
            # Direct manipulation on the [3, N, D] tensor
            cos = apply_interleaved_rope(cos_all, self.mrope_section)
            sin = apply_interleaved_rope(sin_all, self.mrope_section)
        else:
            # --- Chunked Mode (Existing Logic) ---
            # 2. Split and Select based on mrope_section
            cos_splits = jnp.split(cos_all, self.split_indices, axis=-1)
            sin_splits = jnp.split(sin_all, self.split_indices, axis=-1)

            # Select specific rows for specific sections
            # section 0 uses row 0 (Time), section 1 uses row 1 (Height), section 2 uses row 2 (Width)
            final_cos_list = []
            final_sin_list = []

            for i, split_tensor in enumerate(cos_splits):
                # split_tensor shape: [3, num_tokens, section_dim]
                # We take the i-th row: [num_tokens, section_dim]
                final_cos_list.append(split_tensor[i])

            for i, split_tensor in enumerate(sin_splits):
                final_sin_list.append(split_tensor[i])

            # Concatenate back: [num_tokens, rotary_dim // 2]
            cos = jnp.concatenate(final_cos_list, axis=-1)
            sin = jnp.concatenate(final_sin_list, axis=-1)

        # 3. Apply RoPE
        # Reshape query/key to [num_tokens, num_heads, head_size]
        query_real = query[:num_tokens]
        query_shape = query_real.shape
        query_real = query_real.reshape(num_tokens, -1, self.head_size)
        query_rot = query_real[..., : self.rotary_dim]
        query_pass = query_real[..., self.rotary_dim :]

        query_rot = _apply_rotary_emb(query_rot, cos, sin, self.is_neox_style)
        query_real = jnp.concatenate((query_rot, query_pass), axis=-1).reshape(query_shape)
        query = query.at[:num_tokens].set(query_real)

        key_real = key[:num_tokens]
        key_shape = key_real.shape
        key_real = key_real.reshape(num_tokens, -1, self.head_size)
        key_rot = key_real[..., : self.rotary_dim]
        key_pass = key_real[..., self.rotary_dim :]

        key_rot = _apply_rotary_emb(key_rot, cos, sin, self.is_neox_style)
        key_real = jnp.concatenate((key_rot, key_pass), axis=-1).reshape(key_shape)
        key = key.at[:num_tokens].set(key_real)

        return query, key

    @staticmethod
    def get_rope_index(
        spatial_merge_size: int,
        image_token_id: int,
        video_token_id: int,
        vision_start_token_id: int,
        model_type: str,
        tokens_per_second: int | None = None,
        input_ids: np.ndarray | None = None,
        image_grid_thw: np.ndarray | None = None,
        video_grid_thw: np.ndarray | None = None,
        second_per_grid_ts: np.ndarray | None = None,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.
        """
        # Handle video grid modification for Qwen3-VL
        if (
            model_type.startswith("qwen3_vl") or model_type.startswith("qwen3_vl_moe")
        ) and video_grid_thw is not None:
            video_grid_thw = np.repeat(video_grid_thw, video_grid_thw[:, 0], axis=0)
            video_grid_thw[:, 0] = 1

        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            # [3, batch, seq_len]
            position_ids = np.ones(
                (3, input_ids.shape[0], input_ids.shape[1]),
                dtype=input_ids.dtype,
            )
            image_index, video_index = 0, 0

            for i, input_ids_row in enumerate(total_input_ids):
                image_nums, video_nums = 0, 0
                # Find vision start tokens
                vision_start_indices = np.argwhere(input_ids_row == vision_start_token_id).squeeze(
                    1
                )

                # Determine if following tokens are image or video
                if vision_start_indices.size > 0:
                    # Safety check for index bounds
                    valid_indices = vision_start_indices + 1 < len(input_ids_row)
                    vision_tokens = input_ids_row[vision_start_indices[valid_indices] + 1]
                    image_nums = np.sum(vision_tokens == image_token_id)
                    video_nums = np.sum(vision_tokens == video_token_id)

                input_tokens = input_ids_row.tolist()
                llm_pos_ids_list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums

                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        try:
                            ed_image = input_tokens.index(image_token_id, st)
                        except ValueError:
                            ed_image = len(input_tokens) + 1
                    else:
                        ed_image = len(input_tokens) + 1

                    if video_token_id in input_tokens and remain_videos > 0:
                        try:
                            ed_video = input_tokens.index(video_token_id, st)
                        except ValueError:
                            ed_video = len(input_tokens) + 1
                    else:
                        ed_video = len(input_tokens) + 1

                    if ed_image < ed_video:
                        t, h, w = image_grid_thw[image_index]
                        second_per_grid_t = 0
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = video_grid_thw[video_index]
                        if second_per_grid_ts is not None:
                            second_per_grid_t = second_per_grid_ts[video_index]
                        else:
                            second_per_grid_t = 1.0
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video

                    llm_grid_t = t.item()
                    llm_grid_h = h.item() // spatial_merge_size
                    llm_grid_w = w.item() // spatial_merge_size

                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    # Text part
                    llm_pos_ids_list.append(
                        np.arange(text_len).reshape(1, -1).repeat(3, axis=0) + st_idx
                    )

                    if model_type == "qwen2_5_vl":
                        range_tensor = np.arange(llm_grid_t).reshape(-1, 1)
                        expanded_range = np.tile(range_tensor, (1, llm_grid_h * llm_grid_w))
                        time_tensor = expanded_range * second_per_grid_t * tokens_per_second
                        t_index = time_tensor.astype(np.int64).flatten()

                    elif model_type in ("qwen2_vl", "qwen3_vl", "qwen3_vl_moe"):
                        t_index = np.tile(
                            np.arange(llm_grid_t).reshape(-1, 1), (1, llm_grid_h * llm_grid_w)
                        ).flatten()
                    else:
                        raise RuntimeError(f"Unimplemented model type: {model_type}")

                    h_index = np.tile(
                        np.arange(llm_grid_h).reshape(1, -1, 1), (llm_grid_t, 1, llm_grid_w)
                    ).flatten()

                    w_index = np.tile(
                        np.arange(llm_grid_w).reshape(1, 1, -1), (llm_grid_t, llm_grid_h, 1)
                    ).flatten()

                    llm_pos_ids_list.append(
                        np.stack([t_index, h_index, w_index]) + text_len + st_idx
                    )
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                # Process remaining text at the end
                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(
                        np.arange(text_len).reshape(1, -1).repeat(3, axis=0) + st_idx
                    )

                llm_positions = np.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
                position_ids[..., i, :] = llm_positions
                mrope_position_deltas.append(llm_positions.max() + 1 - len(input_ids_row))

            mrope_position_deltas = np.array(mrope_position_deltas).reshape(-1, 1)
            return position_ids, mrope_position_deltas
        else:
            # Standard 1D RoPE case
            s = input_ids.shape[1]
            position_ids = np.arange(s)
            position_ids = np.tile(position_ids.reshape(1, 1, -1), (3, input_ids.shape[0], 1))

            max_position_ids = position_ids.max(axis=0).max(axis=-1, keepdims=True)
            mrope_position_deltas = max_position_ids + 1 - s
            return position_ids, mrope_position_deltas


    @staticmethod
    def _get_feat_extract_output_lengths(input_lengths):
        input_lengths_leave = input_lengths % 100
        feat_lengths = (input_lengths_leave - 1) // 2 + 1
        output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
        return output_lengths

    @staticmethod
    def _get_llm_pos_ids_for_vision_numpy(
        st_idx, vision_idx, spatial_merge_size, t_index, grid_hs, grid_ws
    ):
        """NumPy adaptation of _get_llm_pos_ids_for_vision"""
        grid_h = grid_hs[vision_idx] // spatial_merge_size
        grid_w = grid_ws[vision_idx] // spatial_merge_size

        h_index = np.tile(np.arange(grid_h).reshape(1, -1, 1), (len(t_index), 1, grid_w)).flatten()

        w_index = np.tile(np.arange(grid_w).reshape(1, 1, -1), (len(t_index), grid_h, 1)).flatten()

        t_index = np.tile(t_index.reshape(-1, 1), (1, grid_h * grid_w)).flatten()

        llm_pos_ids = np.stack([t_index, h_index, w_index], axis=0) + st_idx
        return llm_pos_ids