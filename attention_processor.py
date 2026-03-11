import os
import sys
import io

# 🌟 禁用 Triton autotune 输出（必须在 import torch 之前）
os.environ["TRITON_PRINT_AUTOTUNING"] = "0"
os.environ["TORCH_LOGS"] = "-all"

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 🌟 禁用 torch.compile 的 autotune 日志
try:
    import torch._inductor.config as inductor_config
    inductor_config.verbose_progress = False
    inductor_config.benchmark_kernel = False
except:
    pass

from torch.nn.attention.flex_attention import create_block_mask, flex_attention

# 🌟 使用 suppress_triton_warnings 模式编译
import functools
@functools.lru_cache(maxsize=None)
def _get_compiled_create_block_mask():
    return torch.compile(create_block_mask)

create_block_mask = _get_compiled_create_block_mask()

# 🌟 全局标志：是否抑制 autotune 输出
_SUPPRESS_AUTOTUNE = True
_autotune_suppressed_ids = set()  # 已经 warmup 过的 compiled 函数

class _AutotuneSuppressor:
    """在第一次调用 compiled 函数时抑制 autotune 输出"""
    def __init__(self):
        self._old_stdout = None
        self._old_stderr = None
    
    def __enter__(self):
        if _SUPPRESS_AUTOTUNE:
            self._old_stdout = sys.stdout
            self._old_stderr = sys.stderr
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
        return self
    
    def __exit__(self, *args):
        if _SUPPRESS_AUTOTUNE and self._old_stdout is not None:
            sys.stdout = self._old_stdout
            sys.stderr = self._old_stderr
from diffusers.models.attention_processor import Attention
from typing import Optional
from functools import partial, lru_cache
from diffusers.models.embeddings import apply_rotary_emb


attn_outputs_teacher = []
attn_outputs = []
BLOCK_MASK = None
HEIGHT = None
WIDTH = None


class FluxAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self, distill=False):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.distill = distill

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        proportional_attention=False
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        train_seq_len = 64 ** 2 + 512
        if proportional_attention:
            attention_scale = math.sqrt(math.log(key.size(2), train_seq_len) / head_dim)
        else:
            attention_scale = math.sqrt(1 / head_dim)

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False, scale=attention_scale)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            if self.distill:
                attn_outputs_teacher.append(hidden_states)
            return hidden_states


@lru_cache
def init_local_downsample_mask_flex(height, width, text_length, window_size, down_factor, device):
    # 🌟 强制使用 CUDA
    if isinstance(device, str):
        device = "cuda" if "cuda" in device else device
    else:
        device = "cuda" if device.type == "cuda" else str(device)
    
    def local_dwonsample_mask(b, h, q_idx, kv_idx):
        q_y = (q_idx - text_length) // width
        q_x = (q_idx - text_length) % width
        kv_y = (kv_idx - text_length) // width
        kv_x = (kv_idx - text_length) % width
        return torch.logical_or(
            torch.logical_and(
                q_idx < text_length, 
                kv_idx < text_length + height * width
            ),
            torch.logical_and(
                q_idx >= text_length, 
                torch.logical_or(
                    torch.logical_or(kv_idx < text_length, kv_idx >= text_length + height * width),
                    (q_y - kv_y) ** 2 + (q_x - kv_x) ** 2 < window_size ** 2)
            )
        )
    
    global BLOCK_MASK, HEIGHT, WIDTH
    BLOCK_MASK = create_block_mask(local_dwonsample_mask, B=None, H=None, device=device,
                                   Q_LEN=text_length + height * width, 
                                   KV_LEN=text_length + height * width + (height // down_factor) * (width // down_factor), _compile=True)
    HEIGHT = height
    WIDTH = width


class LocalDownsampleFlexAttnProcessor(nn.Module):
    
    def __init__(self, down_factor=4, distill=False):
        super().__init__()
        # 🌟 修复：不在 __init__ 绑定 BLOCK_MASK，而是在 __call__ 时动态获取
        self.down_factor = down_factor
        self.spatial_weight = nn.Parameter(torch.ones(1, 1, 1, down_factor, 1, down_factor, 1) / (down_factor * down_factor))
        self.distill = distill
        self._compiled_flex_attn = None
        self._compiled_mask_id = None  # 用于检测 BLOCK_MASK 是否变化
        self._first_run = True  # 🌟 标记第一次运行
        
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        proportional_attention=False
    ) -> torch.FloatTensor:
        # 🌟 动态获取 BLOCK_MASK 并按需 compile
        global BLOCK_MASK, HEIGHT, WIDTH
        current_mask_id = id(BLOCK_MASK)
        need_recompile = self._compiled_flex_attn is None or self._compiled_mask_id != current_mask_id
        
        if need_recompile:
            self._compiled_flex_attn = torch.compile(
                partial(flex_attention, block_mask=BLOCK_MASK), 
                dynamic=False
            )
            self._compiled_mask_id = current_mask_id
            self._first_run = True  # 新编译后需要重新 warmup
        
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        train_seq_len = 64 ** 2 + 512
        if proportional_attention:
            attention_scale = math.sqrt(math.log(10 * key.size(2), train_seq_len) / head_dim)
        else:
            attention_scale = math.sqrt(1 / head_dim)
        
        key_downsample = (key[:, :, 512:].unflatten(2, (HEIGHT // self.down_factor, self.down_factor, 
                                                        WIDTH // self.down_factor, self.down_factor)) * self.spatial_weight).sum(dim=(3, 5)).flatten(2, 3)
        value_downsample = (value[:, :, 512:].unflatten(2, (HEIGHT // self.down_factor, self.down_factor, 
                                                            WIDTH // self.down_factor, self.down_factor)) * self.spatial_weight).sum(dim=(3, 5)).flatten(2, 3)

        # 🌟 第一次调用时抑制 autotune 输出
        if self._first_run:
            with _AutotuneSuppressor():
                hidden_states = self._compiled_flex_attn(query, torch.cat([key, key_downsample], dim=2), torch.cat([value, value_downsample], dim=2), scale=attention_scale)
            self._first_run = False
        else:
            hidden_states = self._compiled_flex_attn(query, torch.cat([key, key_downsample], dim=2), torch.cat([value, value_downsample], dim=2), scale=attention_scale)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
            if self.distill:
                attn_outputs.append(hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            if self.distill:
                attn_outputs.append(hidden_states)
            return hidden_states


@lru_cache
def init_local_mask_flex(height, width, text_length, window_size, device):
    # 🌟 强制使用 CUDA
    if isinstance(device, str):
        device = "cuda" if "cuda" in device else device
    else:
        device = "cuda" if device.type == "cuda" else str(device)
    
    def local_mask(b, h, q_idx, kv_idx):
        q_y = (q_idx - text_length) // width
        q_x = (q_idx - text_length) % width
        kv_y = (kv_idx - text_length) // width
        kv_x = (kv_idx - text_length) % width
        return torch.logical_or(torch.logical_or(q_idx < text_length, kv_idx < text_length),
                                (q_y - kv_y) ** 2 + (q_x - kv_x) ** 2 < window_size ** 2)
    
    global BLOCK_MASK, HEIGHT, WIDTH
    BLOCK_MASK = create_block_mask(local_mask, B=None, H=None, device=device,
                                   Q_LEN=text_length + height * width, 
                                   KV_LEN=text_length + height * width, _compile=True)
    HEIGHT = height
    WIDTH = width


class LocalFlexAttnProcessor:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self, distill=False):
        # 🌟 修复：不在 __init__ 绑定 BLOCK_MASK
        self.distill = distill
        self._compiled_flex_attn = None
        self._compiled_mask_id = None
        self._first_run = True  # 🌟 标记第一次运行

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        proportional_attention=False
    ) -> torch.FloatTensor:
        # 🌟 动态获取 BLOCK_MASK 并按需 compile
        global BLOCK_MASK
        current_mask_id = id(BLOCK_MASK)
        if self._compiled_flex_attn is None or self._compiled_mask_id != current_mask_id:
            self._compiled_flex_attn = torch.compile(
                partial(flex_attention, block_mask=BLOCK_MASK), 
                dynamic=False
            )
            self._compiled_mask_id = current_mask_id
            self._first_run = True  # 新编译后需要重新 warmup
            
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        train_seq_len = 64 ** 2 + 512
        if proportional_attention:
            attention_scale = math.sqrt(math.log(key.size(2), train_seq_len) / head_dim)
        else:
            attention_scale = math.sqrt(1 / head_dim)

        # 🌟 第一次调用时抑制 autotune 输出
        if self._first_run:
            with _AutotuneSuppressor():
                hidden_states = self._compiled_flex_attn(query, key, value, scale=attention_scale)
            self._first_run = False
        else:
            hidden_states = self._compiled_flex_attn(query, key, value, scale=attention_scale)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
            if self.distill:
                attn_outputs.append(hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            if self.distill:
                attn_outputs.append(hidden_states)
            return hidden_states