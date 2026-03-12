# modeling_nayhein.py
# NayheinHDT — Hybrid Diffusion Transformer
# Full model implementation: backbone, GQA+RoPE+YaRN, SwiGLU, MTP, MDLM, Vision.
# Upload this file to HuggingFace with trust_remote_code=True.

import math
import warnings
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from transformers import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.utils import logging

from configuration_nayhein import NayheinConfig

logger = logging.get_logger(__name__)

# Attention backend: PyTorch SDPA (torch.nn.functional.scaled_dot_product_attention).
# No flash-attn dependency required. SDPA automatically selects the most efficient
# kernel available on the current hardware (FlashAttention-2, memory-efficient, or math).


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════


def _make_causal_mask(seq_len: int, dtype: torch.dtype, device: torch.device) -> Tensor:
    """Upper-triangular causal mask (True = masked)."""
    mask = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool).triu(1)
    return mask


def rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor
) -> Tuple[Tensor, Tensor]:
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


# ══════════════════════════════════════════════════════════════════════════════
# RoPE with YaRN context extension
# ══════════════════════════════════════════════════════════════════════════════


class NayheinRotaryEmbedding(nn.Module):
    """
    RoPE with optional YaRN context extension.
    YaRN: https://arxiv.org/abs/2309.00071
    """

    def __init__(self, config: NayheinConfig):
        super().__init__()
        self.dim = config.hidden_size // config.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.base = config.rope_theta
        self.rope_scaling = config.rope_scaling

        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )

        # Apply YaRN scaling if configured
        if self.rope_scaling and self.rope_scaling.get("type") == "yarn":
            inv_freq = self._apply_yarn(inv_freq, config)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(config.max_position_embeddings)

    def _apply_yarn(self, inv_freq: Tensor, config: NayheinConfig) -> Tensor:
        """Apply YaRN frequency interpolation."""
        scale_factor = self.rope_scaling.get("factor", 8.0)
        orig_max_pos = self.rope_scaling.get("original_max_position_embeddings", 4096)
        attn_factor = self.rope_scaling.get("attention_factor", 0.1)

        # YaRN: scale only the low-frequency components
        low_freq_factor = 1.0
        high_freq_factor = 4.0
        low_freq_wavelen = orig_max_pos / low_freq_factor
        high_freq_wavelen = orig_max_pos / high_freq_factor

        new_freqs = []
        for i, freq in enumerate(inv_freq):
            wavelen = 2 * math.pi / freq.item()
            if wavelen < high_freq_wavelen:
                new_freqs.append(freq)
            elif wavelen > low_freq_wavelen:
                new_freqs.append(freq / scale_factor)
            else:
                smooth = (orig_max_pos / wavelen - low_freq_factor) / (
                    high_freq_factor - low_freq_factor
                )
                new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)

        return torch.tensor(new_freqs, dtype=inv_freq.dtype, device=inv_freq.device)

    def _build_cache(self, seq_len: int):
        t = torch.arange(
            seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device
        )
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :], persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :], persistent=False
        )
        self._cached_seq_len = seq_len

    def forward(self, x: Tensor, seq_len: int) -> Tuple[Tensor, Tensor]:
        if seq_len > self._cached_seq_len:
            self._build_cache(seq_len)
        return (
            self.cos_cached[:, :, :seq_len, :].to(x.dtype),
            self.sin_cached[:, :, :seq_len, :].to(x.dtype),
        )


# ══════════════════════════════════════════════════════════════════════════════
# Sinusoidal timestep embedding (for MDLM diffusion conditioning)
# ══════════════════════════════════════════════════════════════════════════════


class SinusoidalTimestepEmbedding(nn.Module):
    """
    Embeds a scalar diffusion timestep t ∈ [0, 1] into hidden_size dimensions.
    Architecture: Sinusoidal(t) → Linear → SiLU → Linear
    Output is added to the residual stream at every transformer layer.
    """

    def __init__(self, hidden_size: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.proj = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t: Tensor) -> Tensor:
        """
        Args:
            t: float tensor of shape (batch,) or scalar in [0, 1]
        Returns:
            emb: (batch, hidden_size)
        """
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(half, device=t.device, dtype=t.dtype)
            / (half - 1)
        )
        # t: (B,) → (B, half)
        t_scaled = t.unsqueeze(-1) * freqs.unsqueeze(0) * 1000
        emb = torch.cat([t_scaled.sin(), t_scaled.cos()], dim=-1)  # (B, freq_dim)
        return self.proj(emb)  # (B, hidden_size)


# ══════════════════════════════════════════════════════════════════════════════
# RMSNorm
# ══════════════════════════════════════════════════════════════════════════════


class NayheinRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * x).to(dtype)


# ══════════════════════════════════════════════════════════════════════════════
# Grouped Query Attention (GQA) with RoPE
# ══════════════════════════════════════════════════════════════════════════════


class NayheinAttention(nn.Module):
    """
    GQA attention with RoPE. Falls back to SDPA if flash_attn unavailable.
    No bias in projection matrices.
    """

    def __init__(self, config: NayheinConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

    def _expand_kv(self, x: Tensor) -> Tensor:
        """Expand KV heads to match Q head count (GQA group repeat)."""
        # x: (B, kv_heads, S, head_dim) → (B, num_heads, S, head_dim)
        B, kv_h, S, D = x.shape
        x = x[:, :, None, :, :].expand(B, kv_h, self.num_kv_groups, S, D)
        return x.reshape(B, kv_h * self.num_kv_groups, S, D)

    def forward(
        self,
        hidden_states: Tensor,
        cos: Tensor,
        sin: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        B, S, _ = hidden_states.shape

        q = (
            self.q_proj(hidden_states)
            .view(B, S, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(hidden_states)
            .view(B, S, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(hidden_states)
            .view(B, S, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Apply RoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Handle KV cache
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        present = (k, v) if use_cache else None

        # Expand KV for GQA
        k_expanded = self._expand_kv(k)
        v_expanded = self._expand_kv(v)

        # Attention computation — PyTorch SDPA (no flash-attn dependency).
        # torch.backends.cuda.enable_flash_sdp / enable_mem_efficient_sdp can be
        # set at the call site to tune the kernel choice on CUDA hardware.
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_out = F.scaled_dot_product_attention(
            q,
            k_expanded,
            v_expanded,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=(attention_mask is None),
            scale=scale,
        )  # (B, H, S, D)
        attn_out = attn_out.transpose(1, 2).reshape(B, S, self.hidden_size)

        return self.o_proj(attn_out), present


# ══════════════════════════════════════════════════════════════════════════════
# SwiGLU Feed-Forward Network
# ══════════════════════════════════════════════════════════════════════════════


class NayheinMLP(nn.Module):
    """SwiGLU MLP: W_gate(x) * silu(W_up(x)) projected by W_down. No bias."""

    def __init__(self, config: NayheinConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ══════════════════════════════════════════════════════════════════════════════
# Transformer Decoder Layer
# ══════════════════════════════════════════════════════════════════════════════


class NayheinDecoderLayer(nn.Module):
    """
    Single transformer decoder layer.
    Pre-norm: RMSNorm → Attention → residual; RMSNorm → MLP → residual.
    Optionally adds diffusion timestep embedding to the residual stream.
    """

    def __init__(self, config: NayheinConfig, layer_idx: int):
        super().__init__()
        self.self_attn = NayheinAttention(config, layer_idx)
        self.mlp = NayheinMLP(config)
        self.input_layernorm = NayheinRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = NayheinRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: Tensor,
        cos: Tensor,
        sin: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
        time_emb: Optional[Tensor] = None,  # (B, hidden_size) for diffusion
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        # Self-attention block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present = self.self_attn(
            hidden_states, cos, sin, attention_mask, past_key_value, use_cache
        )
        hidden_states = residual + hidden_states

        # Inject diffusion timestep (broadcast over sequence)
        if time_emb is not None:
            hidden_states = hidden_states + time_emb.unsqueeze(1)

        # MLP block
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, present


# ══════════════════════════════════════════════════════════════════════════════
# Multi-Token Prediction Module (DeepSeek-V3 style)
# ══════════════════════════════════════════════════════════════════════════════


class NayheinMTPModule(nn.Module):
    """
    Lightweight transformer module for predicting one future token.

    MTP module k predicts token t+k+1 given:
      - shared backbone hidden state at t
      - embedding of the AR prediction at t+k
    """

    def __init__(self, config: NayheinConfig):
        super().__init__()
        self.input_norm = NayheinRMSNorm(config.hidden_size * 2, config.rms_norm_eps)
        self.proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.layers = nn.ModuleList(
            [
                NayheinDecoderLayer(config, layer_idx=i)
                for i in range(config.mtp_num_layers)
            ]
        )
        self.norm = NayheinRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        backbone_hidden: Tensor,  # (B, S, H) — main backbone output
        prev_token_emb: Tensor,  # (B, S, H) — embedding of previously predicted token
        cos: Tensor,
        sin: Tensor,
    ) -> Tensor:
        """Returns hidden states for LM head to predict next future token."""
        x = torch.cat([backbone_hidden, prev_token_emb], dim=-1)
        x = self.input_norm(x)
        x = self.proj(x)
        for layer in self.layers:
            x, _ = layer(x, cos, sin)
        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════════════
# Vision Encoder — SigLIP-style ViT + Perceiver Resampler
# ══════════════════════════════════════════════════════════════════════════════


class VisionPatchEmbed(nn.Module):
    """Image → patch tokens via 2D conv."""

    def __init__(
        self, image_size: int, patch_size: int, in_channels: int, embed_dim: int
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, H, W) → (B, N, embed_dim)
        return self.proj(x).flatten(2).transpose(1, 2)


class VisionTransformerLayer(nn.Module):
    """Single ViT encoder layer with pre-norm."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, embed_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = (
            x
            + self.attn(
                self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False
            )[0]
        )
        x = x + self.mlp(self.norm2(x))
        return x


class PerceiverResampler(nn.Module):
    """
    Perceiver Resampler: compresses patch tokens to num_latents learned queries
    via cross-attention. Reduces variable-length visual sequences to a fixed size.
    """

    def __init__(
        self, num_latents: int, embed_dim: int, vision_dim: int, num_heads: int = 8
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, num_latents, embed_dim) * 0.02)
        self.norm_latents = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, kdim=vision_dim, vdim=vision_dim, batch_first=True
        )
        self.norm_ff = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, patch_features: Tensor) -> Tensor:
        # patch_features: (B, N_patches, vision_dim)
        B = patch_features.shape[0]
        latents = self.latents.expand(B, -1, -1)
        latents = self.norm_latents(latents)
        latents, _ = self.cross_attn(latents, patch_features, patch_features)
        latents = latents + self.ff(self.norm_ff(latents))
        return latents  # (B, num_latents, embed_dim)


class NayheinVisionEncoder(nn.Module):
    """
    Baked-in SigLIP-style Vision Transformer with Perceiver Resampler.
    Outputs vision_num_prefix_tokens visual tokens of size hidden_size.
    Fully bypassed during text-only inference.
    """

    def __init__(self, config: NayheinConfig):
        super().__init__()
        v = config
        self.patch_embed = VisionPatchEmbed(
            image_size=v.vision_image_size,
            patch_size=v.vision_patch_size,
            in_channels=3,
            embed_dim=v.vision_hidden_size,
        )
        num_patches = (v.vision_image_size // v.vision_patch_size) ** 2
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches, v.vision_hidden_size) * 0.02
        )
        self.layers = nn.ModuleList(
            [
                VisionTransformerLayer(v.vision_hidden_size, v.vision_num_heads)
                for _ in range(v.vision_num_layers)
            ]
        )
        self.norm = nn.LayerNorm(v.vision_hidden_size)
        self.resampler = PerceiverResampler(
            num_latents=v.vision_num_prefix_tokens,
            embed_dim=v.hidden_size,
            vision_dim=v.vision_hidden_size,
        )
        # 2-layer MLP projector: vision_hidden_size → hidden_size
        self.projector = nn.Sequential(
            nn.Linear(v.hidden_size, v.hidden_size),
            nn.GELU(),
            nn.Linear(v.hidden_size, v.hidden_size),
        )

    def forward(self, pixel_values: Tensor) -> Tensor:
        """
        Args:
            pixel_values: (B, C, H, W) float32 normalized images
        Returns:
            visual_tokens: (B, num_prefix_tokens, hidden_size)
        """
        x = self.patch_embed(pixel_values) + self.pos_embed
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.resampler(x)  # (B, num_prefix_tokens, hidden_size)
        x = self.projector(x)  # (B, num_prefix_tokens, hidden_size)
        return x


# ══════════════════════════════════════════════════════════════════════════════
# Core Nayhein Model (backbone only, no LM head)
# ══════════════════════════════════════════════════════════════════════════════


class NayheinModel(PreTrainedModel):
    """
    NayheinModel: the shared backbone used by NayheinForCausalLM.
    GQA transformer with RMSNorm, SwiGLU, RoPE+YaRN.
    """

    config_class = NayheinConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def __init__(self, config: NayheinConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0
        )
        self.layers = nn.ModuleList(
            [NayheinDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = NayheinRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = NayheinRotaryEmbedding(config)
        self.gradient_checkpointing = False

        # Vision encoder (optional, frozen during pretrain)
        if config.vision_enabled:
            self.vision_encoder = NayheinVisionEncoder(config)
        else:
            self.vision_encoder = None

        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding):
        self.embed_tokens = value

    def _prepare_vision_tokens(
        self,
        input_ids: Tensor,
        pixel_values: Optional[Tensor],
        vision_start_id: int,
        vision_end_id: int,
    ) -> Tensor:
        """
        Replace [vision_start, ..., vision_end] spans with visual tokens.
        Returns updated input embeddings tensor.
        """
        inputs_embeds = self.embed_tokens(input_ids)
        if pixel_values is None or self.vision_encoder is None:
            return inputs_embeds

        visual_tokens = self.vision_encoder(pixel_values)  # (B, N_vis, H)
        B, S, H = inputs_embeds.shape
        N_vis = visual_tokens.shape[1]

        new_embeds_list = []
        for b in range(B):
            row = inputs_embeds[b]  # (S, H)
            vs_positions = (input_ids[b] == vision_start_id).nonzero(as_tuple=True)[0]
            ve_positions = (input_ids[b] == vision_end_id).nonzero(as_tuple=True)[0]

            if len(vs_positions) == 0 or len(ve_positions) == 0:
                new_embeds_list.append(row)
                continue

            # Replace span [vs:ve+1] with visual tokens (truncate/pad as needed)
            vs = vs_positions[0].item()
            ve = ve_positions[0].item()
            prefix = row[: vs + 1]  # include vision_start token
            suffix = row[ve:]  # include vision_end token
            vis = visual_tokens[b]  # (N_vis, H)
            merged = torch.cat([prefix, vis, suffix], dim=0)
            # Pad/truncate to original seq len for batching
            if merged.shape[0] < S:
                pad = torch.zeros(
                    S - merged.shape[0], H, device=merged.device, dtype=merged.dtype
                )
                merged = torch.cat([merged, pad], dim=0)
            elif merged.shape[0] > S:
                merged = merged[:S]
            new_embeds_list.append(merged)

        return torch.stack(new_embeds_list, dim=0)

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        pixel_values: Optional[Tensor] = None,
        past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
        inputs_embeds: Optional[Tensor] = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
        diffusion_t: Optional[Tensor] = None,  # (B,) scalar timestep for MDLM
        time_embed_fn: Optional[nn.Module] = None,
    ) -> BaseModelOutputWithPast:

        if inputs_embeds is None:
            inputs_embeds = self._prepare_vision_tokens(
                input_ids,
                pixel_values,
                vision_start_id=self.config.bos_token_id,  # mapped via tokenizer
                vision_end_id=self.config.eos_token_id,
            )

        B, S, _ = inputs_embeds.shape
        cos, sin = self.rotary_emb(inputs_embeds, seq_len=S)

        # Compute diffusion timestep embedding if requested
        time_emb = None
        if diffusion_t is not None and time_embed_fn is not None:
            time_emb = time_embed_fn(diffusion_t)  # (B, H)

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        next_cache = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_kv = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                hidden_states, present = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    cos,
                    sin,
                    attention_mask,
                    past_kv,
                    use_cache,
                    time_emb,
                    use_reentrant=False,
                )
            else:
                hidden_states, present = layer(
                    hidden_states,
                    cos,
                    sin,
                    attention_mask,
                    past_kv,
                    use_cache,
                    time_emb,
                )

            if use_cache:
                next_cache.append(present)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache if use_cache else None,
            hidden_states=all_hidden_states,
        )


# ══════════════════════════════════════════════════════════════════════════════
# NayheinForCausalLM — full model with AR head, MTP heads, diffusion head
# ══════════════════════════════════════════════════════════════════════════════


class NayheinForCausalLM(PreTrainedModel):
    """
    Nayhein-V1.2 full model.

    Supports:
    - Autoregressive (AR) language modeling
    - Multi-Token Prediction (MTP) — DeepSeek-V3 style
    - Masked Diffusion Language Modeling (MDLM) — Mercury 2 style
    - Vision-language modeling via baked-in SigLIP ViT

    Use NayheinForCausalLM.from_pretrained(..., trust_remote_code=True)
    """

    config_class = NayheinConfig
    base_model_prefix = "model"
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: NayheinConfig):
        super().__init__(config)
        self.model = NayheinModel(config)

        # ── AR LM head ────────────────────────────────────────────────────────
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # ── Diffusion timestep embedding ──────────────────────────────────────
        self.time_embed = SinusoidalTimestepEmbedding(config.hidden_size)

        # ── MTP modules ───────────────────────────────────────────────────────
        self.mtp_modules = nn.ModuleList(
            [NayheinMTPModule(config) for _ in range(config.mtp_num_future_tokens)]
        )

        # Weight tying
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        self.post_init()

    # ── Weight initialization ─────────────────────────────────────────────────

    def _init_weights(self, module: nn.Module):
        std = 0.02
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    # ── Model helpers ─────────────────────────────────────────────────────────

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def freeze_vision_encoder(self):
        if self.model.vision_encoder is not None:
            for p in self.model.vision_encoder.parameters():
                p.requires_grad = False

    def unfreeze_vision_encoder(self):
        if self.model.vision_encoder is not None:
            for p in self.model.vision_encoder.parameters():
                p.requires_grad = True

    # ── Diffusion helpers ─────────────────────────────────────────────────────

    @staticmethod
    def cosine_mask_rate(t: Tensor) -> Tensor:
        """α(t) = cos²(πt/2) — fraction of tokens to mask at timestep t."""
        return torch.cos(math.pi * t / 2) ** 2

    def corrupt_sequence(
        self, input_ids: Tensor, t: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward diffusion: mask tokens with probability α(t).
        Returns (corrupted_ids, mask) where mask=True at masked positions.
        """
        device = input_ids.device
        B, S = input_ids.shape
        if t is None:
            t = torch.rand(B, device=device)
        alpha = self.cosine_mask_rate(t).unsqueeze(-1)  # (B, 1)
        mask = torch.bernoulli(alpha.expand(B, S)).bool()
        corrupted = input_ids.clone()
        corrupted[mask] = self.config.mask_token_id
        return corrupted, mask

    # ── Core forward pass ─────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        pixel_values: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        past_key_values: Optional[List] = None,
        inputs_embeds: Optional[Tensor] = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
        # Diffusion-specific
        diffusion_t: Optional[Tensor] = None,
        diffusion_mask: Optional[Tensor] = None,
        # Loss weights (can override config)
        loss_ar_weight: float = 1.0,
        loss_mtp_weight: float = 0.3,
        loss_diffusion_weight: float = 0.5,
        use_diffusion_loss: bool = False,
    ) -> CausalLMOutputWithPast:

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            diffusion_t=diffusion_t,
            time_embed_fn=self.time_embed if diffusion_t is not None else None,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self._compute_combined_loss(
                logits,
                labels,
                hidden_states,
                input_ids=input_ids,
                diffusion_t=diffusion_t,
                diffusion_mask=diffusion_mask,
                loss_ar_weight=loss_ar_weight,
                loss_mtp_weight=loss_mtp_weight,
                loss_diffusion_weight=loss_diffusion_weight,
                use_diffusion_loss=use_diffusion_loss,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )

    def _compute_combined_loss(
        self,
        logits: Tensor,
        labels: Tensor,
        hidden_states: Tensor,
        input_ids: Optional[Tensor],
        diffusion_t: Optional[Tensor],
        diffusion_mask: Optional[Tensor],
        loss_ar_weight: float,
        loss_mtp_weight: float,
        loss_diffusion_weight: float,
        use_diffusion_loss: bool,
    ) -> Tensor:
        B, S, V = logits.shape

        # ── AR loss (next-token prediction) ───────────────────────────────────
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        l_ar = F.cross_entropy(
            shift_logits.view(-1, V),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        total_loss = loss_ar_weight * l_ar

        # ── MTP loss ──────────────────────────────────────────────────────────
        if loss_mtp_weight > 0 and labels is not None:
            cos, sin = self.model.rotary_emb(hidden_states, seq_len=S)
            prev_emb = self.model.embed_tokens(
                torch.clamp(labels, min=0)
            )  # (B, S, H); clamp -100 to 0 for embedding lookup
            l_mtp_total = torch.tensor(0.0, device=logits.device)
            for k, mtp_module in enumerate(self.mtp_modules):
                mtp_hidden = mtp_module(hidden_states, prev_emb, cos, sin)
                mtp_logits = self.lm_head(mtp_hidden)  # (B, S, V)
                # MTP module k predicts token t+k+2
                offset = k + 2
                if S > offset:
                    mtp_shift_logits = mtp_logits[:, :-offset, :].contiguous()
                    mtp_shift_labels = labels[:, offset:].contiguous()
                    l_mtp_k = F.cross_entropy(
                        mtp_shift_logits.view(-1, V),
                        mtp_shift_labels.view(-1),
                        ignore_index=-100,
                    )
                    l_mtp_total = l_mtp_total + l_mtp_k
                    # Shift prev_emb for next MTP module
                    prev_emb = self.model.embed_tokens(
                        torch.clamp(torch.argmax(mtp_logits, dim=-1), min=0)
                    )
            total_loss = total_loss + loss_mtp_weight * l_mtp_total

        # ── Diffusion loss (MDLM) ─────────────────────────────────────────────
        if (
            use_diffusion_loss
            and diffusion_mask is not None
            and loss_diffusion_weight > 0
        ):
            l_diff = F.cross_entropy(
                logits.view(-1, V),
                labels.view(-1),
                ignore_index=-100,
                reduction="none",
            ).view(B, S)
            # Only average loss at masked positions
            masked_loss = (l_diff * diffusion_mask.float()).sum() / (
                diffusion_mask.float().sum() + 1e-8
            )
            total_loss = total_loss + loss_diffusion_weight * masked_loss

        return total_loss

    # ── prepare_inputs_for_generation (required by Transformers) ─────────────

    def prepare_inputs_for_generation(
        self,
        input_ids: Tensor,
        past_key_values=None,
        attention_mask=None,
        pixel_values=None,
        **kwargs,
    ):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "use_cache": True,
        }

    def _reorder_cache(self, past_key_values, beam_idx):
        return tuple(
            tuple(past_state.index_select(0, beam_idx) for past_state in layer_past)
            for layer_past in past_key_values
        )

    # ── Convenience: chat interface ───────────────────────────────────────────

    @torch.inference_mode()
    def chat(
        self,
        messages,
        tokenizer=None,
        tools=None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        generation_mode: str = "ar",
    ):
        """
        High-level chat interface. Returns string or ToolCallOutput.
        See generation_utils.py for full generation logic.
        """
        from generation_utils import NayheinGenerationMixin
        from tool_calling import NayheinToolCallingMixin

        mixin = NayheinGenerationMixin(self, tokenizer)
        if tools:
            tc_mixin = NayheinToolCallingMixin()
            tools_xml = tc_mixin.format_tools(tools)
        else:
            tools_xml = None

        return mixin.generate_chat(
            messages=messages,
            tools_xml=tools_xml,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            generation_mode=generation_mode,
        )


# ── Auto-register with HuggingFace ────────────────────────────────────────────
from transformers import AutoConfig, AutoModelForCausalLM

AutoConfig.register("nayhein", NayheinConfig)
AutoModelForCausalLM.register(NayheinConfig, NayheinForCausalLM)
