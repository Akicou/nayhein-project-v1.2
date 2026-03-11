# configuration_nayhein.py
# NayheinConfig — auto-registered with HuggingFace Transformers
# Supports both 50M and 5B model sizes via the same class.

from transformers import PretrainedConfig
from typing import Optional, Dict, Any


class NayheinConfig(PretrainedConfig):
    """
    Configuration for the NayheinHDT (Hybrid Diffusion Transformer) model family.

    Supports both 50M and 5B model sizes. The 5B model is grown from the 50M
    checkpoint using structured weight expansion (see scripts/expand_model.py).

    Architecture components:
    - GQA Transformer backbone (LLaMA-3 style) with RMSNorm + SwiGLU + RoPE+YaRN
    - Multi-Token Prediction (MTP) heads — DeepSeek-V3 style
    - Masked Diffusion Language Model (MDLM) head — Mercury 2 inspired
    - Baked-in SigLIP Vision Encoder + Perceiver Resampler
    - Tool calling via NayheinToolCallingMixin
    """

    model_type = "nayhein"

    def __init__(
        self,
        # ── Transformer backbone ──────────────────────────────────────────────
        vocab_size: int = 65536,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        num_key_value_heads: int = 4,  # GQA — 3:1 ratio for 50M
        intermediate_size: int = 2048,  # SwiGLU FFN inner dim
        hidden_act: str = "silu",  # SwiGLU gate activation
        rms_norm_eps: float = 1e-6,
        # ── Positional encoding ───────────────────────────────────────────────
        max_position_embeddings: int = 32768,
        rope_theta: float = 500000.0,
        rope_scaling: Optional[Dict[str, Any]] = None,
        # ── Attention ─────────────────────────────────────────────────────────
        attention_dropout: float = 0.0,
        # ── Embeddings ────────────────────────────────────────────────────────
        tie_word_embeddings: bool = True,
        # ── MTP (Multi-Token Prediction) ──────────────────────────────────────
        mtp_num_future_tokens: int = 4,
        mtp_num_layers: int = 1,  # transformer layers per MTP head
        # ── MDLM Diffusion ────────────────────────────────────────────────────
        diffusion_steps: int = 16,
        diffusion_mask_rate_schedule: str = "cosine",
        diffusion_min_mask_rate: float = 0.0,
        diffusion_max_mask_rate: float = 1.0,
        diffusion_training_probability: float = 0.5,  # fraction of batches using diffusion loss
        # ── Vision encoder ────────────────────────────────────────────────────
        vision_enabled: bool = True,
        vision_model_type: str = "siglip",
        vision_hidden_size: int = 768,
        vision_num_layers: int = 6,
        vision_num_heads: int = 12,
        vision_patch_size: int = 16,
        vision_image_size: int = 256,
        vision_num_prefix_tokens: int = 64,  # Perceiver resampler output tokens
        vision_freeze_in_pretrain: bool = True,
        # ── Misc ──────────────────────────────────────────────────────────────
        torch_dtype: str = "bfloat16",
        pad_token_id: int = 0,
        bos_token_id: int = 2,
        eos_token_id: int = 1,
        mask_token_id: int = 4,
        **kwargs,
    ):
        # Default YaRN rope scaling if not provided
        if rope_scaling is None:
            rope_scaling = {
                "type": "yarn",
                "factor": 8.0,
                "original_max_position_embeddings": 4096,
                "attention_factor": 0.1,
            }

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps

        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

        self.attention_dropout = attention_dropout
        self.tie_word_embeddings = tie_word_embeddings

        self.mtp_num_future_tokens = mtp_num_future_tokens
        self.mtp_num_layers = mtp_num_layers

        self.diffusion_steps = diffusion_steps
        self.diffusion_mask_rate_schedule = diffusion_mask_rate_schedule
        self.diffusion_min_mask_rate = diffusion_min_mask_rate
        self.diffusion_max_mask_rate = diffusion_max_mask_rate
        self.diffusion_training_probability = diffusion_training_probability

        self.vision_enabled = vision_enabled
        self.vision_model_type = vision_model_type
        self.vision_hidden_size = vision_hidden_size
        self.vision_num_layers = vision_num_layers
        self.vision_num_heads = vision_num_heads
        self.vision_patch_size = vision_patch_size
        self.vision_image_size = vision_image_size
        self.vision_num_prefix_tokens = vision_num_prefix_tokens
        self.vision_freeze_in_pretrain = vision_freeze_in_pretrain

        self.torch_dtype = torch_dtype
        self.mask_token_id = mask_token_id

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @classmethod
    def nayhein_50m(cls) -> "NayheinConfig":
        """Return the canonical 50M pretraining config."""
        return cls(
            vocab_size=65536,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            num_key_value_heads=4,
            intermediate_size=2048,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            max_position_embeddings=32768,
            rope_theta=500000.0,
            rope_scaling={
                "type": "yarn",
                "factor": 8.0,
                "original_max_position_embeddings": 4096,
                "attention_factor": 0.1,
            },
            mtp_num_future_tokens=4,
            mtp_num_layers=1,
            diffusion_steps=16,
            diffusion_mask_rate_schedule="cosine",
            diffusion_min_mask_rate=0.0,
            diffusion_max_mask_rate=1.0,
            vision_enabled=True,
            vision_model_type="siglip",
            vision_hidden_size=768,
            vision_num_layers=6,
            vision_num_heads=12,
            vision_patch_size=16,
            vision_image_size=256,
            vision_num_prefix_tokens=64,
            vision_freeze_in_pretrain=True,
            tie_word_embeddings=True,
            attention_dropout=0.0,
            torch_dtype="bfloat16",
        )

    @classmethod
    def nayhein_5b(cls) -> "NayheinConfig":
        """Return the canonical 5B config (grown from 50M via expand_model.py)."""
        return cls(
            vocab_size=65536,
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            intermediate_size=14336,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            max_position_embeddings=32768,
            rope_theta=500000.0,
            rope_scaling={
                "type": "yarn",
                "factor": 8.0,
                "original_max_position_embeddings": 4096,
                "attention_factor": 0.1,
            },
            mtp_num_future_tokens=4,
            mtp_num_layers=2,
            diffusion_steps=32,
            diffusion_mask_rate_schedule="cosine",
            diffusion_min_mask_rate=0.0,
            diffusion_max_mask_rate=1.0,
            vision_enabled=True,
            vision_model_type="siglip",
            vision_hidden_size=1024,
            vision_num_layers=24,
            vision_num_heads=16,
            vision_patch_size=14,
            vision_image_size=336,
            vision_num_prefix_tokens=256,
            vision_freeze_in_pretrain=True,
            tie_word_embeddings=False,
            attention_dropout=0.0,
            torch_dtype="bfloat16",
        )
