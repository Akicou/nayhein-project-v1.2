# scripts/expand_model.py
# 50M → 5B weight expansion via structured width/depth expansion.
# Implements Net2Net-style expansion + SOLAR-style layer duplication.
#
# Usage:
#   python scripts/expand_model.py \
#       --src_checkpoint outputs/50m-base/final \
#       --src_config configs/nayhein_50m.yaml \
#       --tgt_config configs/nayhein_5b.yaml \
#       --output outputs/5b-init \
#       --vision_init google/siglip-so400m-patch14-336

import argparse
import logging
import math
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from configuration_nayhein import NayheinConfig
from modeling_nayhein import NayheinForCausalLM

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Width expansion helpers
# ══════════════════════════════════════════════════════════════════════════════


def expand_weight_matrix(
    W: torch.Tensor,
    new_out: int,
    new_in: int,
    noise_std: float = 0.02,
    is_ffn: bool = False,
) -> torch.Tensor:
    """
    Expand weight matrix W from shape [out, in] to [new_out, new_in].

    Strategy:
    - Copy original weights into top-left block: W'[:out, :in] = W
    - Attention projections (is_ffn=False): zero-initialize expanded regions
    - FFN weights (is_ffn=True): fill expanded columns with small random noise
      to break symmetry (N(0, noise_std))

    Args:
        W: original weight tensor
        new_out: target output dimension
        new_in: target input dimension
        noise_std: std for random noise in FFN expansion
        is_ffn: whether this weight is an FFN matrix

    Returns:
        Expanded weight tensor of shape [new_out, new_in]
    """
    assert W.dim() == 2, f"Expected 2D weight, got {W.shape}"
    old_out, old_in = W.shape

    W_new = torch.zeros(new_out, new_in, dtype=W.dtype, device=W.device)
    # Copy original weights into top-left block
    W_new[:old_out, :old_in] = W

    if is_ffn and (new_out > old_out or new_in > old_in):
        # Fill expanded FFN regions with small noise to break symmetry
        if new_out > old_out:
            W_new[old_out:, :old_in] = (
                torch.randn(new_out - old_out, old_in, dtype=W.dtype, device=W.device)
                * noise_std
            )
        if new_in > old_in:
            W_new[:old_out, old_in:] = (
                torch.randn(old_out, new_in - old_in, dtype=W.dtype, device=W.device)
                * noise_std
            )
        if new_out > old_out and new_in > old_in:
            W_new[old_out:, old_in:] = (
                torch.randn(
                    new_out - old_out, new_in - old_in, dtype=W.dtype, device=W.device
                )
                * noise_std
            )

    return W_new


def expand_vector(v: torch.Tensor, new_size: int) -> torch.Tensor:
    """Expand 1D weight vector (bias, norm scale) to new_size."""
    assert v.dim() == 1, f"Expected 1D vector, got {v.shape}"
    old_size = v.shape[0]
    v_new = torch.ones(new_size, dtype=v.dtype, device=v.device)
    v_new[:old_size] = v
    return v_new


def expand_embedding(
    emb: torch.Tensor, new_vocab: int, new_dim: int, noise_std: float = 0.02
) -> torch.Tensor:
    """Expand embedding table from [vocab, dim] to [new_vocab, new_dim]."""
    old_vocab, old_dim = emb.shape
    emb_new = torch.zeros(new_vocab, new_dim, dtype=emb.dtype, device=emb.device)
    emb_new[:old_vocab, :old_dim] = emb
    # New tokens get small random init
    if new_vocab > old_vocab:
        emb_new[old_vocab:, :old_dim] = (
            torch.randn(
                new_vocab - old_vocab, old_dim, dtype=emb.dtype, device=emb.device
            )
            * noise_std
        )
    return emb_new


# ══════════════════════════════════════════════════════════════════════════════
# Depth expansion (12→32 layers) via layer duplication + interpolation
# ══════════════════════════════════════════════════════════════════════════════


def compute_layer_mapping(src_num_layers: int, tgt_num_layers: int) -> Dict[int, tuple]:
    """
    Compute SOLAR-style layer mapping from src_num_layers to tgt_num_layers.

    Returns a dict mapping target_layer_idx → (src_layer_a, src_layer_b, alpha)
    where:
        - src_layer_a and src_layer_b are source layer indices
        - alpha ∈ [0, 1] is the interpolation weight
        - If alpha=1.0, use only src_layer_a (direct copy)
    """
    mapping = {}
    for tgt_i in range(tgt_num_layers):
        # Map target layer to source space (float)
        src_pos = tgt_i * (src_num_layers - 1) / (tgt_num_layers - 1)
        src_a = int(math.floor(src_pos))
        src_b = min(src_a + 1, src_num_layers - 1)
        alpha = 1.0 - (src_pos - src_a)
        mapping[tgt_i] = (src_a, src_b, alpha)
    return mapping


def interpolate_layer_state_dicts(sd_a: Dict, sd_b: Dict, alpha: float) -> Dict:
    """
    Linearly interpolate two layer state dicts: alpha * sd_a + (1 - alpha) * sd_b.
    """
    result = {}
    for key in sd_a:
        if key in sd_b:
            result[key] = alpha * sd_a[key] + (1.0 - alpha) * sd_b[key]
        else:
            result[key] = sd_a[key]
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Main expansion function
# ══════════════════════════════════════════════════════════════════════════════


def expand_checkpoint(
    src_config: NayheinConfig,
    tgt_config: NayheinConfig,
    src_state_dict: Dict[str, torch.Tensor],
    noise_std: float = 0.02,
) -> Dict[str, torch.Tensor]:
    """
    Expand a 50M checkpoint to 5B using structured width + depth expansion.

    Steps:
    1. Expand embedding table (vocab × hidden)
    2. For each target layer: compute source layer mapping, expand all weights
    3. Expand MTP module weights
    4. Expand LM head
    5. Expand RMSNorm weights

    Args:
        src_config: source model config (50M)
        tgt_config: target model config (5B)
        src_state_dict: source checkpoint state dict
        noise_std: noise std for FFN symmetry breaking

    Returns:
        Expanded state dict compatible with tgt_config
    """
    logger.info(
        f"Expanding {src_config.num_hidden_layers}L/{src_config.hidden_size}D → "
        f"{tgt_config.num_hidden_layers}L/{tgt_config.hidden_size}D"
    )

    src_H = src_config.hidden_size
    tgt_H = tgt_config.hidden_size
    src_I = src_config.intermediate_size
    tgt_I = tgt_config.intermediate_size
    src_layers = src_config.num_hidden_layers
    tgt_layers = tgt_config.num_hidden_layers
    src_heads = src_config.num_attention_heads
    tgt_heads = tgt_config.num_attention_heads
    src_kv_heads = src_config.num_key_value_heads
    tgt_kv_heads = tgt_config.num_key_value_heads
    src_head_dim = src_H // src_heads
    tgt_head_dim = tgt_H // tgt_heads

    new_sd = {}

    # ── 1. Embeddings ─────────────────────────────────────────────────────────
    logger.info("Expanding embeddings...")
    emb_key = "model.embed_tokens.weight"
    if emb_key in src_state_dict:
        new_sd[emb_key] = expand_embedding(
            src_state_dict[emb_key],
            new_vocab=tgt_config.vocab_size,
            new_dim=tgt_H,
            noise_std=noise_std,
        )

    # ── 2. Transformer layers ─────────────────────────────────────────────────
    logger.info(f"Expanding {src_layers} → {tgt_layers} transformer layers...")
    layer_map = compute_layer_mapping(src_layers, tgt_layers)

    # Collect per-layer source state dicts
    def get_layer_sd(layer_idx: int) -> Dict[str, torch.Tensor]:
        prefix = f"model.layers.{layer_idx}."
        return {
            k[len(prefix) :]: v
            for k, v in src_state_dict.items()
            if k.startswith(prefix)
        }

    for tgt_i in range(tgt_layers):
        src_a, src_b, alpha = layer_map[tgt_i]
        sd_a = get_layer_sd(src_a)
        sd_b = get_layer_sd(src_b) if src_b != src_a else sd_a

        # Interpolate if needed (alpha < 1.0 means blending two adjacent layers)
        if src_a != src_b and alpha < 1.0:
            interpolated_sd = interpolate_layer_state_dicts(sd_a, sd_b, alpha)
        else:
            interpolated_sd = sd_a

        prefix = f"model.layers.{tgt_i}."

        # ── Attention weights ─────────────────────────────────────────────────
        # q_proj: [num_heads * head_dim, hidden] → [tgt_heads * tgt_head_dim, tgt_H]
        q_proj_key = "self_attn.q_proj.weight"
        if q_proj_key in interpolated_sd:
            new_sd[prefix + q_proj_key] = expand_weight_matrix(
                interpolated_sd[q_proj_key],
                tgt_heads * tgt_head_dim,
                tgt_H,
                noise_std=0.0,
                is_ffn=False,
            )

        # k_proj: [num_kv_heads * head_dim, hidden]
        k_proj_key = "self_attn.k_proj.weight"
        if k_proj_key in interpolated_sd:
            new_sd[prefix + k_proj_key] = expand_weight_matrix(
                interpolated_sd[k_proj_key],
                tgt_kv_heads * tgt_head_dim,
                tgt_H,
                noise_std=0.0,
                is_ffn=False,
            )

        # v_proj: [num_kv_heads * head_dim, hidden]
        v_proj_key = "self_attn.v_proj.weight"
        if v_proj_key in interpolated_sd:
            new_sd[prefix + v_proj_key] = expand_weight_matrix(
                interpolated_sd[v_proj_key],
                tgt_kv_heads * tgt_head_dim,
                tgt_H,
                noise_std=0.0,
                is_ffn=False,
            )

        # o_proj: [hidden, num_heads * head_dim]
        o_proj_key = "self_attn.o_proj.weight"
        if o_proj_key in interpolated_sd:
            new_sd[prefix + o_proj_key] = expand_weight_matrix(
                interpolated_sd[o_proj_key],
                tgt_H,
                tgt_heads * tgt_head_dim,
                noise_std=0.0,
                is_ffn=False,
            )

        # ── FFN weights ───────────────────────────────────────────────────────
        gate_key = "mlp.gate_proj.weight"
        if gate_key in interpolated_sd:
            new_sd[prefix + gate_key] = expand_weight_matrix(
                interpolated_sd[gate_key],
                tgt_I,
                tgt_H,
                noise_std=noise_std,
                is_ffn=True,
            )

        up_key = "mlp.up_proj.weight"
        if up_key in interpolated_sd:
            new_sd[prefix + up_key] = expand_weight_matrix(
                interpolated_sd[up_key],
                tgt_I,
                tgt_H,
                noise_std=noise_std,
                is_ffn=True,
            )

        down_key = "mlp.down_proj.weight"
        if down_key in interpolated_sd:
            new_sd[prefix + down_key] = expand_weight_matrix(
                interpolated_sd[down_key],
                tgt_H,
                tgt_I,
                noise_std=noise_std,
                is_ffn=True,
            )

        # ── RMSNorm weights ───────────────────────────────────────────────────
        for norm_key in ["input_layernorm.weight", "post_attention_layernorm.weight"]:
            if norm_key in interpolated_sd:
                new_sd[prefix + norm_key] = expand_vector(
                    interpolated_sd[norm_key], tgt_H
                )

    # ── 3. Final norm ─────────────────────────────────────────────────────────
    final_norm_key = "model.norm.weight"
    if final_norm_key in src_state_dict:
        new_sd[final_norm_key] = expand_vector(src_state_dict[final_norm_key], tgt_H)

    # ── 4. LM head ────────────────────────────────────────────────────────────
    if not tgt_config.tie_word_embeddings:
        lm_head_key = "lm_head.weight"
        if lm_head_key in src_state_dict:
            new_sd[lm_head_key] = expand_embedding(
                src_state_dict[lm_head_key],
                new_vocab=tgt_config.vocab_size,
                new_dim=tgt_H,
                noise_std=noise_std,
            )
        else:
            # Tied embeddings: use the expanded embedding
            new_sd[lm_head_key] = new_sd[emb_key].clone()

    # ── 5. MTP modules ────────────────────────────────────────────────────────
    logger.info("Expanding MTP modules...")
    for mtp_idx in range(tgt_config.mtp_num_future_tokens):
        # Source has mtp_idx 0 only (1 module for 50M), rest are random-init
        src_mtp_prefix = "mtp_modules.0."
        tgt_mtp_prefix = f"mtp_modules.{mtp_idx}."

        mtp_sd = {
            k[len(src_mtp_prefix) :]: v
            for k, v in src_state_dict.items()
            if k.startswith(src_mtp_prefix)
        }

        if mtp_idx == 0 and mtp_sd:
            # Expand existing MTP module 0
            mtp_map = {
                "proj.weight": (tgt_H, tgt_H * 2, False),  # hidden → 2*hidden
                "norm.weight": (tgt_H,),
                "input_norm.weight": (tgt_H * 2,),
            }
            for sub_key, sub_v in mtp_sd.items():
                full_key = tgt_mtp_prefix + sub_key
                if sub_v.dim() == 2:
                    new_sd[full_key] = expand_weight_matrix(
                        sub_v, tgt_H, tgt_H * 2, noise_std=0.0, is_ffn=False
                    )
                elif sub_v.dim() == 1:
                    # Expand appropriately based on expected size
                    new_size = tgt_H if sub_v.shape[0] == src_H else tgt_H * 2
                    new_sd[full_key] = expand_vector(sub_v, new_size)
        else:
            # New MTP module: random initialization (handled by model init)
            pass

    logger.info(f"Expansion complete. New state dict has {len(new_sd)} keys.")
    return new_sd


# ══════════════════════════════════════════════════════════════════════════════
# Vision encoder initialization from SigLIP
# ══════════════════════════════════════════════════════════════════════════════


def init_vision_from_siglip(
    expanded_model: NayheinForCausalLM,
    siglip_model_id: str = "google/siglip-so400m-patch14-336",
) -> NayheinForCausalLM:
    """
    Initialize the 5B vision encoder from a pretrained SigLIP checkpoint.
    Only loads compatible layers (patch_embed, transformer layers, norm).
    """
    try:
        from transformers import SiglipVisionModel

        logger.info(f"Loading SigLIP from {siglip_model_id}...")
        siglip = SiglipVisionModel.from_pretrained(siglip_model_id)
        siglip_sd = siglip.state_dict()

        nayhein_vision_sd = {}
        for k, v in siglip_sd.items():
            # Map SigLIP parameter names to NayheinVisionEncoder names
            new_k = k
            new_k = new_k.replace("vision_model.encoder.layers.", "layers.")
            new_k = new_k.replace(
                "vision_model.embeddings.patch_embedding.", "patch_embed.proj."
            )
            new_k = new_k.replace("vision_model.post_layernorm.", "norm.")

            # Transformers use fc1/fc2 for MLP, NayheinVisionEncoder uses mlp.0/mlp.2
            new_k = new_k.replace(".mlp.fc1.", ".mlp.0.")
            new_k = new_k.replace(".mlp.fc2.", ".mlp.2.")

            nayhein_vision_sd[new_k] = v

        missing, unexpected = expanded_model.model.vision_encoder.load_state_dict(
            nayhein_vision_sd, strict=False
        )
        logger.info(
            f"Vision encoder loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}"
        )

        del siglip
        torch.cuda.empty_cache()

    except Exception as e:
        logger.warning(
            f"Could not load SigLIP weights: {e}. Vision encoder uses random init."
        )

    return expanded_model


# ══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Expand Nayhein-V1.2 50M checkpoint to 5B via structured weight expansion."
    )
    parser.add_argument(
        "--src_checkpoint", required=True, help="Path to 50M checkpoint directory"
    )
    parser.add_argument("--src_config", required=True, help="Path to 50M YAML config")
    parser.add_argument("--tgt_config", required=True, help="Path to 5B YAML config")
    parser.add_argument(
        "--output", required=True, help="Output directory for expanded checkpoint"
    )
    parser.add_argument(
        "--vision_init",
        default="google/siglip-so400m-patch14-336",
        help="HuggingFace model ID for vision encoder initialization",
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        default=0.02,
        help="Noise std for FFN weight expansion (symmetry breaking)",
    )
    args = parser.parse_args()

    # ── Load source checkpoint ────────────────────────────────────────────────
    logger.info(f"Loading 50M checkpoint from {args.src_checkpoint}...")
    src_config = NayheinConfig.from_pretrained(args.src_checkpoint)
    tgt_config = NayheinConfig.nayhein_5b()

    # Load source state dict
    src_model = NayheinForCausalLM.from_pretrained(
        args.src_checkpoint,
        config=src_config,
        torch_dtype=torch.float32,  # expand in fp32 for precision
    )
    src_sd = {k: v.cpu() for k, v in src_model.state_dict().items()}
    del src_model
    torch.cuda.empty_cache()
    logger.info(f"Loaded source state dict ({len(src_sd)} keys)")

    # ── Expand weights ────────────────────────────────────────────────────────
    expanded_sd = expand_checkpoint(
        src_config, tgt_config, src_sd, noise_std=args.noise_std
    )

    # ── Build 5B model and load expanded weights ──────────────────────────────
    logger.info("Building 5B model...")
    tgt_model = NayheinForCausalLM(tgt_config)

    # Load with strict=False — vision encoder and newly-added MTP layers
    # may not have direct matches from the expansion
    missing, unexpected = tgt_model.load_state_dict(expanded_sd, strict=False)
    logger.info(
        f"Loaded expanded weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}"
    )

    if missing:
        logger.info(
            f"Missing keys (will use random init): {missing[:10]}{'...' if len(missing) > 10 else ''}"
        )

    # ── Initialize vision encoder from SigLIP ─────────────────────────────────
    if args.vision_init:
        tgt_model = init_vision_from_siglip(tgt_model, args.vision_init)

    # ── Save expanded model ───────────────────────────────────────────────────
    os.makedirs(args.output, exist_ok=True)
    logger.info(f"Saving 5B expanded model to {args.output}...")

    try:
        from safetensors.torch import save_file

        save_file(
            tgt_model.state_dict(), os.path.join(args.output, "model.safetensors")
        )
    except ImportError:
        torch.save(
            tgt_model.state_dict(), os.path.join(args.output, "pytorch_model.bin")
        )

    tgt_config.save_pretrained(args.output)
    logger.info("Expansion complete.")

    # Report parameter count
    n_params = sum(p.numel() for p in tgt_model.parameters()) / 1e9
    logger.info(f"5B model parameter count: {n_params:.2f}B")


if __name__ == "__main__":
    main()
