# train/pretrain.py
# Pretraining script for Nayhein-V1.2 (Stage 1: 50M base, Stage 2: 5B adaptation).
# Uses torchrun + FSDP (FullyShardedDataParallel) for 4×H100 multi-GPU training.
#
# Launch:
#   torchrun --nproc_per_node=4 --master_port=29500 \
#       train/pretrain.py --config configs/nayhein_50m.yaml \
#       --output_dir outputs/50m-base --run_name nayhein-v1.2-50m-pretrain

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from torch.utils.data import DataLoader, DistributedSampler
import yaml

# Add parent dir to path so local imports resolve
sys.path.insert(0, str(Path(__file__).parent.parent))

from configuration_nayhein import NayheinConfig
from modeling_nayhein import NayheinForCausalLM, NayheinDecoderLayer
from tokenization_nayhein import NayheinTokenizer
from train.data_utils import PackedSequenceDataset, build_pretrain_token_stream

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Config loading
# ══════════════════════════════════════════════════════════════════════════════


def load_config(config_path: str) -> Dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ══════════════════════════════════════════════════════════════════════════════
# LR scheduler
# ══════════════════════════════════════════════════════════════════════════════


def get_cosine_with_warmup_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    from torch.optim.lr_scheduler import LambdaLR

    return LambdaLR(optimizer, lr_lambda)


# ══════════════════════════════════════════════════════════════════════════════
# FSDP setup
# ══════════════════════════════════════════════════════════════════════════════


def build_fsdp_model(model: NayheinForCausalLM, cfg: Dict) -> FSDP:
    """Wrap model in FSDP with configured settings."""
    from functools import partial

    # Mixed precision policy
    bf16_policy = (
        MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
        if cfg.get("fsdp_mixed_precision") == "bfloat16"
        else None
    )

    # Wrap policy: wrap each transformer layer individually
    auto_wrap = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={NayheinDecoderLayer},
    )

    sharding = ShardingStrategy.FULL_SHARD

    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap,
        sharding_strategy=sharding,
        mixed_precision=bf16_policy,
        device_id=torch.cuda.current_device(),
        use_orig_params=True,  # required for torch.compile
    )

    # Enable activation checkpointing
    if cfg.get("fsdp_activation_checkpointing", True):
        from torch.distributed.fsdp.wrap import enable_wrap, wrap
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            apply_activation_checkpointing,
            checkpoint_wrapper,
            CheckpointImpl,
        )

        apply_activation_checkpointing(
            fsdp_model,
            checkpoint_wrapper_fn=checkpoint_wrapper,
            check_fn=lambda m: isinstance(m, NayheinDecoderLayer),
        )

    return fsdp_model


# ══════════════════════════════════════════════════════════════════════════════
# Save checkpoint
# ══════════════════════════════════════════════════════════════════════════════


def save_checkpoint(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    step: int,
    output_dir: str,
    rank: int,
    is_final: bool = False,
):
    """Save model weights (rank 0 only) and optimizer state."""
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        state_dict = model.state_dict()

    if rank == 0:
        tag = "final" if is_final else f"step-{step}"
        checkpoint_dir = os.path.join(output_dir, tag)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save using safetensors if available
        try:
            from safetensors.torch import save_file

            save_file(state_dict, os.path.join(checkpoint_dir, "model.safetensors"))
        except ImportError:
            torch.save(state_dict, os.path.join(checkpoint_dir, "pytorch_model.bin"))

        # Save optimizer state
        torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
        logger.info(f"[Rank 0] Saved checkpoint to {checkpoint_dir}")

    dist.barrier()


# ══════════════════════════════════════════════════════════════════════════════
# Main training loop
# ══════════════════════════════════════════════════════════════════════════════


def train(args):
    # ── Distributed init ──────────────────────────────────────────────────────
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        logger.info(f"Starting training with {world_size} GPUs")
        logger.info(f"Config: {args.config}")

    cfg = load_config(args.config)

    # ── WandB (rank 0 only) ───────────────────────────────────────────────────
    wandb_run = None
    if rank == 0 and os.environ.get("WANDB_API_KEY"):
        try:
            import wandb

            wandb_run = wandb.init(
                project=cfg.get("wandb_project", "nayhein-v1.2"),
                name=args.run_name,
                config=cfg,
            )
        except ImportError:
            logger.warning("wandb not installed, skipping W&B logging.")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer_path = cfg.get("tokenizer_path", args.output_dir)
    if os.path.exists(os.path.join(tokenizer_path, "tokenizer.json")):
        tokenizer = NayheinTokenizer.from_pretrained(tokenizer_path)
    else:
        logger.warning(
            "No tokenizer found at %s. Using GPT-2 tokenizer as placeholder.",
            tokenizer_path,
        )
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # ── Model ─────────────────────────────────────────────────────────────────
    if args.resume_from:
        model_config = NayheinConfig.from_pretrained(args.resume_from)
        model = NayheinForCausalLM.from_pretrained(
            args.resume_from, config=model_config, torch_dtype=torch.bfloat16
        )
        logger.info(f"Resumed model from {args.resume_from}")
    else:
        model_size = cfg.get("model_size", "50M")
        if "5B" in model_size.upper():
            model_config = NayheinConfig.nayhein_5b()
        else:
            model_config = NayheinConfig.nayhein_50m()
        model = NayheinForCausalLM(model_config)

    # Freeze vision encoder during pretrain
    if model_config.vision_freeze_in_pretrain:
        model.freeze_vision_encoder()

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    if rank == 0:
        logger.info(f"Model parameters: {n_params:.1f}M")

    # ── FSDP ──────────────────────────────────────────────────────────────────
    model = build_fsdp_model(model, cfg)

    # ── torch.compile ─────────────────────────────────────────────────────────
    if cfg.get("compile", True) and hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead")
        if rank == 0:
            logger.info("torch.compile() applied.")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    lr = cfg.get("learning_rate", 3e-3)
    weight_decay = cfg.get("weight_decay", 0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(cfg.get("beta1", 0.9), cfg.get("beta2", 0.95)),
        eps=cfg.get("epsilon", 1e-8),
        weight_decay=weight_decay,
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    seq_len = cfg.get("sequence_length", 4096)
    if rank == 0:
        logger.info("Building streaming dataset...")

    token_stream = build_pretrain_token_stream(tokenizer, max_seq_len=seq_len)
    dataset = PackedSequenceDataset(
        token_stream, max_seq_len=seq_len, eos_token_id=tokenizer.eos_token_id
    )

    batch_size = cfg.get("batch_size_per_device", 16)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=2, pin_memory=True)

    # ── Scheduler ─────────────────────────────────────────────────────────────
    total_tokens = cfg.get("total_tokens", 100_000_000_000)
    tokens_per_step = (
        batch_size * world_size * seq_len * cfg.get("gradient_accumulation_steps", 4)
    )
    total_steps = total_tokens // tokens_per_step
    warmup_steps = cfg.get("warmup_steps", 2000)

    scheduler = get_cosine_with_warmup_scheduler(optimizer, warmup_steps, total_steps)

    # ── Training loop ─────────────────────────────────────────────────────────
    grad_accum = cfg.get("gradient_accumulation_steps", 4)
    max_grad_norm = cfg.get("max_grad_norm", 1.0)
    log_every = cfg.get("log_every_n_steps", 10)
    save_every = cfg.get("save_every_n_steps", 25000)
    diffusion_prob = cfg.get("diffusion_training_probability", 0.5)

    loss_ar_w = cfg.get("loss_ar_weight", 1.0)
    loss_mtp_w = cfg.get("loss_mtp_weight", 0.3)
    loss_diff_w = cfg.get("loss_diffusion_weight", 0.5)

    global_step = 0
    tokens_seen = 0
    t_start = time.time()
    running_loss = 0.0

    os.makedirs(args.output_dir, exist_ok=True)

    if rank == 0:
        logger.info(
            f"Training for {total_steps} steps ({total_tokens / 1e9:.1f}B tokens)"
        )

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        # Randomly apply diffusion loss on 50% of batches
        use_diffusion = torch.rand(1).item() < diffusion_prob

        diffusion_t = None
        diffusion_mask = None
        model_input_ids = input_ids

        if use_diffusion:
            diffusion_t = torch.rand(input_ids.shape[0], device=device)
            model_input_ids, diffusion_mask = model.module.corrupt_sequence(
                input_ids, diffusion_t
            )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=model_input_ids,
                labels=labels,
                attention_mask=attention_mask,
                diffusion_t=diffusion_t,
                diffusion_mask=diffusion_mask,
                loss_ar_weight=loss_ar_w,
                loss_mtp_weight=loss_mtp_w,
                loss_diffusion_weight=loss_diff_w,
                use_diffusion_loss=use_diffusion,
            )
            loss = outputs.loss / grad_accum

        loss.backward()
        running_loss += loss.item() * grad_accum

        if (batch_idx + 1) % grad_accum == 0:
            # Gradient clipping
            if hasattr(model, "clip_grad_norm_"):
                model.clip_grad_norm_(max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            tokens_seen += tokens_per_step

            # Logging
            if rank == 0 and global_step % log_every == 0:
                elapsed = time.time() - t_start
                tokens_per_sec = tokens_seen / elapsed
                lr_current = scheduler.get_last_lr()[0]
                avg_loss = running_loss / log_every
                running_loss = 0.0
                logger.info(
                    f"Step {global_step}/{total_steps} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"LR: {lr_current:.2e} | "
                    f"Tokens: {tokens_seen / 1e9:.2f}B | "
                    f"Throughput: {tokens_per_sec / 1e3:.1f}k tok/s"
                )
                if wandb_run:
                    wandb_run.log(
                        {
                            "train/loss": avg_loss,
                            "train/lr": lr_current,
                            "train/tokens_seen": tokens_seen,
                            "train/tokens_per_sec": tokens_per_sec,
                        },
                        step=global_step,
                    )

            # Save checkpoint
            if global_step % save_every == 0:
                save_checkpoint(model, optimizer, global_step, args.output_dir, rank)

            if global_step >= total_steps:
                break

    # Final save
    if rank == 0:
        logger.info("Training complete. Saving final checkpoint...")
    save_checkpoint(model, optimizer, global_step, args.output_dir, rank, is_final=True)

    # Save config and tokenizer to final dir
    if rank == 0:
        final_dir = os.path.join(args.output_dir, "final")
        model_config.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        logger.info(f"Saved config and tokenizer to {final_dir}")

    if wandb_run and rank == 0:
        wandb_run.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML training config")
    parser.add_argument("--output_dir", required=True, help="Directory for checkpoints")
    parser.add_argument("--run_name", default="nayhein-pretrain", help="W&B run name")
    parser.add_argument("--resume_from", default=None, help="Checkpoint to resume from")
    args = parser.parse_args()
    train(args)
