# train/sft.py
# Supervised Fine-Tuning (SFT) for Nayhein-V1.2 (Stage 3).
# Supports full fine-tuning (50M) and QLoRA (5B).
# Uses torchrun + FSDP.
#
# Launch (50M):
#   torchrun --nproc_per_node=4 --master_port=29502 train/sft.py \
#       --config configs/sft_50m.yaml --base_model outputs/50m-base/final \
#       --output_dir outputs/50m-sft
#
# Launch (5B QLoRA):
#   torchrun --nproc_per_node=4 --master_port=29503 train/sft.py \
#       --config configs/sft_5b.yaml --base_model outputs/5b-base/final \
#       --output_dir outputs/5b-sft

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Dataset
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from configuration_nayhein import NayheinConfig
from modeling_nayhein import NayheinForCausalLM
from tokenization_nayhein import NayheinTokenizer
from train.data_utils import build_sft_dataset
from train.pretrain import (
    build_fsdp_model,
    get_cosine_with_warmup_scheduler,
    save_checkpoint,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# SFT Dataset wrapper
# ══════════════════════════════════════════════════════════════════════════════


class SFTDataset(Dataset):
    def __init__(self, examples: List[Dict]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def sft_collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


# ══════════════════════════════════════════════════════════════════════════════
# QLoRA setup
# ══════════════════════════════════════════════════════════════════════════════


def apply_qlora(model: NayheinForCausalLM, cfg: Dict) -> NayheinForCausalLM:
    """
    Apply QLoRA (4-bit quantization + LoRA adapters) for memory-efficient 5B SFT.
    """
    try:
        from peft import get_peft_model, LoraConfig, TaskType
        import bitsandbytes as bnb
        from transformers import BitsAndBytesConfig
    except ImportError:
        logger.warning(
            "peft/bitsandbytes not installed. Skipping QLoRA — using full FT."
        )
        return model

    lora_config = LoraConfig(
        r=cfg.get("lora_r", 64),
        lora_alpha=cfg.get("lora_alpha", 128),
        target_modules=cfg.get(
            "lora_target_modules",
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Training function
# ══════════════════════════════════════════════════════════════════════════════


def train_sft(args):
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── WandB ─────────────────────────────────────────────────────────────────
    wandb_run = None
    if rank == 0 and os.environ.get("WANDB_API_KEY"):
        try:
            import wandb

            wandb_run = wandb.init(
                project=cfg.get("wandb_project", "nayhein-v1.2"),
                name=f"sft-{args.output_dir.split('/')[-1]}",
                config=cfg,
            )
        except ImportError:
            pass

    # ── Tokenizer + Model ────────────────────────────────────────────────────
    tokenizer = NayheinTokenizer.from_pretrained(args.base_model)
    model = NayheinForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Unfreeze vision encoder for SFT (Stage 2 of vision training)
    if not cfg.get("vision_encoder_frozen", False):
        model.unfreeze_vision_encoder()

    # Apply QLoRA if configured
    if cfg.get("qlora_enabled", False):
        model = apply_qlora(model, cfg)

    model = model.to(device)

    # FSDP wrapping (skip for QLoRA to avoid PEFT conflicts)
    if not cfg.get("qlora_enabled", False):
        model = build_fsdp_model(model, cfg)

    # ── Dataset ──────────────────────────────────────────────────────────────
    if rank == 0:
        logger.info("Building SFT dataset...")

    examples = build_sft_dataset(
        tokenizer,
        max_seq_len=cfg.get("sequence_length", 4096),
        loss_mask_assistant_only=cfg.get("loss_mask", True),
    )
    dataset = SFTDataset(examples)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(
        dataset,
        batch_size=cfg.get("batch_size_per_device", 4),
        sampler=sampler,
        collate_fn=sft_collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # ── Optimizer + Scheduler ────────────────────────────────────────────────
    lr = cfg.get("learning_rate", 2e-5)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=0.01,
    )

    num_epochs = cfg.get("epochs", 3)
    warmup_ratio = cfg.get("warmup_ratio", 0.03)
    total_steps = (
        len(dataset) // (cfg.get("batch_size_per_device", 4) * world_size)
    ) * num_epochs
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    scheduler = get_cosine_with_warmup_scheduler(optimizer, warmup_steps, total_steps)

    grad_accum = cfg.get("gradient_accumulation_steps", 8)
    os.makedirs(args.output_dir, exist_ok=True)
    global_step = 0

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        model.train()

        for batch_idx, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask,
                    loss_ar_weight=1.0,
                    loss_mtp_weight=0.0,  # MTP disabled during SFT
                    loss_diffusion_weight=0.0,
                )
                loss = outputs.loss / grad_accum

            loss.backward()

            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if rank == 0 and global_step % 10 == 0:
                    lr_current = scheduler.get_last_lr()[0]
                    logger.info(
                        f"Epoch {epoch + 1}/{num_epochs} | Step {global_step} | "
                        f"Loss: {loss.item() * grad_accum:.4f} | LR: {lr_current:.2e}"
                    )
                    if wandb_run:
                        wandb_run.log(
                            {
                                "sft/loss": loss.item() * grad_accum,
                                "sft/lr": lr_current,
                            },
                            step=global_step,
                        )

    # ── Save ──────────────────────────────────────────────────────────────────
    if rank == 0:
        final_dir = os.path.join(args.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)

        # Merge LoRA adapters if QLoRA was used
        if cfg.get("qlora_enabled", False):
            try:
                merged = model.merge_and_unload()
                merged.save_pretrained(final_dir)
            except Exception as e:
                logger.warning(f"Could not merge LoRA: {e}. Saving adapter only.")
                model.save_pretrained(final_dir)
        else:
            save_checkpoint(
                model, optimizer, global_step, args.output_dir, rank, is_final=True
            )
            model.config.save_pretrained(final_dir)

        tokenizer.save_pretrained(final_dir)
        logger.info(f"SFT complete. Saved to {final_dir}")

    if wandb_run and rank == 0:
        wandb_run.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--base_model", required=True, help="Path to pretrained base model"
    )
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    train_sft(args)
