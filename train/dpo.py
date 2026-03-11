# train/dpo.py
# Direct Preference Optimization (DPO) for Nayhein-V1.2 (Stage 4).
# Uses trl's DPOTrainer wrapped with FSDP-aware settings.
#
# Launch (50M):
#   torchrun --nproc_per_node=4 --master_port=29504 train/dpo.py \
#       --config configs/dpo_50m.yaml --sft_model outputs/50m-sft/final \
#       --output_dir outputs/50m-dpo
#
# Launch (5B):
#   torchrun --nproc_per_node=4 --master_port=29505 train/dpo.py \
#       --config configs/dpo_5b.yaml --sft_model outputs/5b-sft/final \
#       --output_dir outputs/5b-dpo

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.distributed as dist
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from configuration_nayhein import NayheinConfig
from modeling_nayhein import NayheinForCausalLM
from tokenization_nayhein import NayheinTokenizer
from train.data_utils import build_dpo_dataset, format_conversation_to_chatml

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# DPO Training function
# ══════════════════════════════════════════════════════════════════════════════


def train_dpo(args):
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── WandB ─────────────────────────────────────────────────────────────────
    if rank == 0 and os.environ.get("WANDB_API_KEY"):
        try:
            import wandb

            wandb.init(
                project=cfg.get("wandb_project", "nayhein-v1.2"),
                name=f"dpo-{args.output_dir.split('/')[-1]}",
                config=cfg,
            )
        except ImportError:
            pass

    # ── Load model ────────────────────────────────────────────────────────────
    tokenizer = NayheinTokenizer.from_pretrained(args.sft_model)
    model = NayheinForCausalLM.from_pretrained(
        args.sft_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = model.to(device)

    # Reference model (frozen SFT checkpoint)
    ref_model = NayheinForCausalLM.from_pretrained(
        args.sft_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    ref_model = ref_model.to(device)
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.eval()

    # ── Build DPO dataset ─────────────────────────────────────────────────────
    if rank == 0:
        logger.info("Building DPO dataset...")
    dpo_pairs = build_dpo_dataset()

    # Convert to HuggingFace Dataset for trl DPOTrainer
    try:
        from datasets import Dataset as HFDataset
        from trl import DPOConfig, DPOTrainer

        hf_data = {
            "prompt": [p["prompt"] for p in dpo_pairs],
            "chosen": [p["chosen"] for p in dpo_pairs],
            "rejected": [p["rejected"] for p in dpo_pairs],
        }
        hf_dataset = HFDataset.from_dict(hf_data)

        dpo_config = DPOConfig(
            beta=cfg.get("beta", 0.1),
            learning_rate=cfg.get("learning_rate", 5e-7),
            num_train_epochs=cfg.get("epochs", 1),
            per_device_train_batch_size=cfg.get("batch_size_per_device", 2),
            gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 16),
            max_length=cfg.get("sequence_length", 4096),
            output_dir=args.output_dir,
            bf16=True,
            logging_steps=10,
            save_steps=500,
            report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        )

        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=dpo_config,
            train_dataset=hf_dataset,
            tokenizer=tokenizer,
        )
        trainer.train()

        # Save final
        if rank == 0:
            final_dir = os.path.join(args.output_dir, "final")
            trainer.save_model(final_dir)
            tokenizer.save_pretrained(final_dir)
            model.config.save_pretrained(final_dir)
            logger.info(f"DPO complete. Saved to {final_dir}")

    except ImportError as e:
        # Fallback: manual DPO loss implementation
        logger.warning(f"trl not available ({e}). Running manual DPO.")
        _manual_dpo_loop(
            model, ref_model, tokenizer, dpo_pairs, cfg, args, device, rank, world_size
        )

    dist.destroy_process_group()


def _manual_dpo_loop(
    model, ref_model, tokenizer, pairs, cfg, args, device, rank, world_size
):
    """Minimal manual DPO training loop if trl is unavailable."""
    beta = cfg.get("beta", 0.1)
    lr = cfg.get("learning_rate", 5e-7)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    seq_len = cfg.get("sequence_length", 4096)
    grad_accum = cfg.get("gradient_accumulation_steps", 16)
    total_steps = len(pairs) // (cfg.get("batch_size_per_device", 2) * world_size)

    model.train()
    optimizer.zero_grad()

    for step, pair in enumerate(pairs):
        if step >= total_steps:
            break

        prompt = pair["prompt"]
        chosen_text = format_conversation_to_chatml(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": pair["chosen"]},
            ]
        )
        rejected_text = format_conversation_to_chatml(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": pair["rejected"]},
            ]
        )

        chosen_ids = tokenizer.encode(
            chosen_text, return_tensors="pt", truncation=True, max_length=seq_len
        ).to(device)
        rejected_ids = tokenizer.encode(
            rejected_text, return_tensors="pt", truncation=True, max_length=seq_len
        ).to(device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            # Policy log probs
            chosen_out = model(input_ids=chosen_ids, labels=chosen_ids)
            rejected_out = model(input_ids=rejected_ids, labels=rejected_ids)

            # Reference log probs
            with torch.no_grad():
                ref_chosen_out = ref_model(input_ids=chosen_ids, labels=chosen_ids)
                ref_rejected_out = ref_model(
                    input_ids=rejected_ids, labels=rejected_ids
                )

            # DPO loss: -log(σ(β * (log π(yw) - log π(yl) - log π_ref(yw) + log π_ref(yl))))
            log_ratio_chosen = -chosen_out.loss + ref_chosen_out.loss
            log_ratio_rejected = -rejected_out.loss + ref_rejected_out.loss
            dpo_loss = (
                -torch.nn.functional.logsigmoid(
                    beta * (log_ratio_chosen - log_ratio_rejected)
                )
                / grad_accum
            )

        dpo_loss.backward()

        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            if rank == 0 and (step // grad_accum) % 10 == 0:
                logger.info(
                    f"DPO Step {step // grad_accum} | Loss: {dpo_loss.item() * grad_accum:.4f}"
                )

    if rank == 0:
        final_dir = os.path.join(args.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        model.config.save_pretrained(final_dir)
        logger.info(f"Manual DPO complete. Saved to {final_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sft_model", required=True, help="Path to SFT checkpoint")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    train_dpo(args)
