# eval/perplexity_eval.py
# Standalone perplexity evaluation script for Nayhein-V1.2 models.
# Evaluates on Wikipedia samples to compute perplexity gate metric.
#
# Usage:
#   python eval/perplexity_eval.py \
#       --model_path outputs/50m-base/final \
#       --n_samples 5000

import argparse
import logging
import math
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def compute_perplexity(
    model_path: str,
    dataset: str = "wikimedia/wikipedia",
    dataset_config: str = "20231101.en",
    n_samples: int = 5000,
    max_length: int = 512,
    device: str = "auto",
) -> float:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    logger.info(f"Loading {n_samples} Wikipedia samples...")
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset, dataset_config, split="train", streaming=True)
    except Exception as e:
        logger.error(f"Could not load dataset: {e}")
        return 999.0

    total_nll = 0.0
    total_tokens = 0
    count = 0

    with torch.inference_mode():
        for example in ds:
            if count >= n_samples:
                break
            text = example.get("text", "")
            if not text.strip():
                continue

            input_ids = tokenizer.encode(
                text, return_tensors="pt", truncation=True, max_length=max_length
            ).to(next(model.parameters()).device)

            if input_ids.shape[-1] < 2:
                continue

            outputs = model(input_ids=input_ids, labels=input_ids)
            if outputs.loss is None:
                continue

            n_toks = input_ids.shape[-1]
            total_nll += outputs.loss.item() * n_toks
            total_tokens += n_toks
            count += 1

            if count % 500 == 0:
                running_ppl = math.exp(total_nll / total_tokens)
                logger.info(f"  [{count}/{n_samples}] Running PPL: {running_ppl:.2f}")

    if total_tokens == 0:
        return 999.0

    avg_nll = total_nll / total_tokens
    perplexity = math.exp(avg_nll)
    logger.info(f"Final perplexity on {count} samples: {perplexity:.2f}")
    return perplexity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    ppl = compute_perplexity(
        args.model_path,
        n_samples=args.n_samples,
        max_length=args.max_length,
        device=args.device,
    )
    print(f"Perplexity: {ppl:.4f}")


if __name__ == "__main__":
    main()
