# scripts/train_tokenizer.py
# Train a custom BPE tokenizer for Nayhein-V1.2 with 65,536 vocab size.
# Corpus: FineWeb-Edu + Wikipedia + StarCoder samples (5–10GB total).
# Saves tokenizer files to the specified output directory.
#
# Usage:
#   python scripts/train_tokenizer.py --output_dir tokenizer_output [--sample_gb 10]

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Generator, List

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Special tokens (must be registered in this exact order / IDs)
SPECIAL_TOKENS = [
    "<|pad|>",  # 0
    "<|eos|>",  # 1
    "<|bos|>",  # 2
    "<|unk|>",  # 3
    "<|mask|>",  # 4
    "<|im_start|>",  # 5
    "<|im_end|>",  # 6
    "<|vision_start|>",  # 7
    "<|vision_end|>",  # 8
    "<tool_call>",  # 9
    "</tool_call>",  # 10
    "<tool_result>",  # 11
    "</tool_result>",  # 12
    "<|diffusion|>",  # 13
]

VOCAB_SIZE = 65536
SAMPLE_GB_DEFAULT = 10


def text_iterator(sample_gb: float) -> Generator[str, None, None]:
    """
    Stream text from training corpus datasets.
    Yields individual text strings.
    """
    from datasets import load_dataset

    # Approximate tokens per byte = 0.25 for English text
    # 1GB ≈ 250M tokens. We sample proportionally.
    bytes_per_source = {
        "fineweb-edu": 0.65 * sample_gb * 1e9,
        "wikipedia": 0.15 * sample_gb * 1e9,
        "starcoderdata": 0.10 * sample_gb * 1e9,
        "proof-pile-2": 0.10 * sample_gb * 1e9,
    }

    # ── FineWeb-Edu ───────────────────────────────────────────────────────────
    logger.info("Streaming FineWeb-Edu...")
    bytes_seen = 0
    target = bytes_per_source["fineweb-edu"]
    try:
        ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
        for ex in ds:
            text = ex.get("text", "")
            if text:
                yield text
                bytes_seen += len(text.encode("utf-8"))
                if bytes_seen >= target:
                    break
    except Exception as e:
        logger.warning(f"FineWeb-Edu error: {e}")

    # ── Wikipedia ─────────────────────────────────────────────────────────────
    logger.info("Streaming Wikipedia...")
    bytes_seen = 0
    target = bytes_per_source["wikipedia"]
    try:
        ds = load_dataset(
            "wikimedia/wikipedia", "20231101.en", split="train", streaming=True
        )
        for ex in ds:
            text = ex.get("text", "")
            if text:
                yield text
                bytes_seen += len(text.encode("utf-8"))
                if bytes_seen >= target:
                    break
    except Exception as e:
        logger.warning(f"Wikipedia error: {e}")

    # ── StarCoder ─────────────────────────────────────────────────────────────
    logger.info("Streaming StarCoder (Python/JS/Go)...")
    bytes_seen = 0
    target = bytes_per_source["starcoderdata"]
    try:
        for lang in ["python", "javascript", "go"]:
            ds = load_dataset(
                "bigcode/starcoderdata",
                data_dir=lang,
                split="train",
                streaming=True,
            )
            for ex in ds:
                text = ex.get("content", "")
                if text:
                    yield text
                    bytes_seen += len(text.encode("utf-8"))
                    if bytes_seen >= target:
                        break
            if bytes_seen >= target:
                break
    except Exception as e:
        logger.warning(f"StarCoder error: {e}")

    # ── Proof-Pile-2 ──────────────────────────────────────────────────────────
    logger.info("Streaming Proof-Pile-2...")
    bytes_seen = 0
    target = bytes_per_source["proof-pile-2"]
    try:
        ds = load_dataset("EleutherAI/proof-pile-2", split="train", streaming=True)
        for ex in ds:
            text = ex.get("text", "")
            if text:
                yield text
                bytes_seen += len(text.encode("utf-8"))
                if bytes_seen >= target:
                    break
    except Exception as e:
        logger.warning(f"Proof-Pile-2 error: {e}")


def train_tokenizer(output_dir: str, sample_gb: float = SAMPLE_GB_DEFAULT):
    """
    Train BPE tokenizer and save all tokenizer files.

    Tokenizer design:
    - ByteLevel pre-tokenizer (byte fallback enabled)
    - NFC unicode normalization
    - BPE with 65,536 vocab (including 14 special tokens)
    - min_frequency=2
    """
    try:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.normalizers import NFC
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        from transformers import PreTrainedTokenizerFast
    except ImportError:
        logger.error("tokenizers library not installed. Run: pip install tokenizers")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Build base tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
    tokenizer.normalizer = NFC()
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    logger.info(f"Training tokenizer on ~{sample_gb}GB of text...")
    tokenizer.train_from_iterator(
        text_iterator(sample_gb),
        trainer=trainer,
    )

    # Verify special token IDs are correct
    for token, expected_id in zip(SPECIAL_TOKENS, range(len(SPECIAL_TOKENS))):
        actual_id = tokenizer.token_to_id(token)
        if actual_id != expected_id:
            logger.warning(
                f"Special token ID mismatch: {token} has ID {actual_id}, expected {expected_id}"
            )

    # Save raw tokenizer.json
    raw_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(raw_path)
    logger.info(f"Saved raw tokenizer to {raw_path}")

    # Wrap in PreTrainedTokenizerFast
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=raw_path,
        bos_token="<|bos|>",
        eos_token="<|eos|>",
        unk_token="<|unk|>",
        pad_token="<|pad|>",
        mask_token="<|mask|>",
        model_max_length=32768,
    )

    # Set ChatML chat template
    fast_tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n' }}"
        "{% endif %}"
    )

    fast_tokenizer.save_pretrained(output_dir)
    logger.info(f"Saved tokenizer to {output_dir}")
    logger.info(f"Vocab size: {fast_tokenizer.vocab_size}")
    logger.info(f"Special tokens: {fast_tokenizer.special_tokens_map}")

    # Quick sanity check
    test_text = "Hello, world! <|im_start|>user\nWhat is AI?<|im_end|>"
    tokens = fast_tokenizer.encode(test_text)
    logger.info(f"Sanity check: '{test_text[:30]}...' → {len(tokens)} tokens")

    return fast_tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Train BPE tokenizer for Nayhein-V1.2 (65,536 vocab)"
    )
    parser.add_argument(
        "--output_dir", required=True, help="Directory to save tokenizer files"
    )
    parser.add_argument(
        "--sample_gb",
        type=float,
        default=SAMPLE_GB_DEFAULT,
        help="Approximate GB of text to sample for training (default: 10)",
    )
    args = parser.parse_args()
    train_tokenizer(args.output_dir, args.sample_gb)


if __name__ == "__main__":
    main()
