# scripts/upload_to_hf.py
# Upload all 4 Nayhein-V1.2 model checkpoints to HuggingFace Hub.
# Creates repos if they don't exist. Excludes optimizer/scheduler states.
# Generates and pushes README.md for each repo.
#
# Usage:
#   python scripts/upload_to_hf.py \
#       --output_root ./outputs \
#       --hf_token $HF_TOKEN

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Repo definitions ──────────────────────────────────────────────────────────

REPOS = {
    "50m-base": {
        "repo_id": "Nayhein/Nayhein-V1.2-50M-Base",
        "local_dir": "50m-base/final",
        "variant": "Base",
        "size": "50M",
        "pipeline_tag": "text-generation",
        "pretrain_tokens": "100B",
        "is_instruct": False,
        "has_vision": False,
        "base_model_line": "",
    },
    "50m-instruct": {
        "repo_id": "Nayhein/Nayhein-V1.2-50M-Instruct",
        "local_dir": "50m-instruct/final",
        "variant": "Instruct",
        "size": "50M",
        "pipeline_tag": "text-generation",
        "pretrain_tokens": "100B",
        "is_instruct": True,
        "has_vision": True,
        "base_model_line": "base_model: Nayhein/Nayhein-V1.2-50M-Base",
    },
    "5b-base": {
        "repo_id": "Nayhein/Nayhein-V1.2-5B-Base",
        "local_dir": "5b-base/final",
        "variant": "Base",
        "size": "5B",
        "pipeline_tag": "text-generation",
        "pretrain_tokens": "120B",
        "is_instruct": False,
        "has_vision": False,
        "base_model_line": "base_model: Nayhein/Nayhein-V1.2-50M-Base",
    },
    "5b-instruct": {
        "repo_id": "Nayhein/Nayhein-V1.2-5B-Instruct",
        "local_dir": "5b-instruct/final",
        "variant": "Instruct",
        "size": "5B",
        "pipeline_tag": "image-text-to-text",
        "pretrain_tokens": "120B",
        "is_instruct": True,
        "has_vision": True,
        "base_model_line": "base_model: Nayhein/Nayhein-V1.2-5B-Base",
    },
}

# Files to always ignore during upload
IGNORE_PATTERNS = [
    "optimizer.pt",
    "scheduler.pt",
    "*.tmp",
    "*.lock",
    "__pycache__/*",
    "*.pyc",
]

# Source files to bundle in every model repo
MODEL_SOURCE_FILES = [
    "modeling_nayhein.py",
    "configuration_nayhein.py",
    "tokenization_nayhein.py",
    "processing_nayhein.py",
    "generation_utils.py",
    "tool_calling.py",
]


# ══════════════════════════════════════════════════════════════════════════════
# README generation
# ══════════════════════════════════════════════════════════════════════════════


def generate_readme(repo_cfg: Dict) -> str:
    """Generate README.md content for a model repo."""
    size = repo_cfg["size"]
    variant = repo_cfg["variant"]
    pipeline_tag = repo_cfg["pipeline_tag"]
    base_model_line = repo_cfg["base_model_line"]
    pretrain_tokens = repo_cfg["pretrain_tokens"]
    is_instruct = repo_cfg["is_instruct"]
    has_vision = repo_cfg["has_vision"]
    repo_id = repo_cfg["repo_id"]

    if size == "50M":
        param_count = "~50M"
        arch_note = (
            f"Nayhein-V1.2-{size}-{variant} is a compact {param_count} parameter model suited for "
            "edge deployment, research, and rapid prototyping. It features the full NayheinHDT "
            "architecture at a smaller scale, making it ideal for resource-constrained environments."
        )
    else:
        param_count = "~5B"
        arch_note = (
            f"Nayhein-V1.2-{size}-{variant} is a {param_count} parameter model grown from the 50M "
            "checkpoint via structured weight expansion (not trained from scratch). It delivers "
            "significantly higher quality across reasoning, coding, and multimodal tasks."
        )

    if variant == "Instruct":
        variant_note = (
            f"This is the instruction-tuned variant, fine-tuned via SFT on WildChat, LLaVA-Instruct, "
            "UltraChat, and CodeFeedback, then aligned with DPO using UltraFeedback and HH-RLHF. "
            "Recommended for conversational use cases."
        )
    else:
        variant_note = (
            f"This is the base pretrained variant, trained on {pretrain_tokens} tokens of web text, "
            "Wikipedia, code, and math data. Use this checkpoint for fine-tuning or continued pretraining."
        )

    vision_usage = ""
    if has_vision:
        vision_usage = """
## Vision Usage

```python
from PIL import Image
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
image = Image.open("photo.jpg").convert("RGB")

messages = [{"role": "user", "content": [
    {"type": "image"},
    {"type": "text", "text": "What is shown in this image?"}
]}]

inputs = processor(messages, images=image, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```
"""

    tool_usage = ""
    if is_instruct:
        tool_usage = """
## Tool Calling

```python
tools = [{
    "name": "search_web",
    "description": "Search the internet for current information.",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"]
    }
}]

response = model.chat(messages, tools=tools, trust_remote_code=True)
# Returns ToolCallOutput(tool_name="search_web", arguments={"query": "..."})
```
"""

    return f'''---
license: apache-2.0
language:
- en
tags:
- nayhein
- hybrid-diffusion
- autoregressive
- multimodal
- mtp
- tool-calling
- text-generation
{base_model_line}
pipeline_tag: {pipeline_tag}
library_name: transformers
---

<p align="center">
  <a href="https://nayhein.com">
    <img src="https://nayhein.com/logo.png" width="200" alt="Nayhein"/>
  </a>
</p>

<h1 align="center">Nayhein-V1.2-{size}-{variant}</h1>

<p align="center">
  <a href="https://nayhein.com">Website</a> •
  <a href="https://huggingface.co/Nayhein">HuggingFace Org</a>
</p>

---

## Overview

Nayhein-V1.2-{size}-{variant} is part of the **Nayhein-V1.2** model family, a hybrid autoregressive-diffusion language model developed by [Nayhein](https://nayhein.com).

{arch_note}

{variant_note}

## Architecture

Nayhein-V1.2 is built on the **NayheinHDT (Hybrid Diffusion Transformer)** architecture:

- **Shared backbone**: GQA transformer with RMSNorm, SwiGLU, and RoPE + YaRN context extension
- **Autoregressive head**: standard next-token prediction
- **MTP (Multi-Token Prediction)**: 4 future tokens predicted per forward pass for faster inference
- **MDLM diffusion head**: Mercury 2-inspired masked diffusion for parallel decoding
- **Vision encoder**: baked-in SigLIP ViT + Perceiver Resampler (image-text-to-text)
- **Tool calling**: structured JSON decoding with OpenAI-compatible schemas
- **Context window**: 4,096 tokens native, extendable to 32,768 via YaRN RoPE scaling

## Quickstart

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "{repo_id}"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

messages = [
    {{"role": "system", "content": "You are Nayhein, a helpful AI assistant."}},
    {{"role": "user", "content": "Explain diffusion language models in simple terms."}},
]

input_ids = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
).to(model.device)

output = model.generate(
    input_ids,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    generation_mode="ar",   # "ar" | "diffusion" | "hybrid"
)
print(tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True))
```

## Generation Modes

| Mode | `generation_mode` | Description |
|---|---|---|
| Autoregressive | `"ar"` | Standard left-to-right token generation |
| Diffusion | `"diffusion"` | Parallel masked diffusion decoding |
| Hybrid | `"hybrid"` | AR chain-of-thought prefix + diffusion answer suffix |
{vision_usage}{tool_usage}
## Training Details

| Property | Value |
|---|---|
| Pretraining tokens | {pretrain_tokens} |
| Pretraining data | FineWeb-Edu, Wikipedia, StarCoder, Proof-Pile-2 |
| SFT data | WildChat-4.8M, LLaVA-Instruct-150K, UltraChat-200K |
| DPO data | UltraFeedback, HH-RLHF, UltraInteract |
| Training hardware | 4× NVIDIA H100 80GB |
| Training precision | bfloat16 |
| Context length | 4,096 (extendable to 32,768) |
| Optimizer | AdamW (β₁=0.9, β₂=0.95) |

## Benchmarks

| Benchmark | Score |
|---|---|
| HellaSwag | — |
| MMLU | — |
| HumanEval | — |
| MT-Bench | — |

*Benchmarks will be updated after formal evaluation.*

## Limitations

- Research model — outputs may be inaccurate, biased, or hallucinated
- Tool calling works best with clear, well-defined schemas
- Vision performance scales with input resolution; keep images ≥ 256px
- Diffusion mode generation quality may vary; AR mode is most stable

## Citation

```bibtex
@misc{{nayhein2025v12,
  title  = {{Nayhein-V1.2: A Hybrid Autoregressive-Diffusion Language Model}},
  author = {{Nayhein Team}},
  year   = {{2025}},
  url    = {{https://nayhein.com}},
}}
```

## License

Released under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).
© 2025 Nayhein — [https://nayhein.com](https://nayhein.com)
'''


# ══════════════════════════════════════════════════════════════════════════════
# Upload function
# ══════════════════════════════════════════════════════════════════════════════


def upload_all(output_root: str, hf_token: str, push_readmes: bool = True):
    """
    Upload all 4 model checkpoints to HuggingFace Hub.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    project_root = Path(__file__).parent.parent

    results = {}

    for key, repo_cfg in REPOS.items():
        local_dir = os.path.join(output_root, repo_cfg["local_dir"])

        if not os.path.exists(local_dir):
            logger.warning(
                f"Checkpoint not found at {local_dir}. Skipping {repo_cfg['repo_id']}."
            )
            results[key] = {"status": "skipped", "reason": "checkpoint not found"}
            continue

        repo_id = repo_cfg["repo_id"]
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Uploading {repo_id}...")

        # Create repo if needed
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=False,
            exist_ok=True,
        )

        # Copy source files into local_dir so they're uploaded with the model
        for src_file in MODEL_SOURCE_FILES:
            src_path = project_root / src_file
            dst_path = Path(local_dir) / src_file
            if src_path.exists() and not dst_path.exists():
                import shutil

                shutil.copy2(str(src_path), str(dst_path))
                logger.info(f"  Copied {src_file} to checkpoint dir")

        # Generate and write README
        if push_readmes:
            readme_content = generate_readme(repo_cfg)
            readme_path = os.path.join(local_dir, "README.md")
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(readme_content)
            logger.info(f"  Generated README.md")

        # Upload folder
        try:
            api.upload_folder(
                folder_path=local_dir,
                repo_id=repo_id,
                repo_type="model",
                ignore_patterns=IGNORE_PATTERNS,
                commit_message=f"Release {repo_id} — Nayhein-V1.2",
            )
            logger.info(f"  Uploaded {repo_id}")
            logger.info(f"  URL: https://huggingface.co/{repo_id}")
            results[key] = {
                "status": "success",
                "url": f"https://huggingface.co/{repo_id}",
            }
        except Exception as e:
            logger.error(f"  Failed to upload {repo_id}: {e}")
            results[key] = {"status": "failed", "error": str(e)}

    logger.info(f"\n{'=' * 60}")
    logger.info("Upload summary:")
    for key, res in results.items():
        status = res["status"]
        url = res.get("url", res.get("reason", res.get("error", "")))
        logger.info(f"  {REPOS[key]['repo_id']}: {status} — {url}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Upload Nayhein-V1.2 checkpoints to HuggingFace Hub"
    )
    parser.add_argument(
        "--output_root", default="./outputs", help="Root directory of checkpoints"
    )
    parser.add_argument("--hf_token", default=None, help="HuggingFace write token")
    parser.add_argument(
        "--no_readmes", action="store_true", help="Skip README generation"
    )
    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.error("No HuggingFace token provided. Set --hf_token or $HF_TOKEN.")
        sys.exit(1)

    upload_all(args.output_root, hf_token, push_readmes=not args.no_readmes)


if __name__ == "__main__":
    main()
