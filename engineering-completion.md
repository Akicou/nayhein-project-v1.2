══════════════════════════════════════════════════════════════
  TO REPRODUCE ON PRIME INTELLECT 4×H100 (80GB), RUN:
══════════════════════════════════════════════════════════════
  git clone https://github.com/nayhein/nayhein-v1.2
  cd nayhein-v1.2
  pip install -r requirements.txt
  export HF_TOKEN="your_hf_write_token"
  export WANDB_API_KEY="your_wandb_key"
  bash scripts/run_full_pipeline.sh --gpus 4 --output_root ./outputs
  Estimated wall-clock time on 4×H100:
    Stage 1 (50M pretrain, 100B tokens):    ~18 hours
    Stage 2 (5B expansion + adapt, 20B):    ~24 hours
    Stage 3+4 (SFT+DPO, both sizes):        ~12 hours
    Total:                                  ~54 hours
══════════════════════════════════════════════════════════════
---
The complete Nayhein-V1.2 codebase has been written to nayhein-v1.2/. Here is a summary of every file and what it implements:
What was built
Core model files (uploaded to every HF repo)
File	Contents
configuration_nayhein.py	NayheinConfig with nayhein_50m() and nayhein_5b() class methods
modeling_nayhein.py	Full NayheinForCausalLM: GQA+RoPE+YaRN backbone, SwiGLU MLP, SigLIP ViT + Perceiver Resampler, MTP heads (DeepSeek-V3 style), MDLM diffusion (Mercury 2 style), Flash Attention 2 / SDPA fallback
tokenization_nayhein.py	NayheinTokenizer with 14 special tokens, ChatML template, vision/tool tokens
processing_nayhein.py	NayheinProcessor: multi-modal text+image preprocessing
generation_utils.py	NayheinGenerationMixin: AR, MDLM diffusion, hybrid (AR prefix + diff suffix) generation
tool_calling.py	NayheinToolCallingMixin: OpenAI-compatible tool schemas → XML injection, <tool_call> parsing, optional outlines constrained decoding
Training pipeline
File	Contents
train/data_utils.py	Streaming dataset loading, WildChat filtering, sequence packing, SFT/DPO dataset builders
train/pretrain.py	FSDP + torchrun pretraining with combined AR+MTP+MDLM loss
train/sft.py	SFT with assistant-token loss masking; QLoRA for 5B
train/dpo.py	DPO via trl with manual fallback
Scripts
File	Contents
scripts/train_tokenizer.py	BPE tokenizer training on FineWeb-Edu + Wikipedia + StarCoder
scripts/expand_model.py	50M→5B: width expansion (zero-pad attn, noise-init FFN), depth expansion (SOLAR-style interpolation), SigLIP vision init
scripts/coherence_gate.py	Full test battery (factual, instruct, multi-turn, degeneration, tool, vision), perplexity gate, GPT-4 judge with heuristic fallback
scripts/upload_to_hf.py	Creates HF repos, bundles source files, generates README.md per repo, uploads
scripts/run_full_pipeline.sh	End-to-end orchestration with retry logic (max 3×) on gate failures
Configs
All 6 YAML configs: nayhein_50m.yaml, nayhein_5b_adapt.yaml, sft_50m.yaml, sft_5b.yaml, dpo_50m.yaml, dpo_5b.yaml
