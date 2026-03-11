#!/usr/bin/env bash
# scripts/run_full_pipeline.sh
# Full Nayhein-V1.2 training pipeline: tokenizer → pretrain → expand → SFT → DPO → upload
#
# USAGE:
#   bash scripts/run_full_pipeline.sh \
#       --gpus 4 \
#       --hf_token $HF_TOKEN \
#       --wandb_key $WANDB_API_KEY \
#       --output_root ./outputs
#
# ESTIMATED WALL-CLOCK TIME ON 4×H100 (80GB):
#   Stage 0 (tokenizer):                    ~2 hours
#   Stage 1 (50M pretrain, 100B tokens):   ~18 hours
#   Stage 2 (5B expansion + adapt, 20B):   ~24 hours
#   Stage 3+4 (SFT+DPO, both sizes):       ~12 hours
#   Total:                                 ~56 hours

set -euo pipefail

# ── Argument parsing ──────────────────────────────────────────────────────────
GPUS=4
HF_TOKEN=""
WANDB_KEY=""
OUTPUT_ROOT="./outputs"
SKIP_TOKENIZER=false
SKIP_PRETRAIN_50M=false
SKIP_EXPAND=false
SKIP_PRETRAIN_5B=false
SKIP_SFT=false
SKIP_DPO=false
SKIP_UPLOAD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)           GPUS="$2"; shift 2;;
        --hf_token)       HF_TOKEN="$2"; shift 2;;
        --wandb_key)      WANDB_KEY="$2"; shift 2;;
        --output_root)    OUTPUT_ROOT="$2"; shift 2;;
        --skip_tokenizer) SKIP_TOKENIZER=true; shift;;
        --skip_pretrain_50m) SKIP_PRETRAIN_50M=true; shift;;
        --skip_expand)    SKIP_EXPAND=true; shift;;
        --skip_pretrain_5b) SKIP_PRETRAIN_5B=true; shift;;
        --skip_sft)       SKIP_SFT=true; shift;;
        --skip_dpo)       SKIP_DPO=true; shift;;
        --skip_upload)    SKIP_UPLOAD=true; shift;;
        *)                echo "Unknown argument: $1"; exit 1;;
    esac
done

# ── Environment setup ─────────────────────────────────────────────────────────
export HF_TOKEN="${HF_TOKEN}"
export WANDB_API_KEY="${WANDB_KEY}"
export HF_ORG="Nayhein"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

PIPELINE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PIPELINE_ROOT"

mkdir -p "$OUTPUT_ROOT"
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR"

START_TIME=$(date +%s)

log() {
    local timestamp
    timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    echo "[$timestamp] $*" | tee -a "$LOG_DIR/pipeline.log"
}

log "========================================================"
log "NAYHEIN-V1.2 TRAINING PIPELINE STARTING"
log "GPUs: $GPUS | Output: $OUTPUT_ROOT"
log "========================================================"

# ── Check GPU visibility ─────────────────────────────────────────────────────
log "STEP 0: Verifying GPU availability..."
GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
log "  Detected $GPU_COUNT CUDA GPUs"
if [ "$GPU_COUNT" -lt "$GPUS" ]; then
    log "WARNING: Requested $GPUS GPUs but only $GPU_COUNT available."
    GPUS="$GPU_COUNT"
fi

# ── STEP 1: Train tokenizer ───────────────────────────────────────────────────
TOKENIZER_DIR="$OUTPUT_ROOT/tokenizer"
if [ "$SKIP_TOKENIZER" = false ] && [ ! -f "$TOKENIZER_DIR/tokenizer.json" ]; then
    log "STEP 1: Training tokenizer (65536 vocab, ~10GB corpus)..."
    python3 scripts/train_tokenizer.py \
        --output_dir "$TOKENIZER_DIR" \
        --sample_gb 10 \
        2>&1 | tee "$LOG_DIR/step1_tokenizer.log"
    log "STEP 1: Tokenizer training complete."
else
    log "STEP 1: Tokenizer already exists at $TOKENIZER_DIR. Skipping."
fi

# ── STEP 2: Pretrain 50M Base ─────────────────────────────────────────────────
OUT_50M_BASE="$OUTPUT_ROOT/50m-base"
if [ "$SKIP_PRETRAIN_50M" = false ] && [ ! -f "$OUT_50M_BASE/final/config.json" ]; then
    log "STEP 2: Pretraining 50M Base (100B tokens, ~18 hours on 4×H100)..."
    torchrun \
        --nproc_per_node="$GPUS" \
        --master_port=29500 \
        train/pretrain.py \
        --config configs/nayhein_50m.yaml \
        --output_dir "$OUT_50M_BASE" \
        --run_name nayhein-v1.2-50m-pretrain \
        2>&1 | tee "$LOG_DIR/step2_pretrain_50m.log"
    log "STEP 2: 50M pretraining complete."
else
    log "STEP 2: 50M-Base checkpoint found. Skipping pretraining."
fi

# ── STEP 3: Coherence gate on 50M-Base ────────────────────────────────────────
log "STEP 3: Running coherence gate on 50M-Base..."
GATE_PASSES=0
MAX_RETRIES=3

for attempt in $(seq 1 $MAX_RETRIES); do
    if python3 scripts/coherence_gate.py \
        --model_path "$OUT_50M_BASE/final" \
        --model_type 50m-base \
        2>&1 | tee "$LOG_DIR/step3_gate_50m_base_attempt${attempt}.log"; then
        GATE_PASSES=1
        log "STEP 3: 50M-Base coherence gate PASSED (attempt $attempt)."
        break
    else
        if [ "$attempt" -lt "$MAX_RETRIES" ]; then
            log "STEP 3: Gate FAILED (attempt $attempt). Running 5k more steps..."
            torchrun \
                --nproc_per_node="$GPUS" \
                --master_port=29500 \
                train/pretrain.py \
                --config configs/nayhein_50m.yaml \
                --output_dir "$OUT_50M_BASE" \
                --resume_from "$OUT_50M_BASE/final" \
                --run_name nayhein-v1.2-50m-retry${attempt} \
                2>&1 | tee -a "$LOG_DIR/step3_retry${attempt}.log"
        fi
    fi
done

if [ "$GATE_PASSES" -eq 0 ]; then
    log "ERROR: 50M-Base coherence gate failed after $MAX_RETRIES attempts. HALTING."
    exit 1
fi

# ── STEP 4: Upload 50M-Base ────────────────────────────────────────────────────
if [ "$SKIP_UPLOAD" = false ] && [ -n "$HF_TOKEN" ]; then
    log "STEP 4: Uploading 50M-Base to HuggingFace..."
    python3 scripts/upload_to_hf.py \
        --output_root "$OUTPUT_ROOT" \
        --hf_token "$HF_TOKEN" \
        2>&1 | grep -E "(50m-base|Uploading|Upload)" | tee "$LOG_DIR/step4_upload_50m_base.log"
    log "STEP 4: 50M-Base uploaded."
fi

# ── STEP 5: Expand 50M → 5B ──────────────────────────────────────────────────
OUT_5B_INIT="$OUTPUT_ROOT/5b-init"
if [ "$SKIP_EXPAND" = false ] && [ ! -d "$OUT_5B_INIT" ]; then
    log "STEP 5: Expanding 50M checkpoint to 5B (weight expansion)..."
    python3 scripts/expand_model.py \
        --src_checkpoint "$OUT_50M_BASE/final" \
        --src_config configs/nayhein_50m.yaml \
        --tgt_config configs/nayhein_5b_adapt.yaml \
        --output "$OUT_5B_INIT" \
        --vision_init google/siglip-so400m-patch14-336 \
        2>&1 | tee "$LOG_DIR/step5_expand.log"
    log "STEP 5: Model expansion complete."
else
    log "STEP 5: 5B init checkpoint found. Skipping expansion."
fi

# ── STEP 6: Continued pretrain 5B ────────────────────────────────────────────
OUT_5B_BASE="$OUTPUT_ROOT/5b-base"
if [ "$SKIP_PRETRAIN_5B" = false ] && [ ! -f "$OUT_5B_BASE/final/config.json" ]; then
    log "STEP 6: 5B continued pretraining (20B tokens, ~24 hours on 4×H100)..."
    torchrun \
        --nproc_per_node="$GPUS" \
        --master_port=29501 \
        train/pretrain.py \
        --config configs/nayhein_5b_adapt.yaml \
        --resume_from "$OUT_5B_INIT" \
        --output_dir "$OUT_5B_BASE" \
        --run_name nayhein-v1.2-5b-adapt \
        2>&1 | tee "$LOG_DIR/step6_pretrain_5b.log"
    log "STEP 6: 5B continued pretraining complete."
else
    log "STEP 6: 5B-Base checkpoint found. Skipping."
fi

# ── STEP 7: Coherence gate on 5B-Base ────────────────────────────────────────
log "STEP 7: Running coherence gate on 5B-Base..."
GATE_PASSES=0
for attempt in $(seq 1 $MAX_RETRIES); do
    if python3 scripts/coherence_gate.py \
        --model_path "$OUT_5B_BASE/final" \
        --model_type 5b-base \
        2>&1 | tee "$LOG_DIR/step7_gate_5b_base_attempt${attempt}.log"; then
        GATE_PASSES=1
        log "STEP 7: 5B-Base coherence gate PASSED."
        break
    else
        if [ "$attempt" -lt "$MAX_RETRIES" ]; then
            log "STEP 7: Gate FAILED (attempt $attempt). Continuing training 5k steps..."
            torchrun --nproc_per_node="$GPUS" --master_port=29501 \
                train/pretrain.py --config configs/nayhein_5b_adapt.yaml \
                --resume_from "$OUT_5B_BASE/final" --output_dir "$OUT_5B_BASE" \
                --run_name nayhein-v1.2-5b-retry${attempt} \
                2>&1 | tee -a "$LOG_DIR/step7_retry${attempt}.log"
        fi
    fi
done
if [ "$GATE_PASSES" -eq 0 ]; then
    log "ERROR: 5B-Base coherence gate failed. HALTING."
    exit 1
fi

# Upload 5B-Base
if [ "$SKIP_UPLOAD" = false ] && [ -n "$HF_TOKEN" ]; then
    log "STEP 8: Uploading 5B-Base..."
    python3 scripts/upload_to_hf.py --output_root "$OUTPUT_ROOT" --hf_token "$HF_TOKEN" \
        2>&1 | grep -E "(5b-base|Uploading)" | tee "$LOG_DIR/step8_upload_5b_base.log"
fi

# ── STEP 9+10: SFT + DPO (50M) ───────────────────────────────────────────────
OUT_50M_SFT="$OUTPUT_ROOT/50m-sft"
OUT_50M_DPO="$OUTPUT_ROOT/50m-dpo"

if [ "$SKIP_SFT" = false ] && [ ! -f "$OUT_50M_SFT/final/config.json" ]; then
    log "STEP 9: SFT 50M..."
    torchrun --nproc_per_node="$GPUS" --master_port=29502 \
        train/sft.py --config configs/sft_50m.yaml \
        --base_model "$OUT_50M_BASE/final" --output_dir "$OUT_50M_SFT" \
        2>&1 | tee "$LOG_DIR/step9_sft_50m.log"
fi

if [ "$SKIP_DPO" = false ] && [ ! -f "$OUT_50M_DPO/final/config.json" ]; then
    log "STEP 10: DPO 50M..."
    torchrun --nproc_per_node="$GPUS" --master_port=29504 \
        train/dpo.py --config configs/dpo_50m.yaml \
        --sft_model "$OUT_50M_SFT/final" --output_dir "$OUT_50M_DPO" \
        2>&1 | tee "$LOG_DIR/step10_dpo_50m.log"
fi

# Symlink final 50M-Instruct
mkdir -p "$OUTPUT_ROOT/50m-instruct"
if [ -d "$OUT_50M_DPO/final" ] && [ ! -d "$OUTPUT_ROOT/50m-instruct/final" ]; then
    cp -r "$OUT_50M_DPO/final" "$OUTPUT_ROOT/50m-instruct/final"
fi

# ── STEP 11: Coherence gate on 50M-Instruct ──────────────────────────────────
log "STEP 11: Running coherence gate on 50M-Instruct..."
if ! python3 scripts/coherence_gate.py \
    --model_path "$OUTPUT_ROOT/50m-instruct/final" \
    --model_type 50m-instruct \
    2>&1 | tee "$LOG_DIR/step11_gate_50m_instruct.log"; then

    log "Gate failed. Running 1 extra SFT epoch..."
    torchrun --nproc_per_node="$GPUS" --master_port=29502 \
        train/sft.py --config configs/sft_50m.yaml \
        --base_model "$OUT_50M_SFT/final" --output_dir "$OUT_50M_SFT" \
        2>&1 | tee "$LOG_DIR/step11_sft_retry.log"

    if ! python3 scripts/coherence_gate.py \
        --model_path "$OUTPUT_ROOT/50m-instruct/final" \
        --model_type 50m-instruct 2>&1; then
        log "ERROR: 50M-Instruct gate failed after retry. HALTING."
        exit 1
    fi
fi

if [ "$SKIP_UPLOAD" = false ] && [ -n "$HF_TOKEN" ]; then
    log "STEP 12: Uploading 50M-Instruct..."
    python3 scripts/upload_to_hf.py --output_root "$OUTPUT_ROOT" --hf_token "$HF_TOKEN" \
        2>&1 | grep -E "(50m-instruct|Uploading)" | tee "$LOG_DIR/step12_upload_50m_instruct.log"
fi

# ── STEP 13+14: SFT + DPO (5B) ───────────────────────────────────────────────
OUT_5B_SFT="$OUTPUT_ROOT/5b-sft"
OUT_5B_DPO="$OUTPUT_ROOT/5b-dpo"

if [ "$SKIP_SFT" = false ] && [ ! -f "$OUT_5B_SFT/final/config.json" ]; then
    log "STEP 13: SFT 5B (QLoRA)..."
    torchrun --nproc_per_node="$GPUS" --master_port=29503 \
        train/sft.py --config configs/sft_5b.yaml \
        --base_model "$OUT_5B_BASE/final" --output_dir "$OUT_5B_SFT" \
        2>&1 | tee "$LOG_DIR/step13_sft_5b.log"
fi

if [ "$SKIP_DPO" = false ] && [ ! -f "$OUT_5B_DPO/final/config.json" ]; then
    log "STEP 14: DPO 5B..."
    torchrun --nproc_per_node="$GPUS" --master_port=29505 \
        train/dpo.py --config configs/dpo_5b.yaml \
        --sft_model "$OUT_5B_SFT/final" --output_dir "$OUT_5B_DPO" \
        2>&1 | tee "$LOG_DIR/step14_dpo_5b.log"
fi

mkdir -p "$OUTPUT_ROOT/5b-instruct"
if [ -d "$OUT_5B_DPO/final" ] && [ ! -d "$OUTPUT_ROOT/5b-instruct/final" ]; then
    cp -r "$OUT_5B_DPO/final" "$OUTPUT_ROOT/5b-instruct/final"
fi

# ── STEP 15+16: Gate + Upload 5B-Instruct ────────────────────────────────────
log "STEP 15: Running coherence gate on 5B-Instruct..."
if ! python3 scripts/coherence_gate.py \
    --model_path "$OUTPUT_ROOT/5b-instruct/final" \
    --model_type 5b-instruct \
    2>&1 | tee "$LOG_DIR/step15_gate_5b_instruct.log"; then
    log "Gate failed. Running 1 extra SFT epoch..."
    torchrun --nproc_per_node="$GPUS" --master_port=29503 \
        train/sft.py --config configs/sft_5b.yaml \
        --base_model "$OUT_5B_SFT/final" --output_dir "$OUT_5B_SFT" \
        2>&1 | tee "$LOG_DIR/step15_sft_5b_retry.log"
    if ! python3 scripts/coherence_gate.py \
        --model_path "$OUTPUT_ROOT/5b-instruct/final" \
        --model_type 5b-instruct 2>&1; then
        log "ERROR: 5B-Instruct gate failed. HALTING."
        exit 1
    fi
fi

if [ "$SKIP_UPLOAD" = false ] && [ -n "$HF_TOKEN" ]; then
    log "STEP 16: Uploading all remaining models..."
    python3 scripts/upload_to_hf.py \
        --output_root "$OUTPUT_ROOT" \
        --hf_token "$HF_TOKEN" \
        2>&1 | tee "$LOG_DIR/step16_upload_all.log"
fi

# ── COMPLETION NOTICE ─────────────────────────────────────────────────────────
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
HOURS=$(( ELAPSED / 3600 ))
MINUTES=$(( (ELAPSED % 3600) / 60 ))
SECONDS=$(( ELAPSED % 60 ))
ELAPSED_STR=$(printf "%02d:%02d:%02d" "$HOURS" "$MINUTES" "$SECONDS")

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          ✅  NAYHEIN-V1.2 PIPELINE COMPLETE                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Published Repositories:"
echo "  • Nayhein/Nayhein-V1.2-50M-Base      → https://huggingface.co/Nayhein/Nayhein-V1.2-50M-Base"
echo "  • Nayhein/Nayhein-V1.2-50M-Instruct  → https://huggingface.co/Nayhein/Nayhein-V1.2-50M-Instruct"
echo "  • Nayhein/Nayhein-V1.2-5B-Base       → https://huggingface.co/Nayhein/Nayhein-V1.2-5B-Base"
echo "  • Nayhein/Nayhein-V1.2-5B-Instruct   → https://huggingface.co/Nayhein/Nayhein-V1.2-5B-Instruct"
echo ""
echo "Total training time: $ELAPSED_STR"
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  TO REPRODUCE ON PRIME INTELLECT 4×H100 (80GB), RUN:"
echo "══════════════════════════════════════════════════════════════"
echo ""
echo "  git clone https://github.com/nayhein/nayhein-v1.2"
echo "  cd nayhein-v1.2"
echo "  pip install -r requirements.txt"
echo "  export HF_TOKEN=\"your_hf_write_token\""
echo "  export WANDB_API_KEY=\"your_wandb_key\""
echo "  bash scripts/run_full_pipeline.sh --gpus 4 --output_root ./outputs"
echo ""
echo "  Estimated wall-clock time on 4×H100:"
echo "    Stage 1 (50M pretrain, 100B tokens):    ~18 hours"
echo "    Stage 2 (5B expansion + adapt, 20B):    ~24 hours"
echo "    Stage 3+4 (SFT+DPO, both sizes):        ~12 hours"
echo "    Total:                                  ~54 hours"
echo "══════════════════════════════════════════════════════════════"
