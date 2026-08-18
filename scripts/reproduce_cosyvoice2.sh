#!/usr/bin/env bash
# One-command reproduction of SPADE-on-CosyVoice2 (from zero to results).
#
# Flow:
#   0. environment setup (CosyVoice upstream + deps + checkpoint)
#   1. download LibriSpeech parquet data
#   2. data preparation (embeddings + speech tokens -> CosyVoice parquet)
#   3. WLI (leave-one-out WER over the 24 LLM layers)
#   4. prune 24 -> 12 layers (WLI-driven)
#   5. distillation (one shard per GPU, then checkpoint averaging)
#   6. evaluation (teacher / pruned / distilled)
#
# Env knobs (all optional):
#   COSYVOICE_ROOT     where the CosyVoice repo lives (default ../CosyVoice-main)
#   COSYVOICE2_MODEL_DIR  pretrained checkpoint dir
#   SPADE_WORK_DIR     output dir (default outputs/cosyvoice2/repro)
#   SPADE_GPUS         space-separated GPU ids (default "0 1")
#   SPADE_NUM_UTTS / SPADE_TRAIN_UTTS / SPADE_EVAL_UTTS / SPADE_EPOCHS
#   SPADE_TARGET_LAYERS
#   SPADE_SKIP_WLI=1   reuse an existing wli_report.json instead of re-running
#   SPADE_FAST=1       tiny smoke configuration (minutes instead of hours)
set -euo pipefail

SPADE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COSYVOICE_ROOT="${COSYVOICE_ROOT:-$SPADE_ROOT/../CosyVoice-main}"
MODEL_DIR="${COSYVOICE2_MODEL_DIR:-$COSYVOICE_ROOT/pretrained_models/CosyVoice2-0.5B}"
WORK_DIR="${SPADE_WORK_DIR:-$SPADE_ROOT/outputs/cosyvoice2/repro}"
GPUS=(${SPADE_GPUS:-0 1})
NUM_UTTS="${SPADE_NUM_UTTS:-5500}"
TRAIN_UTTS="${SPADE_TRAIN_UTTS:-5500}"
EVAL_UTTS="${SPADE_EVAL_UTTS:-200}"
EPOCHS="${SPADE_EPOCHS:-7}"
TARGET_LAYERS="${SPADE_TARGET_LAYERS:-12}"
SKIP_WLI="${SPADE_SKIP_WLI:-0}"
FAST="${SPADE_FAST:-0}"
WLI_SUBSET="${SPADE_WLI_SUBSET:-8}"
EVAL_SUBSET="${SPADE_EVAL_SUBSET:-30}"

if [ "$FAST" = "1" ]; then
  NUM_UTTS=300; TRAIN_UTTS=280; EVAL_UTTS=20; EPOCHS=1
  WLI_SUBSET=2; EVAL_SUBSET=6
  echo "### FAST smoke mode: 300 utts, 1 epoch (numbers are NOT the real result)"
fi
N_SHARDS=${#GPUS[@]}

mkdir -p "$WORK_DIR"
log() { echo; echo "### $*"; }

log "0/6 environment setup"
bash "$SPADE_ROOT/spade_cosyvoice2/setup.sh"

log "1/6 download LibriSpeech parquet data"
bash "$SPADE_ROOT/scripts/download_cosyvoice2_data.sh"

log "2/6 data preparation"
cat > "$WORK_DIR/data_prep.yaml" <<EOF
hf_parquet: $COSYVOICE_ROOT/data/train_clean_100/*.parquet
eval_hf_parquet: $COSYVOICE_ROOT/data/librispeech_dev_clean.parquet
num_utts: $NUM_UTTS
train_utts: $TRAIN_UTTS
eval_utts: $EVAL_UTTS
train_shards: $N_SHARDS
num_threads: 12
seed: 0
out_root: $COSYVOICE_ROOT/data/spade_repro
EOF
python -u -m spade_cosyvoice2.data_prep --config "$WORK_DIR/data_prep.yaml"

DATA_META="$COSYVOICE_ROOT/data/spade_repro/data_prep.json"
EVAL_LIST=$(python3 -c "import json;print(json.load(open('$DATA_META'))['eval_list'])")
SHARD_LISTS=($(python3 -c "import json;print(' '.join(json.load(open('$DATA_META'))['train_shard_lists']))"))
WLI_REPORT="$WORK_DIR/wli_report.json"

log "3/6 WER-based layer importance (WLI)"
if [ "$SKIP_WLI" = "1" ] && [ -f "$WLI_REPORT" ]; then
  echo "    reusing $WLI_REPORT (SPADE_SKIP_WLI=1)"
else
  cat > "$WORK_DIR/wli.yaml" <<EOF
eval_list: $EVAL_LIST
subset_size: $WLI_SUBSET
whisper_model: base
device: auto
EOF
  python -u -m spade_cosyvoice2.wli --config "$WORK_DIR/wli.yaml" \
    && cp "$(dirname "$EVAL_LIST")/wli_report.json" "$WLI_REPORT"
fi

log "4/6 prune 24 -> $TARGET_LAYERS layers (WLI-driven)"
RETAINED=$(python3 - "$WLI_REPORT" "$TARGET_LAYERS" <<'PY'
import json, sys
import numpy as np
from spade.pruning import select_layers_to_keep
report = json.load(open(sys.argv[1]))
print(" ".join(map(str, select_layers_to_keep(
    np.array(report["wli"]), int(sys.argv[2]), force_keep_first_last=True))))
PY
)
echo "    retained: $RETAINED"
RETAINED_YAML="[${RETAINED// /, }]"
python -u -m spade_cosyvoice2.prune_llm \
  --llm-pt "$MODEL_DIR/llm.pt" \
  --target-layers "$TARGET_LAYERS" \
  --retained $RETAINED \
  --out "$WORK_DIR/pruned_llm.pt"

log "5/6 distillation ($N_SHARDS shard(s) x $EPOCHS epochs) + averaging"
PIDS=()
for ((s = 0; s < N_SHARDS; s++)); do
  gpu="${GPUS[$s]}"
  cat > "$WORK_DIR/distill_shard$s.yaml" <<EOF
train_list: ${SHARD_LISTS[$s]}
llm_pt: $MODEL_DIR/llm.pt
student_llm_pt: $WORK_DIR/pruned_llm.pt
retained: $RETAINED_YAML
model_dir: $MODEL_DIR
epochs: $EPOCHS
lr: 0.00001
weight_decay: 0.0
grad_clip: 5.0
accum_grad: 2
alpha: 0.25
beta: 0.1
skew_mode: forward
use_ce: true
use_logit: true
use_latent: true
use_attention: true
use_embedding: true
matching: dynamic
device: auto
seed: $s
log_every: 50
save_every: 1000
use_amp: true
resume: true
out_llm_pt: $WORK_DIR/distilled_shard$s.pt
EOF
  echo "    shard $s on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" python -u -m spade_cosyvoice2.distill \
    --config "$WORK_DIR/distill_shard$s.yaml" > "$WORK_DIR/distill_shard$s.log" 2>&1 &
  PIDS+=($!)
done
for pid in "${PIDS[@]}"; do wait "$pid"; done

if [ "$N_SHARDS" -gt 1 ]; then
  CKPTS=()
  for ((s = 0; s < N_SHARDS; s++)); do CKPTS+=("$WORK_DIR/distilled_shard$s.pt"); done
  python -u -m spade_cosyvoice2.average_checkpoints \
    --checkpoints "${CKPTS[@]}" --out "$WORK_DIR/distilled_avg.pt"
  FINAL="$WORK_DIR/distilled_avg.pt"
else
  FINAL="$WORK_DIR/distilled_shard0.pt"
fi

log "6/6 evaluation (teacher / pruned / distilled)"
for name in teacher pruned distilled; do
  case "$name" in
    teacher)   llm="$MODEL_DIR/llm.pt";            retained="[]";;
    pruned)    llm="$WORK_DIR/pruned_llm.pt";      retained="$RETAINED_YAML";;
    distilled) llm="$FINAL";                       retained="$RETAINED_YAML";;
  esac
  cat > "$WORK_DIR/eval_$name.yaml" <<EOF
eval_list: $EVAL_LIST
llm_pt: $llm
retained: $retained
subset_size: $EVAL_SUBSET
whisper_model: base
model_dir: $MODEL_DIR
device: auto
EOF
  python -u -m spade_cosyvoice2.evaluate --config "$WORK_DIR/eval_$name.yaml" \
    --json "$WORK_DIR/eval_$name.json" > /dev/null 2>> "$WORK_DIR/eval_$name.err"
done

echo
echo "================ SUMMARY ================"
python3 - "$WORK_DIR" <<'PY'
import json, sys
work = sys.argv[1]
names = {"teacher": "teacher (24L)", "pruned": "pruned (12L)", "distilled": "distilled (12L)"}
print(f"{'model':<18}{'WER':>8}{'RTF':>8}{'params':>12}")
for name, label in names.items():
    d = json.load(open(f"{work}/eval_{name}.json"))
    print(f"{label:<18}{d['wer']:>8.4f}{d['rtf']:>8.3f}{d['params']:>12,}")
print(f"\nartifacts: {work}")
print("logs:      distill_shard*.log, eval_*.err")
print("NOTE: SPADE_FAST=1 smoke numbers are not meaningful; run without it")
print("      (5500 utts x 7 epochs) for the real result.")
PY
