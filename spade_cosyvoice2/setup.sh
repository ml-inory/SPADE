#!/usr/bin/env bash
# One-shot environment setup for SPADE-on-CosyVoice2.
#
# Prepares:
#   1. the CosyVoice upstream repository (existing $COSYVOICE_ROOT or sibling
#      CosyVoice-main, otherwise downloaded),
#   2. the Python dependencies CosyVoice needs beyond a standard torch stack,
#   3. the CosyVoice2-0.5B pretrained checkpoint (modelscope first, then
#      Hugging Face, honoring http_proxy/https_proxy if set).
set -euo pipefail

SPADE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COSYVOICE_ROOT="${COSYVOICE_ROOT:-$SPADE_ROOT/../CosyVoice-main}"
MODEL_DIR="${COSYVOICE2_MODEL_DIR:-$COSYVOICE_ROOT/pretrained_models/CosyVoice2-0.5B}"

echo "==> [1/3] CosyVoice upstream source"
if [ -d "$COSYVOICE_ROOT/cosyvoice" ]; then
  echo "    using existing CosyVoice at $COSYVOICE_ROOT"
else
  mkdir -p "$(dirname "$COSYVOICE_ROOT")"
  echo "    downloading CosyVoice main tarball"
  curl -sL --max-time 600 -o /tmp/cosyvoice.tar.gz \
    https://github.com/FunAudioLLM/CosyVoice/archive/refs/heads/main.tar.gz
  tar -xzf /tmp/cosyvoice.tar.gz -C "$(dirname "$COSYVOICE_ROOT")"
  if [ "$(basename "$COSYVOICE_ROOT")" != "CosyVoice-main" ]; then
    mv "$(dirname "$COSYVOICE_ROOT")/CosyVoice-main" "$COSYVOICE_ROOT"
  fi
fi

echo "==> [2/3] Python dependencies"
python3 - <<'PY'
import importlib.util as u
missing = [m for m in ("matcha", "wetext") if u.find_spec(m) is None]
if missing:
    print("    installing:", *missing)
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "--no-build-isolation", *missing])
else:
    print("    all CosyVoice inference deps present")
PY

echo "==> [3/3] CosyVoice2-0.5B checkpoint"
if [ -f "$MODEL_DIR/llm.pt" ] && [ -f "$MODEL_DIR/flow.pt" ] && [ -f "$MODEL_DIR/hift.pt" ]; then
  echo "    checkpoint already present at $MODEL_DIR"
else
  mkdir -p "$MODEL_DIR"
  python3 - "$MODEL_DIR" <<'PY'
import sys
from pathlib import Path
target = Path(sys.argv[1])
try:
    from modelscope import snapshot_download
    snapshot_download("iic/CosyVoice2-0.5B", local_dir=str(target))
    print("    downloaded via modelscope")
except Exception as exc:  # fall back to Hugging Face
    print("    modelscope failed (%s), trying Hugging Face" % exc)
    from huggingface_hub import snapshot_download
    snapshot_download("FunAudioLLM/CosyVoice2-0.5B", local_dir=str(target))
    print("    downloaded via Hugging Face")
PY
fi

echo "==> done. Set COSYVOICE_ROOT=$COSYVOICE_ROOT"
echo "    Set COSYVOICE2_MODEL_DIR=$MODEL_DIR"

