#!/usr/bin/env bash
# Download the LibriSpeech parquet data used by the CosyVoice2 reproduction:
#   - validation.clean  (dev-clean, 342 MB) -> eval set
#   - train.clean.100 shards 0000-0002     -> train set (~6000 utterances)
# Files are stored under $COSYVOICE_ROOT/data and are resumable (-C -).
set -euo pipefail

SPADE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COSYVOICE_ROOT="${COSYVOICE_ROOT:-$SPADE_ROOT/../CosyVoice-main}"
DATA_DIR="$COSYVOICE_ROOT/data"

# Honor proxy env vars (e.g. http_proxy/https_proxy) if set.
proxy_arg=()
if [ -n "${HTTPS_PROXY:-}${https_proxy:-}" ]; then
  proxy_arg=(-x "${HTTPS_PROXY:-$https_proxy}")
fi

dl() {
  local url="$1" out="$2" expect="$3"
  mkdir -p "$(dirname "$out")"
  if [ -f "$out" ] && [ "$(stat -c%s "$out")" -ge "$expect" ]; then
    echo "    already present: $out"
    return
  fi
  echo "    downloading $out"
  curl -sL -C - --retry 5 --retry-delay 3 "${proxy_arg[@]}" -o "$out" "$url"
  echo "    done: $out ($(stat -c%s "$out") bytes)"
}

BASE="https://huggingface.co/datasets/openslr/librispeech_asr/resolve/main/all"

echo "==> [1/3] LibriSpeech dev-clean (eval)"
dl "$BASE/validation.clean/0000.parquet" "$DATA_DIR/librispeech_dev_clean.parquet" 300000000

echo "==> [2/3] LibriSpeech train-clean-100 shards 0000-0002 (train)"
for i in 0000 0001 0002; do
  dl "$BASE/train.clean.100/$i.parquet" "$DATA_DIR/train_clean_100/$i.parquet" 400000000
done

echo "==> [3/3] done. Data is under $DATA_DIR"
