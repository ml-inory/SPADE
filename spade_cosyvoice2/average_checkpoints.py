"""Average multiple CosyVoice2 LLM checkpoints (model averaging)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def average_checkpoints(checkpoints: list[str], out_path: str | Path) -> None:
    sds = [torch.load(p, map_location="cpu", weights_only=True) for p in checkpoints]
    keys = set(sds[0])
    for sd in sds[1:]:
        if set(sd) != keys:
            raise ValueError("checkpoints have mismatched state-dict keys")
    n = len(sds)
    averaged = {k: sum(sd[k] for sd in sds) / n for k in keys}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(averaged, out_path)
    print(f"[average] {len(checkpoints)} checkpoints -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Average CosyVoice2 LLM checkpoints")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    average_checkpoints(args.checkpoints, args.out)


if __name__ == "__main__":
    main()

