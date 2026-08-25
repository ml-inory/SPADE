#!/usr/bin/env python3
"""Ask a vision LLM (e.g. Bailian/DashScope qwen-vl) about image(s).

Credentials are read from $VISION_API_KEY / $VISION_BASE_URL / $VISION_MODEL
or from ~/.config/agent-vision-toolkit/env (used by this workspace).

Examples:
  python scripts/vision_ask.py --image a.png --image b.png \\
      --prompt "请对比这两张频谱图，指出差异与可能的人工痕迹"
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path

import requests


def _load_env(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Vision LLM question answering")
    parser.add_argument("--image", action="append", required=True, help="image path (repeatable)")
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()

    toolkit_env = _load_env(Path.home() / ".config" / "agent-vision-toolkit" / "env")
    api_key = os.environ.get("VISION_API_KEY") or toolkit_env.get("VISION_API_KEY")
    base_url = os.environ.get("VISION_BASE_URL") or toolkit_env.get(
        "VISION_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    model = os.environ.get("VISION_MODEL") or toolkit_env.get("VISION_MODEL", "qwen-vl-max")
    if not api_key:
        raise SystemExit("VISION_API_KEY not found (env or ~/.config/agent-vision-toolkit/env)")

    content: list[dict] = [{"type": "text", "text": args.prompt}]
    for path in args.image:
        b64 = base64.b64encode(Path(path).read_bytes()).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

    resp = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 1200,
        },
        timeout=180,
    )
    resp.raise_for_status()
    data = resp.json()
    print(data["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
