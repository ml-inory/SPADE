"""Smoke test: load CosyVoice2 and synthesize one zero-shot utterance."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="CosyVoice2 smoke inference")
    parser.add_argument("--out", default="outputs/cosyvoice2/smoke.wav")
    parser.add_argument("--tts-text", default="This is a smoke test of the SPADE CosyVoice two integration.")
    parser.add_argument("--prompt-text", default="希望你以后能够做的比我还好呦。")
    args = parser.parse_args()

    from spade_cosyvoice2.paths import ensure_import, model_dir

    ensure_import()
    import soundfile as sf
    from cosyvoice.cli.cosyvoice import CosyVoice2

    cosyvoice = CosyVoice2(str(model_dir()))
    prompt_wav = str(ensure_import() / "asset" / "zero_shot_prompt.wav")
    if not Path(prompt_wav).exists():
        prompt_wav = str(Path(__file__).resolve().parent / "asset" / "zero_shot_prompt.wav")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for output in cosyvoice.inference_zero_shot(
        args.tts_text,
        args.prompt_text,
        prompt_wav,
        stream=False,
    ):
        speech = output["tts_speech"].squeeze(0).cpu().numpy()
        sf.write(str(out), speech, cosyvoice.sample_rate)
        print(f"saved {out} ({speech.shape[0] / cosyvoice.sample_rate:.2f}s)")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
