"""Path helpers for the CosyVoice upstream repo and pretrained checkpoint."""

from __future__ import annotations

import os
import sys
from pathlib import Path


SPADE_ROOT = Path(__file__).resolve().parent.parent


def cosyvoice_root() -> Path:
    """Location of the CosyVoice upstream repo (env override supported)."""
    root = os.environ.get("COSYVOICE_ROOT")
    if root:
        return Path(root)
    candidate = SPADE_ROOT.parent / "CosyVoice-main"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        "CosyVoice upstream not found. Run spade_cosyvoice2/setup.sh or set "
        "COSYVOICE_ROOT."
    )


def model_dir() -> Path:
    """Location of the CosyVoice2-0.5B pretrained checkpoint."""
    model_dir = os.environ.get("COSYVOICE2_MODEL_DIR")
    if model_dir:
        return Path(model_dir)
    return cosyvoice_root() / "pretrained_models" / "CosyVoice2-0.5B"


def ensure_import() -> Path:
    """Put the CosyVoice repo on sys.path and return its root."""
    root = cosyvoice_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    patch_load_wav()
    # Enable CosyVoice's online feature extraction from the checkpoint dir.
    os.environ.setdefault("onnx_path", str(model_dir()))
    return root


def patch_load_wav() -> None:
    """Make CosyVoice's ``load_wav`` decode with soundfile.

    torchaudio 2.9 in this environment fails to load torchcodec (missing
    ``libnvrtc.so.13``), which breaks ``torchaudio.load`` even with
    ``backend='soundfile'``. ``soundfile`` decodes wav/flac without
    torchcodec; ``torchaudio.transforms.Resample`` is unaffected.
    """
    import soundfile as sf
    import torch
    from torchaudio.transforms import Resample

    from cosyvoice.utils import file_utils

    if getattr(file_utils, "_load_wav_patched", False):
        return

    def load_wav(wav, target_sr, min_sr=16000):
        data, sample_rate = sf.read(wav, dtype="float32", always_2d=True)
        speech = torch.from_numpy(data.T)  # (channels, T)
        speech = speech.mean(dim=0, keepdim=True)
        if sample_rate != target_sr:
            assert sample_rate >= min_sr, (
                f"wav sample rate {sample_rate} must be >= {min_sr}"
            )
            speech = Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)
        return speech

    file_utils.load_wav = load_wav
    file_utils._load_wav_patched = True
