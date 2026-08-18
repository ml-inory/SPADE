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
    patch_torchaudio_load()
    # Enable CosyVoice's online feature extraction from the checkpoint dir.
    os.environ.setdefault("onnx_path", str(model_dir()))
    return root


def patch_torchaudio_load() -> None:
    """Make ``torchaudio.load`` decode with soundfile.

    torchaudio 2.9 in this environment fails to load torchcodec (missing
    ``libnvrtc.so.13``), which breaks ``torchaudio.load`` even with
    ``backend='soundfile'``. This patch routes every ``torchaudio.load``
    call (file paths and ``BytesIO``) through ``soundfile``, which decodes
    wav/flac/ogg without torchcodec; ``torchaudio.transforms.Resample`` is
    unaffected.
    """
    import soundfile as sf
    import torch
    import torchaudio

    if getattr(torchaudio, "_spade_load_patched", False):
        return

    def load(
        uri,
        frame_offset=0,
        num_frames=-1,
        normalize=True,
        channels_first=True,
        format=None,
        buffer_size=4096,
        backend=None,
    ):
        data, sample_rate = sf.read(uri, dtype="float32", always_2d=True)
        speech = torch.from_numpy(data.T)  # (channels, T)
        if num_frames is not None and num_frames > 0:
            speech = speech[:, frame_offset : frame_offset + num_frames]
        elif frame_offset:
            speech = speech[:, frame_offset:]
        if not channels_first:
            speech = speech.transpose(0, 1)
        return speech, sample_rate

    torchaudio.load = load
    torchaudio._spade_load_patched = True
