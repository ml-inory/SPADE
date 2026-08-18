"""SPADE applied to the real CosyVoice 2 (0.5B) LLM-TTS backbone.

This package integrates the SPADE pruning/distillation machinery from
``spade/`` with the official CosyVoice repository and the pretrained
CosyVoice2-0.5B checkpoint. Only the LLM (a 24-layer Qwen2 stack) is
pruned/distilled; the Flow matching model and HiFi-GAN vocoder are reused
unchanged, mirroring the paper's CosyVoice2 experiments.
"""

__version__ = "0.1.0"

