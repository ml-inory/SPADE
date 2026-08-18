# SPADE: Structured Pruning and Adaptive Distillation for Efficient LLM-TTS

SPADE is a framework for compressing LLM-based text-to-speech (LLM-TTS)
models. It removes non-essential Transformer layers using a **word-error-rate
based layer importance** (WLI) criterion, then recovers quality with
**adaptive multi-level knowledge distillation** against the original
un-pruned model as the teacher. This repository is a faithful, self-contained
implementation of the method from:

> Tan Dat Nguyen, Jaehun Kim, Ji-Hoon Kim, Shukjae Choi, Youshin Lim, Joon Son
> Chung, "SPADE: Structured Pruning and Adaptive Distillation for Efficient
> LLM-TTS", ICASSP 2026 ([arXiv:2509.20802](https://arxiv.org/abs/2509.20802)).

## Why SPADE

Recent LLM-TTS systems (CosyVoice, VALL-E, LLaSA, ...) give strong
controllability and zero-shot generalization, but inherit the cost of
large language models: many parameters, high memory, slow autoregressive
decoding. The paper shows that many Transformer layers contribute little to
intelligibility and can be removed, and that a compact student can regain
the teacher's quality using only a fraction of the original training data.

## How it works

### 1. Pruning: WER-based layer importance (WLI)

Transformer layers refine latents through residual connections
(`x^l = x^{l-1} + f_l(x^{l-1})`), so a layer can be removed by simply
skipping `f_l`. For every layer `i`, SPADE measures what happens to
intelligibility when that layer is removed:

```
WLI_i = E_D[ WER( model(x_2; θ∖i, x_1, y_1), y_2 ) ]
```

where `θ∖i` is the model without layer `i`, `(x_1, y_1)` is a reference
text/audio prompt, `(x_2, y_2)` is a query pair, and WER is computed against
the reference transcript (Whisper for real speech). Layers with low WLI
(removing them barely hurts) are pruned. The repo also implements the
cosine-based layer importance (`CLI = 1 - mean cos(x^{l-1}, x^l)`) used in
the paper as the baseline that does **not** align with true contribution.

The pruned student is initialized by copying the retained layers'
parameters from the teacher.

### 2. Recovery: adaptive multi-level distillation

The original un-pruned model acts as a frozen teacher while the student is
fine-tuned with the paper's composite loss:

```
L = α·L_CE + (1-α)/4 · (L_logit + L_l + L_a + L_e),   α = 0.25
```

* `L_CE` — supervised cross-entropy on the speech-code tokens.
* `L_logit` — **Skew KL divergence** (DistiLLM,
  [arXiv:2402.03898](https://arxiv.org/abs/2402.03898)) between teacher and
  student output distributions, `KL(p || βp + (1-β)q)` with `β = 0.1`
  (forward mode; reverse mode also provided).
* `L_l`, `L_a`, `L_e` — MSE on intermediate **latents**, **attention maps**,
  and **embedding outputs**.

### 3. Dynamic layer matching

Instead of matching each student layer to its own retained teacher layer,
student layer `n` (copied from retained teacher layer `r_n`) is aligned to
the *last teacher layer before the next retained layer*, i.e.
`target(n) = r_{n+1} - 1` (the final student layer targets the final teacher
layer). This makes the student absorb the transformations of every removed
layer in between. The static matching (target = own retained layer) is
provided as the ablation baseline.

## Repository layout

```
spade/
├── models/          # GPT-style LLM-TTS backbone (causal attn, residual blocks, LM heads)
├── data/            # char tokenizer + invertible synthetic speech codec + dataset
├── pruning/         # WLI / CLI importance, layer selection, student construction
├── distill/         # Skew-KL + MSE losses, dynamic layer matching, distillation trainer
├── evaluation/      # WER scorers (token-codec and Whisper) + efficiency metrics
├── train.py         # stage CLI: train teacher
├── prune.py         # stage CLI: WLI + prune
├── distill_cli.py   # stage CLI: distill student
├── evaluate.py      # stage CLI: WER / params / depth / RTF
└── pipeline.py      # one-command end-to-end run
configs/             # YAML configs (teacher / wli / distill / smoke)
tests/               # pytest suite incl. end-to-end smoke test
```

## Installation

```bash
pip install -e .          # core deps: torch, numpy, pyyaml, tqdm
pip install -e ".[wer]"   # optional: Whisper-based WER for real audio
```

## Quickstart

The repo ships with a small **synthetic speech-codec benchmark**: text is
mapped deterministically to discrete code tokens (speaker-,
position-, and context-dependent, invertible), so the full loop -- train
teacher → compute WLI → prune → distill → evaluate -- runs end-to-end in a
few minutes without downloading a speech corpus. WER is computed with a
Viterbi maximum-consistency decoder (analogous to ASR decoding with language
constraints), so isolated code errors do not cascade and WER is a meaningful
intelligibility measure.

```bash
# End-to-end (tiny config, ~1 minute):
python -m spade.pipeline --config configs/smoke.yaml

# End-to-end (medium demo, ~5-6 minutes on an L4 GPU) -- shows the full
# SPADE story: teacher WER ~0.27 -> pruned (half depth) WER ~0.78 ->
# distilled WER ~0.15, with depth x2 / params x1.9 / RTF x1.9:
python -m spade.pipeline --config configs/demo.yaml

# Or run the stages individually:
python -m spade.train --config configs/teacher.yaml --out outputs/teacher.pt
python -m spade.prune --teacher outputs/teacher.pt \
    --config configs/wli.yaml --out outputs
python -m spade.distill_cli --teacher outputs/teacher.pt \
    --student outputs/student_pruned.pt \
    --config configs/distill.yaml --out outputs/distilled.pt
python -m spade.evaluate --model outputs/distilled.pt \
    --config configs/wli.yaml --json outputs/distilled_metrics.json
```

Every pipeline run writes `outputs/pipeline_report.json` with per-layer WLI
and CLI, retained indices, WER before/after distillation, parameter/depth
counts, and generation latency / synthetic real-time factor. On the synthetic
benchmark the expected WLI pattern matches the paper: the earliest and final
layers score as most important, and the dynamic layer matching lets the
student absorb the pruned segment (e.g. student layer 2 targeting teacher
layer 6 when layers 2-6 are removed).

## Configuration

All stages are driven by YAML (see `configs/`). Key knobs:

* `model.n_layer`, `n_embd`, `n_head`, `vocab_size` — backbone shape.
* `data.num_samples`, `num_eval_samples`, `num_speakers`, `max_words` —
  synthetic benchmark size (substitute your own dataset for real speech).
* `prune.target_layers` — student depth; `force_keep_first_last` keeps
  boundary layers; `eval_subset_size` controls WLI cost.
* `distill.alpha` (0.25), `distill.beta` (0.1), `distill.skew_mode`,
  `distill.matching` (`dynamic` / `static`), and per-term toggles
  (`use_ce`, `use_logit`, `use_latent`, `use_attention`, `use_embedding`)
  for ablations.

## Using real models (CosyVoice 2 / LLaSA)

The framework is backbone-agnostic. To run the paper's setups:

1. **Data**: produce discrete speech-code tokens (e.g. EnCodec/SNAC) plus
   transcripts and store them as `(input_ids, labels, cond_len)` samples, or
   use the synthetic dataset as a template for a custom `Dataset`.
2. **WER**: for real audio, implement a codec decoder
   `codes -> waveform` and use `WhisperWERScorer`
   (`spade/evaluation/scorers.py`), which transcribes generated audio and
   computes WER against the transcript.
3. **Backbones**: `prune_layers` / the distillation trainer operate on any
   residual Transformer exposing per-block hidden states and attention maps
   with the same interface as `spade.models.LLMTTSBackbone`; adapters for
   Hugging Face-style `ModuleList` stacks are a drop-in extension point.

## Testing

```bash
python -m pytest -x -q
```

The suite covers the backbone, codec round-trip, WLI/CLI pruning, layer
matching, every distillation loss term, trainer correctness (teacher frozen,
student updated), and an end-to-end smoke pipeline.

## Limitations

* The built-in benchmark uses a synthetic codec, not real speech; WER there
  is an exact token-level proxy. Real-data evaluation requires the
  Whisper/EnCodec adapters and an external corpus.
* WLI requires autoregressive decoding per layer; use a small
  `eval_subset_size` to keep it cheap.
* This is a research implementation: pruning + distillation of full-scale
  CosyVoice 2 / LLaSA models needs multi-GPU training and their corpora,
  which are out of scope here.

## References

1. Nguyen et al., *SPADE: Structured Pruning and Adaptive Distillation for
   Efficient LLM-TTS*, ICASSP 2026, arXiv:2509.20802.
2. Ko et al., *DistiLLM: Towards Streamlined Distillation for Large Language
   Models*, ICML 2024, arXiv:2402.03898 (skew KL divergence).
3. Muralidharan et al., *Compact Language Models via Pruning and Knowledge
   Distillation*, NeurIPS 2024 (multi-level distillation loss, cosine layer
   importance).
