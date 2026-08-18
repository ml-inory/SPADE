import json

from spade.checkpoints import load_checkpoint, save_checkpoint
from spade.models import LLMTTSBackbone, LLMTTSConfig
from spade.pipeline import PipelineConfig, run_pipeline


def _tiny_pipeline_cfg(tmp_path):
    return PipelineConfig(
        output_dir=str(tmp_path),
        device="cpu",
        seed=0,
        teacher={
            "data": {
                "num_samples": 24,
                "num_eval_samples": 8,
                "num_speakers": 2,
                "use_prompt": False,
                "max_words": 5,
                "seed": 0,
            },
            "model": {
                "vocab_size": 512,
                "n_layer": 3,
                "n_head": 4,
                "n_embd": 32,
                "max_position_embeddings": 128,
                "num_codebooks": 1,
            },
            "batch_size": 4,
            "epochs": 1,
            "lr": 0.001,
            "log_every": 1000,
        },
        prune={
            "target_layers": 2,
            "eval_subset_size": 4,
            "force_keep_first_last": True,
            "compute_cli": True,
            "max_new_tokens": 48,
        },
        distill={
            "batch_size": 4,
            "distill": {
                "alpha": 0.25,
                "beta": 0.1,
                "matching": "dynamic",
                "epochs": 1,
                "log_every": 1000,
            },
        },
        eval={
            "subset_size": 4,
            "max_new_tokens": 48,
            "hop_seconds": 0.08,
            "n_runs": 1,
        },
    )


def test_smoke_pipeline_end_to_end(tmp_path):
    cfg = _tiny_pipeline_cfg(tmp_path)
    report = run_pipeline(cfg)

    # Pruning kept exactly the boundary layers (3 -> 2).
    assert report["pruning"]["retained_indices"] == [0, 2]
    assert report["pruning"]["target_layers"] == 2
    assert report["pruning"]["baseline_wer"] >= 0.0

    # Depth and parameter reduction.
    t = report["teacher"]["metrics"]
    s = report["student_before_distill"]["metrics"]
    d = report["student_after_distill"]["metrics"]
    assert t["depth"] == 3
    assert d["depth"] == 2
    assert d["params"] < t["params"]
    assert s["params"] == d["params"]

    # WER is a non-negative fraction (can exceed 1.0 due to insertions).
    assert 0.0 <= t["wer"]
    assert 0.0 <= d["wer"]
    assert report["speedup"]["depth_ratio"] == 1.5
    assert len(report["student_after_distill"]["history"]["train"]) == 1
    pipeline_report = tmp_path / "pipeline_report.json"
    assert pipeline_report.exists()
    assert json.loads(pipeline_report.read_text())["teacher"]["metrics"]["depth"] == 3


def test_checkpoint_roundtrip(tmp_path):
    model = LLMTTSBackbone(
        LLMTTSConfig(vocab_size=64, n_layer=2, n_head=2, n_embd=16)
    )
    path = tmp_path / "model.pt"
    save_checkpoint(model, path, meta={"hello": 1})
    loaded, meta = load_checkpoint(path)
    assert meta["meta"]["hello"] == 1
    assert loaded.config.n_layer == 2
    assert loaded.config.vocab_size == 64
    for k, v in model.state_dict().items():
        assert (loaded.state_dict()[k] == v).all()
