"""Opt-in external refits, saved artifacts and final-ranking integration."""
from pathlib import Path
import importlib.util
import json
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def cli():
    spec = importlib.util.spec_from_file_location("external_training_cli", ROOT / "scripts/final_v7.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("args", [
    ["replay", "--retrain-external", "both"],
    ["infer", "--data-root", ".", "--retrain-external", "audio", "--device", "cpu"],
    ["infer", "--data-root", ".", "--retrain-external", "audio", "--audio-freeze", ".", "--device", "cuda"],
])
def test_reject_invalid_refit_requests_before_work(tmp_path, args):
    with pytest.raises(SystemExit):
        cli().main([*args, "--output", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("selected", ["audio", "video", "both"])
def test_cli_refit_selection_reaches_final_submission(tmp_path, monkeypatch, selected):
    module = cli()
    captured = {}
    with np.load(ROOT / "scores/final.npz") as archive:
        frozen = {k: archive[k] for k in archive.files if k.startswith("fused__")}

    def fake_scores(*args, **kwargs):
        captured.update(kwargs)
        return frozen

    monkeypatch.setattr(module, "score_models", fake_scores)
    import eeg_decoder.external_training as training
    monkeypatch.setattr(training, "require_training_device", lambda device: None)
    output = tmp_path / "result"
    module.main(["infer", "--data-root", str(tmp_path), "--output", str(output),
                 "--device", "cuda", "--retrain-external", selected])
    expected = ["audio", "video"] if selected == "both" else [selected]
    assert captured["retrain_external"] == expected
    report = json.loads((output / "run.json").read_text())
    assert report["retrained_external"] == expected
    for subject in ("sub-05", "sub-06"):
        result = np.load(output / f"{subject}_ranked_candidate_list.npy")
        assert result.shape == (102, 102)
        np.testing.assert_array_equal(result, np.load(ROOT / f"predictions/{subject}_ranked_candidate_list.npy"))


@pytest.mark.parametrize("kind", ["audio", "video"])
def test_real_tiny_gpu_refit_can_be_reloaded(tmp_path, monkeypatch, kind):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        pytest.skip("Original teammate training requires a BF16 CUDA GPU")
    import eeg_decoder.external_training as training
    from eeg_decoder.external_families import external_scores, _load_coworker_module
    source = ROOT / f"external/{kind}-freeze"
    data = tmp_path / "data"
    (data / "trn").mkdir(parents=True)
    (data / "tst").mkdir()
    rng = np.random.default_rng(91)
    for sub in range(1, 7):
        np.savez(data / f"trn/sub-{sub:02d}_trn.npz", **{
            str(i): rng.normal(size=(17, 1500)).astype(np.float32) for i in range(1, 21)})
        np.save(data / f"tst/sub-{sub:02d}_tst.npy", rng.normal(size=(102, 17, 1500)).astype(np.float32))
    original_loader = training._load_coworker_module

    def tiny_loader(*args, **kwargs):
        model_module = original_loader(*args, **kwargs)
        model_module.TRAIN_VIDEO_IDS = np.arange(1, 21)
        model_module.FT_MAX_EPOCHS = 1
        return model_module

    monkeypatch.setattr(training, "_load_coworker_module", tiny_loader)
    destination = tmp_path / f"{kind}-freeze"
    subjects = [f"sub-{i:02d}" for i in range(1, 7)]
    scores = training.retrain_family(kind, source, data, destination, subjects, "cuda")
    assert len(list((destination / "model").glob("seed*.pt"))) == 4
    assert (destination / "model/normalisation.npz").exists()
    report = json.loads((destination / "training.json").read_text())
    assert report["test_labels_used"] is False
    assert report["fit_video_ids"] == list(range(1, 21))
    assert all(len(row["training"]["history"]) == 1 for row in report["seeds"])
    # Re-load the saved weights and transformations through the real production adapter.
    loaded = external_scores({"kind": f"coworker-{kind}", "path": str(destination)}, data, subjects, device="cuda")
    for s in subjects:
        assert scores[s].shape == (102, 102)
        assert np.isfinite(scores[s]).all()
        np.testing.assert_allclose(loaded[s], scores[s], atol=2e-5, rtol=2e-5)
    with pytest.raises(FileExistsError):
        training.retrain_family(kind, source, data, destination, subjects, "cuda")


@pytest.mark.parametrize("selected", [("audio",), ("video",), ("audio", "video")])
def test_scoring_replaces_only_selected_families(tmp_path, monkeypatch, selected):
    pytest.importorskip("torch")
    from types import SimpleNamespace
    import eeg_decoder.external_training as training
    import eeg_decoder.external_families as external
    import eeg_decoder.stimulus_models as models
    import eeg_decoder.stimulus_pipeline as features
    import eeg_decoder.cortiva_challenge as challenge
    rng = np.random.default_rng(12)
    subjects = ["sub-01", "sub-02"]
    internal = {s: rng.normal(size=(102, 102)) for s in subjects}
    frozen = {k: {s: rng.normal(size=(102, 102)) for s in subjects} for k in ("audio", "video")}
    retrained = {k: {s: rng.normal(size=(102, 102)) for s in subjects} for k in ("audio", "video")}
    seen = []

    class Model:
        def __init__(self, *args): pass
        def fit(self, *args): pass
        def state(self): return {}
        def score(self, eeg, subject, ids): return internal[subject]

    def refit(kind, source, data, destination, names, device):
        seen.append(kind)
        return retrained[kind]

    monkeypatch.setattr(models, "DualEncoder", Model)
    monkeypatch.setattr(models, "prepare_eeg", lambda x, *args: x)
    monkeypatch.setattr(features, "build_features", lambda *args: None)
    monkeypatch.setattr(challenge, "load_challenge_test_subject", lambda *args: SimpleNamespace(test_eeg=None))
    monkeypatch.setattr(challenge, "load_challenge_training_subject", lambda *args: SimpleNamespace(train_eeg=None, train_video_ids=None))
    monkeypatch.setattr(training, "retrain_family", refit)
    monkeypatch.setattr(external, "external_scores", lambda spec, *args, **kwargs: frozen[spec["kind"].removeprefix("coworker-")])
    config = {"subjects": subjects, "candidate_ids": list(range(1001, 1103)), "band": [.2, 12],
              "families": [{"name": "tiny", "features": {}, "config": {}, "seeds": [0]}],
              "group_weights": {"internal": .7, "coworker-audio": .2, "coworker-video": .1}}
    actual = cli().score_models(config, tmp_path, tmp_path, "cuda", train=True, retrain_external=selected)
    assert seen == list(selected)
    for subject in subjects:
        groups = [internal[subject]] + [(retrained if k in selected else frozen)[k][subject] for k in ("audio", "video")]
        # Internal aggregation standardises once more before final fusion.
        groups[0] = models.fuse([internal[subject]])
        np.testing.assert_allclose(actual[f"fused__{subject}"], models.fuse(groups, [.7, .2, .1]), atol=1e-12)
