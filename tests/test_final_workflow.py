from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_cli():
    spec = importlib.util.spec_from_file_location("final_v7", ROOT / "scripts/final_v7.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_replay_exact_without_data_or_labels(tmp_path):
    subprocess.run([sys.executable, str(ROOT / "scripts/final_v7.py"), "replay", "--output", str(tmp_path)], check=True)
    assert (tmp_path / "submission.zip").read_bytes() == (ROOT / "predictions/submission.zip").read_bytes()


def test_decision_audit_reproduces_recorded_family_and_weight_searches(tmp_path):
    output = tmp_path / "audit.json"
    subprocess.run([sys.executable, str(ROOT / "scripts/audit_decisions.py"), "--output", str(output)], check=True)
    result = json.loads(output.read_text())
    assert result["selected"] == {"audio": .2, "video": .1, "internal": .7, "csls_k": 1}
    assert result["family_greedy_searches"]["family-select"][-1]["added"] == "I-arich-splice"


def test_output_protection_and_required_data():
    cli = load_cli()
    for argv in (["replay", "--output", str(ROOT / "predictions")], ["infer", "--output", str(ROOT / "outputs/test")]):
        with pytest.raises(SystemExit):
            cli.main(argv)


def test_all_frozen_checkpoint_configs_agree_with_explicit_config():
    torch = pytest.importorskip("torch")
    config = json.loads((ROOT / "config/final.json").read_text())
    for family in config["families"]:
        for seed in family["seeds"]:
            state = torch.load(ROOT / f"checkpoints/{family['name']}-seed{seed}.pt", map_location="cpu", weights_only=True)
            assert state["subjects"] == config["subjects"]
            assert state["config"] == {**family["config"], "seed": seed}


def test_synthetic_internal_training_then_weight_inference(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(2)
    from eeg_decoder.stimulus_models import DualConfig
    import dataclasses
    import eeg_decoder.stimulus_pipeline as features
    import eeg_decoder.external_families as external
    cli = load_cli()
    subjects = ["sub-01", "sub-02"]
    rng = np.random.default_rng(11)
    data = tmp_path / "data"
    for split in ("trn", "tst"):
        (data / split).mkdir(parents=True)
    for s in subjects:
        np.savez(data / f"trn/{s}_trn.npz", **{str(i): rng.normal(size=(17,1500)).astype(np.float32) for i in range(1,5)})
        np.save(data / f"tst/{s}_tst.npy", rng.normal(size=(102,17,1500)).astype(np.float32))
    bank = rng.normal(size=(1102,150,3)).astype(np.float32)
    monkeypatch.setattr(features, "build_features", lambda *a, **kw: bank)
    ext = {s: rng.normal(size=(102,102)) for s in subjects}
    monkeypatch.setattr(external, "external_scores", lambda *a, **kw: ext)
    cfg = dataclasses.asdict(DualConfig(hidden=8, latent=4, blocks=1, multipos=2, epochs=1, stop_epoch=1,
                                       splice=0, noise=0, batch=4))
    cfg.pop("seed")
    config = {"subjects": subjects, "candidate_ids": list(range(1001,1103)), "band": [0.2,12],
              "families": [{"name": "tiny", "features": {}, "config": cfg, "seeds": [7]}],
              "group_weights": {"internal": .7, "coworker-audio": .2, "coworker-video": .1}}
    output = tmp_path / "trained"
    output.mkdir()
    trained = cli.score_models(config, data, output, "cpu", train=True)
    monkeypatch.setattr(cli, "ROOT", output)
    inferred = cli.score_models(config, data, output, "cpu", train=False)
    for s in subjects:
        np.testing.assert_array_equal(trained[f"fused__{s}"], inferred[f"fused__{s}"])
