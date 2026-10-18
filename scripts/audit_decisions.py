"""Recalculate fusion shares and CSLS selection from training-holdout scores only."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eeg_decoder.stimulus_models import zrow, ranked_candidates
from eeg_decoder.stimulus_postprocess import retrieval_metrics
from eeg_decoder.cortiva_assignment import csls_assignment_rankings, assignment_rankings

FILES = ["G-w32-splice", "O-w32-arich", "D-clip-splice", "H-lowvis-splice", "I-arich-splice"]


def audit():
    config = json.loads((ROOT / "config/final.json").read_text())
    recorded = json.loads((ROOT / "validation/recorded_curves.json").read_text())
    root, subjects = ROOT / "validation/holdout", config["subjects"]
    archives = [np.load(root / f"{n}-k6.npz", allow_pickle=False) for n in FILES]
    audio = np.load(root / "coworker-holdout-k4.npz", allow_pickle=False)
    video = np.load(root / "coworker-video-holdout-k4.npz", allow_pickle=False)
    labels = {s: archives[0][f"ids__{s}"] for s in subjects}
    candidates = {s: np.unique(labels[s]) for s in subjects}
    split = np.sort(np.random.default_rng(17).choice(np.arange(1, 1001), size=150, replace=False))
    for archive in archives:
        for s in subjects:
            np.testing.assert_array_equal(archive[f"ids__{s}"], labels[s])
            assert np.isin(labels[s], split).all()
            assert archive[f"val__{s}"].shape == (len(labels[s]), len(candidates[s]))
    internal = {s: np.mean([zrow(a[f"val__{s}"]) for a in archives], axis=0) for s in subjects}

    def evaluate(scores, assignment=False, k=None):
        per = []
        for s in subjects:
            if k is not None:
                ranked = csls_assignment_rankings(scores[s], candidates[s], k=k)
            elif assignment:
                ranked = assignment_rankings(scores[s], candidates[s])
            else:
                ranked = ranked_candidates(scores[s], candidates[s])
            per.append(retrieval_metrics(ranked, labels[s]))
        return {m: float(np.mean([p[m] for p in per])) for m in ("top1", "top5", "mrr", "mean_rank")}

    searches = json.loads((ROOT / "validation/family_search.json").read_text())
    family_data = {p.name.removesuffix("-k6.npz"): np.load(p, allow_pickle=False)
                   for p in sorted(root.glob("*-k6.npz"))}
    def family_mix(names):
        for name in names:
            for s in subjects:
                np.testing.assert_array_equal(family_data[name][f"ids__{s}"], labels[s])
        return {s: np.mean([zrow(family_data[n][f"val__{s}"]) for n in names], axis=0) for s in subjects}
    greedy = {}
    for name, search in searches.items():
        remaining, chosen, best, steps = list(search["families"]), [], -1., []
        while remaining:
            results = {n: evaluate(family_mix(chosen + [n])) for n in remaining}
            candidate = max(remaining, key=lambda n: results[n]["mrr"])
            if results[candidate]["mrr"] <= best:
                break
            chosen.append(candidate)
            remaining.remove(candidate)
            best = results[candidate]["mrr"]
            steps.append({"added": candidate, "mrr": best})
        assert chosen == search["selected"], (name, chosen, search["selected"])
        greedy[name] = steps

    audio_curve = {}
    for w in recorded["audio"]:
        weight = float(w)
        scores = {s: (1 - weight) * zrow(internal[s]) + weight * zrow(audio[s]) for s in subjects}
        audio_curve[w] = evaluate(scores)
        for metric, expected in recorded["audio"][w].items():
            np.testing.assert_allclose(audio_curve[w][metric], expected, atol=1e-10, rtol=0)
    chosen_audio = max(audio_curve, key=lambda w: audio_curve[w]["mrr"])
    video_curve = {}
    for w in recorded["video"]:
        weight = float(w)
        scores = {s: (1 - float(chosen_audio) - weight) * zrow(internal[s]) + float(chosen_audio) * zrow(audio[s])
                     + weight * zrow(video[s]) for s in subjects}
        video_curve[w] = {"independent": evaluate(scores), "csls1_hungarian": evaluate(scores, k=1)}
        for route, measured in video_curve[w].items():
            for metric, expected in recorded["video"][w][route].items():
                np.testing.assert_allclose(measured[metric], expected, atol=1e-10, rtol=0)
    chosen_video = max(video_curve, key=lambda w: video_curve[w]["independent"]["mrr"])
    a, v = float(chosen_audio), float(chosen_video)
    final = {s: (1 - a - v) * zrow(internal[s]) + a * zrow(audio[s]) + v * zrow(video[s]) for s in subjects}
    csls = {str(k): evaluate(final, k=k) for k in (1, 5, 10, 20)}
    chosen_k = int(max(csls, key=lambda k: tuple(round(csls[k][m], 12) for m in ("top1", "top5", "mrr"))))
    np.testing.assert_allclose([a, v], [config["group_weights"]["coworker-audio"], config["group_weights"]["coworker-video"]])
    assert chosen_k == config["csls_k"]
    return {"protocol": "subject-macro metrics, training-video validation, not blind test",
            "split_seed": 17, "heldout_video_ids": split.tolist(),
            "query_counts": {s: len(labels[s]) for s in subjects},
            "candidate_counts": {s: len(candidates[s]) for s in subjects},
            "family_greedy_searches": greedy,
            "family_standalone": {n: evaluate(family_mix([n])) for n in family_data},
            "final_internal": evaluate(internal),
            "final_internal_leave_one_out": {n: evaluate(family_mix([m for m in FILES if m != n])) for n in FILES},
            "audio_curve": audio_curve, "video_curve": video_curve, "csls_curve": csls,
            "independent": evaluate(final), "hungarian_without_csls": evaluate(final, assignment=True),
            "selected": {"audio": a, "video": v, "internal": round(1-a-v, 12), "csls_k": chosen_k}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/decision-audit.json")
    args = parser.parse_args()
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["selected"], indent=2))
    print(f"All recorded weight curves reproduced; report: {args.output}")


if __name__ == "__main__":
    main()
