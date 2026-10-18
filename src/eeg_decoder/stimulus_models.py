"""Stimulus-driven EEG retrieval models (PyTorch): final contrastive dual encoder.

Rule compliance
---------------
* Fitting uses only official *training* recordings (all six subjects) and stimulus videos.
* Test recordings are only transformed with per-set unsupervised normalisation
  (mean over the set's trials removed, channels scaled) and then scored.
* Scores are passed to the separate CSLS/Hungarian final ranking stage.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .cortiva_preprocessing import bandpass_trials

EEG_RATE = 50
T, C = 150, 17


# ---------------------------------------------------------------------------
# EEG preparation (per-set unsupervised normalisation)
# ---------------------------------------------------------------------------
def prepare_eeg(eeg: NDArray, band: tuple[float, float] = (0.5, 8.0)) -> NDArray[np.float32]:
    """(N, 17, 1500) @500 Hz -> (N, 150, 17): band-pass, decimate to 50 Hz, remove the
    set-mean response and scale channels by the set's own standard deviation."""
    filtered = bandpass_trials(eeg, sample_rate_hz=500.0, low_hz=band[0], high_hz=band[1])[:, :, :: 500 // EEG_RATE]
    filtered = filtered - filtered.mean(0, keepdims=True)
    filtered = filtered / (filtered.std(axis=(0, 2), keepdims=True) + 1e-8)
    return np.ascontiguousarray(filtered.transpose(0, 2, 1), dtype=np.float32)





# ---------------------------------------------------------------------------
# Dual encoder
# ---------------------------------------------------------------------------
@dataclass
class DualConfig:
    hidden: int = 128
    latent: int = 32
    blocks: int = 4
    kernel: int = 5
    dropout: float = 0.5
    in_dropout: float = 0.4
    eeg_kernel: int = 9
    batch: int = 256
    epochs: int = 60
    stop_epoch: int = 0
    lr: float = 1e-3
    wd: float = 0.05
    noise: float = 0.5
    seed: int = 0
    multipos: int = 0          # >0: batches of this many videos with all their trials (multi-positive InfoNCE)
    splice: float = 0.0        # probability per batch of adding spliced pseudo-videos (needs multipos)
    extra: dict = field(default_factory=dict)


def _torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return torch, nn, F


def build_modules(n_features: int, n_subjects: int, cfg: DualConfig):
    torch, nn, F = _torch()

    class CausalBlock(nn.Module):
        def __init__(self, ch, kernel, dilation, dropout):
            super().__init__()
            self.pad = (kernel - 1) * dilation
            self.conv = nn.Conv1d(ch, 2 * ch, kernel, dilation=dilation)
            self.norm = nn.GroupNorm(1, ch)
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            return self.norm(x + self.drop(F.glu(self.conv(F.pad(x, (self.pad, 0))), dim=1)))

    class StimEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.Dropout(cfg.in_dropout), nn.Linear(n_features, cfg.hidden))
            self.blocks = nn.ModuleList([CausalBlock(cfg.hidden, cfg.kernel, 2 ** i, cfg.dropout)
                                         for i in range(cfg.blocks)])
            self.out = nn.Conv1d(cfg.hidden, cfg.latent, 1)

        def forward(self, x):
            h = self.inp(x).transpose(1, 2)
            for block in self.blocks:
                h = block(h)
            return self.out(h)

    class EEGEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.spatial = nn.Parameter(torch.randn(n_subjects, cfg.latent, C) * 0.1)
            self.shared_spatial = nn.Parameter(torch.randn(cfg.latent, C) * 0.1)
            k = cfg.eeg_kernel
            self.temporal = nn.Conv1d(cfg.latent, cfg.latent, k, padding=k // 2, groups=cfg.latent)

        def forward(self, eeg, subj):
            return self.temporal(torch.einsum("bkc,btc->bkt", self.spatial[subj] + self.shared_spatial, eeg))

    return StimEncoder(), EEGEncoder()


def _znorm(z):
    z = z - z.mean(-1, keepdim=True)
    z = z / (z.std(-1, keepdim=True) + 1e-6)
    return z.flatten(1) / np.sqrt(z.shape[1] * z.shape[2])


class DualEncoder:
    """Contrastive (InfoNCE) alignment of stimulus feature time series and EEG time series."""

    def __init__(self, features: NDArray, subjects: Sequence[str], cfg: DualConfig, device: str = "cuda"):
        torch, nn, _ = _torch()
        self.cfg, self.device, self.subjects = cfg, device, list(subjects)
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.features = torch.as_tensor(np.asarray(features, np.float32), device=device)  # (videos, T, F)
        self.stim, self.eeg = build_modules(self.features.shape[-1], len(self.subjects), cfg)
        self.stim.to(device); self.eeg.to(device)
        self.log_temp = nn.Parameter(torch.tensor(np.log(1 / 0.07), device=device))

    def fit(self, train_eeg: Mapping[str, NDArray], train_ids: Mapping[str, NDArray], log=None):
        """train_eeg[s]: prepared (N, 150, 17); train_ids[s]: 1-based video IDs (rows of features)."""
        torch, _, F = _torch()
        cfg, dev = self.cfg, self.device
        E = torch.as_tensor(np.concatenate([train_eeg[s] for s in self.subjects]), device=dev)
        V = torch.as_tensor(np.concatenate([train_ids[s] for s in self.subjects]), device=dev).long()
        S = torch.as_tensor(np.concatenate([np.full(len(train_ids[s]), i) for i, s in enumerate(self.subjects)]),
                            device=dev).long()
        params = list(self.stim.parameters()) + list(self.eeg.parameters()) + [self.log_temp]
        opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.wd)

        def batches():
            perm = torch.randperm(len(E), device=dev)
            _, first = np.unique(V[perm].cpu().numpy(), return_index=True)
            idx = perm[torch.as_tensor(np.sort(first), device=dev)]
            idx = idx[torch.randperm(len(idx), device=dev)]
            return [idx[i:i + cfg.batch] for i in range(0, len(idx) - cfg.batch + 1, cfg.batch)]

        vnp = V.cpu().numpy()
        videos = np.unique(vnp)
        trials_of = {int(k): np.flatnonzero(vnp == k) for k in videos}

        def video_batches():
            """Multi-positive batches: `multipos` distinct videos together with all their trials."""
            order = np.random.permutation(videos)
            out = []
            for i in range(0, len(order) - cfg.multipos + 1, cfg.multipos):
                vids = order[i:i + cfg.multipos]
                out.append((torch.as_tensor(vids, device=dev).long(),
                            torch.as_tensor(np.concatenate([trials_of[int(v)] for v in vids]), device=dev)))
            return out

        def loss_single(idx):
            eeg = E[idx] + cfg.noise * torch.randn_like(E[idx])
            e = _znorm(self.eeg(eeg, S[idx]))
            s = _znorm(self.stim(self.features[V[idx] - 1]))
            logits = self.log_temp.exp().clamp(max=100) * (e @ s.T)
            target = torch.arange(len(idx), device=dev)
            return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))

        trial_of = {(int(v), int(s)): i for i, (v, s) in enumerate(zip(vnp, S.cpu().numpy()))}
        n_steps = E.shape[1]

        def spliced(vids):
            """Pseudo-videos a[:t] ++ b[t:] (t random in 0.6-2.4 s) with EEG spliced at the same t from
            the SAME subject's training trials of a and b. Multiplies the training pairs without new data."""
            va = vids.cpu().numpy()
            vb = np.random.permutation(va)
            t = int(np.random.randint(30, 121))
            x = torch.cat([self.features[torch.as_tensor(va - 1, device=dev)][:, :t],
                           self.features[torch.as_tensor(vb - 1, device=dev)][:, t:n_steps]], 1)
            ia, ib, col, subj = [], [], [], []
            for k, (a, b) in enumerate(zip(va, vb)):
                if a == b:
                    continue
                for si in range(len(self.subjects)):
                    if (int(a), si) in trial_of and (int(b), si) in trial_of:
                        ia.append(trial_of[(int(a), si)]); ib.append(trial_of[(int(b), si)])
                        col.append(k); subj.append(si)
            if not ia:
                return None
            ia, ib = torch.as_tensor(ia, device=dev), torch.as_tensor(ib, device=dev)
            eeg = torch.cat([E[ia][:, :t], E[ib][:, t:]], 1)
            return x, eeg, torch.as_tensor(subj, device=dev), torch.as_tensor(col, device=dev)

        def loss_multipos(batch):
            vids, trials = batch
            x, eeg, subj = self.features[vids - 1], E[trials], S[trials]
            pos = (V[trials][:, None] == vids[None, :]).float()
            if cfg.splice > np.random.rand():
                sp = spliced(vids)
                if sp is not None:
                    xs, es, ss, col = sp
                    ps = torch.zeros(len(es), len(vids) + len(xs), device=dev)
                    ps[torch.arange(len(es), device=dev), len(vids) + col] = 1.0
                    pos = torch.cat([torch.cat([pos, torch.zeros(len(pos), len(xs), device=dev)], 1), ps], 0)
                    x, eeg, subj = torch.cat([x, xs], 0), torch.cat([eeg, es], 0), torch.cat([subj, ss], 0)
            eeg = eeg + cfg.noise * torch.randn_like(eeg)
            e = _znorm(self.eeg(eeg, subj))
            s = _znorm(self.stim(x))
            logits = self.log_temp.exp().clamp(max=100) * (e @ s.T)          # (trials, videos)
            eeg_to_stim = F.cross_entropy(logits, pos.argmax(1))
            logp = torch.log_softmax(logits, dim=0)                          # each video over all trials
            stim_to_eeg = -((logp * pos).sum(0) / pos.sum(0).clamp(min=1)).mean()
            return 0.5 * (eeg_to_stim + stim_to_eeg)

        make, loss_fn = (video_batches, loss_multipos) if cfg.multipos else (batches, loss_single)
        steps = cfg.epochs * max(1, len(make()))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=steps + 1, pct_start=0.1)
        last = cfg.stop_epoch or cfg.epochs
        for epoch in range(1, last + 1):
            losses = []
            for batch in make():
                loss = loss_fn(batch)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                opt.step()
                sched.step()
                losses.append(loss.item())
            if log and (epoch % 10 == 0 or epoch == last):
                log(f"    epoch {epoch} loss {np.mean(losses):.3f}")
        return self

    def score(self, eeg: NDArray, subject: str, candidate_ids: NDArray) -> NDArray[np.float64]:
        """Similarity (queries, candidates). Candidate embeddings are centred across candidates."""
        torch, _, _ = _torch()
        self.stim.eval(); self.eeg.eval()
        with torch.no_grad():
            s = _znorm(self.stim(self.features[torch.as_tensor(np.asarray(candidate_ids) - 1, device=self.device)]))
            s = s - s.mean(0, keepdim=True)
            x = torch.as_tensor(np.asarray(eeg, np.float32), device=self.device)
            si = torch.full((len(x),), self.subjects.index(subject), device=self.device)
            e = _znorm(self.eeg(x, si))
            out = (e @ s.T).double().cpu().numpy()
        self.stim.train(); self.eeg.train()
        return out

    def state(self) -> dict:
        return {"stim": self.stim.state_dict(), "eeg": self.eeg.state_dict(), "config": asdict(self.cfg),
                "subjects": self.subjects}




def zrow(scores: NDArray) -> NDArray:
    s = scores - scores.mean(1, keepdims=True)
    return s / (s.std(1, keepdims=True) + 1e-9)

def fuse(score_sets: Sequence[NDArray], weights: Sequence[float] | None = None) -> NDArray:
    weights = list(weights) if weights is not None else [1.0] * len(score_sets)
    return sum(w * zrow(s) for w, s in zip(weights, score_sets)) / sum(weights)

def ranked_candidates(scores: NDArray, candidate_ids: NDArray) -> NDArray[np.int64]:
    """Full ranking per row (most to least likely), rows in the original recording order."""
    order = np.argsort(-scores, axis=1, kind="stable")
    return np.asarray(candidate_ids, np.int64)[order]
