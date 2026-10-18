"""Audio-only dual encoder (CORTIVA-style), exported from pretrained_transformers/06 for run 2026-10-18_134354.

The code below is the code that produced this freeze, taken from the notebook kernel's input history.
Use:
    import audio_dual_encoder as ade
    ade.load_data()                                  # our training EEG (needs the challenge data, EEG_DATA_ROOT)
    model = ade.load_frozen_model(seed=0)            # one of the frozen models (seeds [0, 1, 2, 3])
    # score EEG windows X (n, 17, 150; preprocessed like our_inputs) of subject s against candidate videos:
    #   ade.scores(model, X, slot, feats[rows])     with slot = ade.OUR_SLOT[s], feats z-scored with training videos
    # refit on another split (e.g. CORTIVA's holdout), to get holdout scores for the fusion weight:
    #   model, queries, feats = ade.fit_ours(VARIANT, ENCODER, seed, fit_video_ids, holdout_video_ids, 'cuda:0')
"""
import json, math, os, queue, threading
from math import gcd
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

FREEZE_DIR = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get('EEG_DATA_ROOT', str(Path.home() / 'eeg-data' / 'eeg-mind-reading-challenge'))).expanduser()
if (DATA_ROOT / 'eeg-mind-reading-challenge').is_dir():
    DATA_ROOT /= 'eeg-mind-reading-challenge'
VARIANT, ENCODER = 'scratch', 'cortiva+transformer'
TRAIN_VIDEO_IDS = np.arange(1, 1001)
TEST_VIDEO_IDS = np.arange(1001, 1103)
CKPT_DIR = FREEZE_DIR / 'model'          # only used by pretrained variants (not by 'scratch')
STOP = threading.Event()
INIT_LOCK = threading.Lock()

CHANNELS = ['Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2']
SUBJECTS = [1, 2, 3, 4, 5, 6]
EEG_SFREQ = 500
BAND_HZ = (0.2, 12.0)
RATE_HZ = 50
WINDOW_S = 3.0
N_STEPS = 150
HIDDEN = 128
LATENT = 32
KERNEL = 5
DILATIONS = (1, 2, 4, 8)
INPUT_DROPOUT = 0.5
DROPOUT = 0.5
GROUP_NORM_GROUPS = 8
EEG_KERNEL = 9
TF_DIM = 64
TF_LAYERS = 2
TF_HEADS = 4
TF_DROPOUT = 0.1
TEMPERATURE_INIT = 0.07
LR = 0.002
WEIGHT_DECAY = 0.05
STIMULI_PER_BATCH = 16
FT_LR = 0.0006666666666666666
FT_MAX_EPOCHS = 60
FT_PATIENCE = 6
INNER_FRACTION = 0.1
INNER_SEED = 0
N_FEATURES = 145
N_SLOTS = 77
OUR_SLOT = {1: 71, 2: 72, 3: 73, 4: 74, 5: 75, 6: 76}


class Progress:
    """One shared text progress bar for parallel GPU jobs (thread-safe). expect(n) adds n units to the total,
    tick(n) marks n units done, note(key, text) shows each job's status after the bar."""

    def __init__(self, bar):
        self.bar, self.lock, self.notes = bar, threading.Lock(), {}

    def expect(self, n):
        with self.lock:
            self.bar.total = (self.bar.total or 0) + n
            self.bar.refresh()

    def tick(self, n=1):
        with self.lock:
            self.bar.update(n)

    def note(self, key, text):
        with self.lock:
            self.notes[key] = text
            self.bar.set_postfix_str(' | '.join(f'{k}: {v}' for k, v in sorted(self.notes.items())), refresh=False)


class CausalBlock(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.pad = (KERNEL - 1) * dilation
        self.conv = nn.Conv1d(HIDDEN, 2 * HIDDEN, KERNEL, dilation=dilation)
        self.norm = nn.GroupNorm(GROUP_NORM_GROUPS, HIDDEN)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):                                # (B, HIDDEN, T)
        y = F.glu(self.conv(F.pad(x, (self.pad, 0))), dim=1)
        return x + self.drop(self.norm(y))


class DualEncoder(nn.Module):
    def __init__(self, n_features, n_slots, eeg_encoder):
        super().__init__()
        self.stim_in = nn.Sequential(nn.Dropout(INPUT_DROPOUT), nn.Linear(n_features, HIDDEN))
        self.stim_blocks = nn.Sequential(*[CausalBlock(d) for d in DILATIONS])
        self.stim_out = nn.Conv1d(HIDDEN, LATENT, 1)
        self.subject_proj = nn.Parameter(torch.zeros(n_slots, LATENT, len(CHANNELS)))   # per subject, starts at 0
        self.shared_proj = nn.Linear(len(CHANNELS), LATENT, bias=False)
        self.eeg_conv = nn.Conv1d(LATENT, LATENT, EEG_KERNEL, padding=EEG_KERNEL // 2, groups=LATENT)
        self.use_tf = eeg_encoder == 'cortiva+transformer'
        if self.use_tf:
            self.tf_in = nn.Linear(LATENT, TF_DIM)
            self.tf_pos = nn.Parameter(0.02 * torch.randn(1, N_STEPS, TF_DIM))
            layer = nn.TransformerEncoderLayer(TF_DIM, TF_HEADS, 2 * TF_DIM, TF_DROPOUT, activation='gelu',
                                               batch_first=True, norm_first=True)
            self.tf = nn.TransformerEncoder(layer, TF_LAYERS, enable_nested_tensor=False)
            self.tf_out = nn.Linear(TF_DIM, LATENT)
            nn.init.zeros_(self.tf_out.weight)
            nn.init.zeros_(self.tf_out.bias)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / TEMPERATURE_INIT)))

    @staticmethod
    def zscore(z):                                       # per latent channel over time, then flattened
        z = (z - z.mean(-1, keepdim=True)) / (z.std(-1, keepdim=True) + 1e-6)
        return z.flatten(1) / math.sqrt(z.shape[1] * z.shape[2])

    def stimulus(self, feats):                           # (B, 150, K) -> (B, LATENT * 150)
        h = self.stim_in(feats).transpose(1, 2)
        return self.zscore(self.stim_out(self.stim_blocks(h)))

    def eeg(self, x, slot):                              # (B, 17, 150) -> (B, LATENT * 150)
        w = self.shared_proj.weight + self.subject_proj[slot]                     # (B, LATENT, 17)
        z = self.eeg_conv(torch.bmm(w, x))
        if self.use_tf:
            h = self.tf(self.tf_in(z.transpose(1, 2)) + self.tf_pos)
            z = z + self.tf_out(h).transpose(1, 2)
        return self.zscore(z)


def new_model(n_slots, eeg_encoder, seed, device):
    with INIT_LOCK:
        torch.manual_seed(seed)
        model = DualEncoder(N_FEATURES, n_slots, eeg_encoder)
    return model.to(device)


def info_nce(z_eeg, z_stim, target, logit_scale):
    """Symmetric multi-positive InfoNCE: EEG -> its stimulus among the batch stimuli, and every stimulus -> all of
    its recordings."""
    logits = logit_scale.exp().clamp(max=100) * z_eeg @ z_stim.T                # (EEG, stimuli)
    l_eeg = F.cross_entropy(logits, target)
    pos = F.one_hot(target, z_stim.shape[0]).float()
    l_stim = -(logits.log_softmax(0) * pos).sum(0) / pos.sum(0).clamp(min=1)
    return (l_eeg + l_stim[pos.sum(0) > 0].mean()) / 2


@torch.no_grad()
def scores(model, X, slot, feats):
    """Similarity of EEG windows X (n, 17, 150) to candidate stimuli feats (m, 150, K), candidates centred."""
    model.eval()
    s = model.stimulus(feats)
    s = s - s.mean(0, keepdim=True)
    return torch.cat([model.eeg(X[i:i + 4096], slot[i:i + 4096]) for i in range(0, len(X), 4096)]) @ s.T


def ranks(S, true_cols):
    return (S > S[torch.arange(len(S), device=S.device), true_cols][:, None]).sum(1) + 1


def row_z(S):
    return (S - S.mean(axis=1, keepdims=True)) / S.std(axis=1, keepdims=True)


def metrics(S, true_cols):
    S = np.asarray(S)
    r = (S > S[np.arange(len(S)), true_cols][:, None]).sum(axis=1) + 1
    return dict(n=len(r), top1=float(np.mean(r == 1)), top5=float(np.mean(r <= 5)), mean_rank=float(r.mean()))


def train(model, X, slot, stim_of, feats, epochs, seed, progress, name, val_fn=None, patience=None, params=None,
          lr=LR):
    """Batches of STIMULI_PER_BATCH stimuli with all their recordings. With val_fn: keeps the best epoch and stops
    after `patience` epochs without improvement. params: the trained parameters (default: all). Reports steps and
    scores to the shared progress bar."""
    gen = np.random.default_rng(seed)
    by_stim = {}
    for i, s in enumerate(stim_of.tolist()):
        by_stim.setdefault(s, []).append(i)
    stims = np.array(sorted(by_stim))
    steps = math.ceil(len(stims) / STIMULI_PER_BATCH)
    opt = torch.optim.AdamW(params if params is not None else model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY,
                            fused=True)
    progress.expect(epochs * steps)
    best, best_state, wait, done, best_epoch, history = -math.inf, None, 0, 0, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        order = gen.permutation(stims)
        for b in range(steps):
            if STOP.is_set():
                raise RuntimeError('stopped')
            batch = order[b * STIMULI_PER_BATCH:(b + 1) * STIMULI_PER_BATCH]
            rows = torch.as_tensor(np.concatenate([by_stim[s] for s in batch]), device=X.device)
            local = {s: j for j, s in enumerate(batch.tolist())}
            target = torch.as_tensor([local[s] for s in stim_of[rows.cpu().numpy()].tolist()], device=X.device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                z_stim = model.stimulus(feats[torch.as_tensor(batch, device=X.device)].float())
                z_eeg = model.eeg(X[rows].float(), slot[rows])
            loss = info_nce(z_eeg.float(), z_stim.float(), target, model.logit_scale)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            done += 1
            progress.tick()
        if val_fn is not None:
            score = val_fn(model)
            history.append(dict(epoch=epoch, score=score))
            progress.note(name, f'ep {epoch} val {score:.3f} best {max(best, score):.3f}')
            if score > best:
                best, wait = score, 0
                best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= patience:
                    break
    progress.tick(epochs * steps - done)                 # steps skipped by early stopping
    if best_state is not None:
        model.load_state_dict(best_state)
    return dict(best_val=best, epochs=epoch, best_epoch=best_epoch, history=history)


def bandpass_resample(x, sfreq, device='cuda:0'):
    """(channels, samples) at sfreq -> float32 (channels, samples') at RATE_HZ, band-passed to BAND_HZ.
    One FFT step: the recording is mirrored (no jump at the ends), frequencies outside the band are set to zero,
    and the inverse FFT is taken with the length of the new rate (exact for a band-limited signal). The recording is
    first shortened by < 1 s to a length that converts to a whole number of samples."""
    step = sfreq // gcd(sfreq, RATE_HZ)
    x = torch.as_tensor(np.asarray(x)[:, :x.shape[1] // step * step], dtype=torch.float64, device=device)
    n = x.shape[1]
    m = n * RATE_HZ // sfreq
    spec = torch.fft.rfft(torch.cat([x, x.flip(-1)], dim=-1), dim=-1, norm='forward')
    f = torch.fft.rfftfreq(2 * n, d=1 / sfreq, dtype=torch.float64, device=device)
    spec[..., (f < BAND_HZ[0]) | (f > BAND_HZ[1])] = 0
    return torch.fft.irfft(spec[..., :m + 1], n=2 * m, dim=-1, norm='forward')[..., :m].float().cpu().numpy()


def load_ours(sub, split='trn'):
    if split == 'trn':
        with np.load(DATA_ROOT / 'trn' / f'sub-{sub:02d}_trn.npz') as z:
            keys = sorted(z.files, key=int)
            ids, eeg = np.array([int(k) for k in keys]), np.stack([z[k] for k in keys])
    else:
        ids, eeg = None, np.load(DATA_ROOT / 'tst' / f'sub-{sub:02d}_tst.npy')
    n, c, _ = eeg.shape
    return ids, bandpass_resample(eeg.reshape(n * c, -1), EEG_SFREQ).reshape(n, c, -1)   # all recordings in one call


def our_inputs(fit_videos, query, device):
    """Fitting recordings of all six subjects (normalised with their own statistics) and the query recordings."""
    X, slot, vid, queries = [], [], [], {}
    for sub in SUBJECTS:
        ids, eeg = OURS[sub]
        fit = np.isin(ids, fit_videos)
        mean = eeg[fit].mean(0, keepdims=True)                                   # mean response (17, 150)
        scale = (eeg[fit] - mean).std(axis=(0, 2), keepdims=True)                # one std per channel
        norm = lambda e: torch.as_tensor((e - mean) / scale, dtype=torch.float32, device=device)
        X.append(norm(eeg[fit]))
        vid.append(ids[fit])
        slot.append(torch.full((int(fit.sum()),), OUR_SLOT[sub], device=device))
        if isinstance(query, str):
            queries[sub] = (norm(load_ours(sub, 'tst')[1]), None)
        else:
            q = np.isin(ids, query)
            queries[sub] = (norm(eeg[q]), ids[q])
    return torch.cat(X), torch.cat(slot), np.concatenate(vid), queries


def fit_ours(variant, encoder, seed, fit_videos, query, device, progress=None, name=None, *, return_info=False):
    """One variant on fit_videos (minus an inner early-stopping set); returns the model, query recordings, features."""
    X, slot, vid, queries = our_inputs(fit_videos, query, device)
    fv = VID_FEATS[ROW_OF[fit_videos]]
    mu, sd = fv.mean((0, 1)), fv.std((0, 1)) + 1e-6                              # fitting videos only
    feats = torch.as_tensor((VID_FEATS - mu) / sd, dtype=torch.float32, device=device)
    model = new_model(N_SLOTS, encoder, seed, device)
    if variant != 'scratch':
        model.load_state_dict(torch.load(CKPT_DIR / f'{encoder}_seed{seed}.pt', map_location=device, weights_only=True))
    if variant == 'zero-shot':
        return model, queries, feats
    rng = np.random.default_rng(INNER_SEED)                  # same inner videos for every variant and seed
    inner = np.sort(rng.choice(fit_videos, int(round(INNER_FRACTION * len(fit_videos))), replace=False))
    is_inner = np.isin(vid, inner)
    mask = torch.as_tensor(is_inner, device=device)
    inner_cand = feats[torch.as_tensor(ROW_OF[inner], device=device)]
    inner_true = torch.as_tensor(np.searchsorted(inner, vid[is_inner]), device=device)

    def val_fn(m):                   # higher = better: minus the mean rank of the inner recordings among the inner videos
        return -float(ranks(scores(m, X[mask], slot[mask], inner_cand), inner_true).float().mean())

    params = [model.subject_proj] if variant == 'frozen' else None               # frozen: our projections only
    info = train(model, X[~mask], slot[~mask], ROW_OF[vid[~is_inner]], feats, FT_MAX_EPOCHS, seed,
          progress if progress else Progress(tqdm(disable=True)), name, val_fn, FT_PATIENCE, params=params,
          lr=FT_LR if variant == 'finetune' else LR)
    return (model, queries, feats, info) if return_info else (model, queries, feats)


VID_FEATS = np.load(FREEZE_DIR / 'features' / 'video_features_50hz.npy')
ALL_VIDEO_IDS = np.arange(1, 1 + len(VID_FEATS))
ROW_OF = np.full(ALL_VIDEO_IDS.max() + 1, -1)
ROW_OF[ALL_VIDEO_IDS] = np.arange(len(ALL_VIDEO_IDS))
OURS = {}


def load_data():
    OURS.update({sub: load_ours(sub) for sub in SUBJECTS})


def load_frozen_model(seed, device='cuda:0'):
    model = DualEncoder(N_FEATURES, N_SLOTS, ENCODER)
    model.load_state_dict(torch.load(FREEZE_DIR / 'model' / f'seed{seed}.pt', map_location=device, weights_only=True))
    return model.to(device).eval()
