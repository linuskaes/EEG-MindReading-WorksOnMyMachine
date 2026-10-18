"""EEG transformer pretrained on THINGS-EEG2 and fine-tuned on our videos (EEG -> CLIP ViT-B/32 space),
exported from pretrained_transformers/05 for run 2026-10-18_144944. The code below is the code that produced this freeze,
taken from the notebook kernel's input history.
Use:
    import eeg_transformer_things as etr
    etr.load_data()                                   # our training EEG (needs the challenge data, EEG_DATA_ROOT)
    model = etr.load_frozen_model(seed=0)             # one of the frozen models (seeds [0, 1, 2, 3])
    # refit on another split (e.g. CORTIVA's holdout) and score the held-out videos, for a fusion weight:
    #   queries, model, info, audio_t = etr.train_on(VARIANT, seed, fit_ids, etr.ORDERS['recorded'], WINDOW,
    #                                                holdout_ids, 'cuda:0')
    #   sims = etr.similarities(model, queries, holdout_ids, torch.as_tensor(etr.VIDEO_TARGETS[WINDOW]).cuda(),
    #                           'cuda:0', audio_t)[SCORE]
"""
import json, math, os, threading
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import decimate
from tqdm import tqdm

FREEZE_DIR = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get('EEG_DATA_ROOT', str(Path.home() / 'eeg-data' / 'eeg-mind-reading-challenge'))).expanduser()
if (DATA_ROOT / 'eeg-mind-reading-challenge').is_dir():
    DATA_ROOT /= 'eeg-mind-reading-challenge'
VARIANT, WINDOW, SCORE = 'finetune', 'video 0-0.7 s (causal)', 'visual'
TRAIN_VIDEO_IDS = np.arange(1, 1001)
TEST_VIDEO_IDS = np.arange(1001, 1103)
CKPT_DIR = FREEZE_DIR / 'model' / 'pretrained_things'   # THINGS-EEG2 pretraining checkpoints (start of finetune)
STOP = threading.Event()
INIT_LOCK = threading.Lock()

CHANNELS = ['Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2']
SUBJECTS = [1, 2, 3, 4, 5, 6]
EEG_SFREQ = 500
MODEL_SFREQ = 100
N_CHANNELS = 17
N_SAMPLES = 80
PATCH = 5
D_MODEL = 128
DEPTH = 4
HEADS = 4
DROPOUT = 0.1
TEMPERATURE_INIT = 0.07
N_SLOTS = 16
THINGS_SLOTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
CHALLENGE_SLOT = {1: 10, 2: 11, 3: 12, 4: 13, 5: 14, 6: 15}
WARMUP_EPOCHS = 1
WEIGHT_DECAY = 0.05
LR = 0.0003
FT_LR = 9.999999999999999e-05
FT_BATCH = 256
FT_MAX_EPOCHS = 200
FT_PATIENCE = 20
INNER_FRACTION = 0.1
INNER_SEED = 0
AUDIO_HEAD = False
AUDIO_WEIGHT = 1.0


class Progress:
    """One shared text progress bar for parallel GPU jobs (thread-safe). Text bars cannot be stacked in a notebook,
    so parallel jobs report into one bar: expect(n) adds n units to the total (e.g. the training steps of one seed),
    tick(n) marks n units done, note(key, text) shows each job's status (epoch, scores) after the bar."""

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


class EEGTransformer(nn.Module):
    def __init__(self, audio_dim=None):
        super().__init__()
        n_patches = N_SAMPLES // PATCH
        self.subject_mix = nn.Parameter(torch.eye(N_CHANNELS).repeat(N_SLOTS, 1, 1))     # per-subject channel mixing
        self.subject_token = nn.Parameter(torch.zeros(N_SLOTS, D_MODEL))                 # per-subject summary token
        self.patch_embed = nn.Linear(PATCH, D_MODEL)
        self.channel_pos = nn.Parameter(0.02 * torch.randn(N_CHANNELS, 1, D_MODEL))
        self.time_pos = nn.Parameter(0.02 * torch.randn(1, n_patches, D_MODEL))
        layer = nn.TransformerEncoderLayer(D_MODEL, HEADS, 4 * D_MODEL, DROPOUT, activation='gelu',
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, DEPTH, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, 512)                                              # CLIP ViT-B/32 dimension
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / TEMPERATURE_INIT)))
        if audio_dim:                    # optional audio head, created last: the other weights are initialised as before
            self.audio_head = nn.Linear(D_MODEL, audio_dim)
            self.audio_logit_scale = nn.Parameter(torch.tensor(math.log(1 / TEMPERATURE_INIT)))

    def forward(self, x, slot, with_audio=False):
        x = torch.bmm(self.subject_mix[slot], x)                                    # (batch, 17, samples)
        b, c, t = x.shape
        tokens = self.patch_embed(x.reshape(b, c, t // PATCH, PATCH))               # (batch, 17, 16, d)
        tokens = (tokens + self.channel_pos + self.time_pos).flatten(1, 2)          # (batch, 272, d)
        h = self.norm(self.encoder(torch.cat([self.subject_token[slot][:, None], tokens], dim=1))[:, 0])
        z = F.normalize(self.head(h).float(), dim=-1)
        return (z, F.normalize(self.audio_head(h).float(), dim=-1)) if with_audio else z


def new_model(seed, device, audio_dim=None):
    with INIT_LOCK:
        torch.manual_seed(seed)
        model = EEGTransformer(audio_dim)
    return model.to(device)


def contrastive_loss(z, targets, y, logit_scale):
    logits = logit_scale.exp().clamp(max=100) * z @ targets[y].T                    # CLIP clamps the scale at 100
    pos = (y[:, None] == y[None, :]).float()                                        # same stimulus = positive
    l_eeg = -(logits.log_softmax(1) * pos).sum(1) / pos.sum(1)
    l_stim = -(logits.log_softmax(0) * pos).sum(0) / pos.sum(0)
    return (l_eeg.mean() + l_stim.mean()) / 2


@torch.no_grad()
def embed(model, X, slot, batch=4096, with_audio=False):
    """Visual embeddings, or (visual, audio) with with_audio=True."""
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out.append(model(X[i:i + batch], slot[i:i + batch], with_audio=with_audio))
    return (torch.cat([o[0] for o in out]), torch.cat([o[1] for o in out])) if with_audio else torch.cat(out)


def fit(model, X, slot, y, targets, params, lr, max_epochs, patience, batch, score_fn, seed, progress=None, name=None,
        audio=None):
    """Contrastive training; keeps the weights of the epoch with the highest score_fn(model).
    With progress: reports every training step (the total assumes all max_epochs; steps skipped by early stopping are
    counted at the end, so the bar reaches 100 %) and shows epoch, validation score and best score under `name`.
    With audio (dict: targets, has_audio per video row, weight): adds the audio InfoNCE of the audio head; videos
    without an audio track are left out of it. Ends at the next step when STOP is set."""
    gen = torch.Generator().manual_seed(seed)
    steps = math.ceil(len(X) / batch)
    total, warm = max_epochs * steps, WARMUP_EPOCHS * steps
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=WEIGHT_DECAY, fused=True)   # fused: one CUDA kernel per step
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda k: min(1.0, (k + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(k, total) / total)))
    best, best_state, best_epoch, wait, history, done = -math.inf, None, 0, 0, [], 0
    if progress is not None:
        progress.expect(total)
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(X), generator=gen).to(X.device)
        losses = []
        for i in range(0, len(X), batch):
            if STOP.is_set():
                raise RuntimeError('stopped')
            idx = order[i:i + batch]
            yi = y[idx]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = model(X[idx], slot[idx], with_audio=audio is not None)
            if audio is None:
                loss = contrastive_loss(out, targets, yi, model.logit_scale)
            else:
                z, za = out
                loss = contrastive_loss(z, targets, yi, model.logit_scale)
                keep = audio['has_audio'][yi]
                if keep.any():
                    loss = loss + audio['weight'] * contrastive_loss(za[keep], audio['targets'], yi[keep],
                                                                     model.audio_logit_scale)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.detach())
            done += 1
            if progress is not None:
                progress.tick()
        score = score_fn(model)
        history.append(dict(epoch=epoch, loss=float(torch.stack(losses).mean()), score=score))
        if progress is not None:
            progress.note(name, f'ep {epoch} val {score:.4f} best {max(best, score):.4f}')
        if score > best:
            best, best_epoch, wait = score, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    if progress is not None:
        progress.tick(total - done)          # steps skipped by early stopping
    model.load_state_dict(best_state)
    return dict(best_score=best, best_epoch=best_epoch, history=history)


def ranks(Z, cand, true_cols):
    S = Z @ cand.T
    return (S > S[torch.arange(len(S), device=S.device), true_cols][:, None]).sum(1) + 1


def metrics(S, true_cols):
    S = np.asarray(S)
    rank = (S > S[np.arange(len(S)), true_cols][:, None]).sum(axis=1) + 1
    return dict(n=len(rank), top1=float(np.mean(rank == 1)), top5=float(np.mean(rank <= 5)),
                mean_rank=float(rank.mean()))


def ledoit_wolf(X):
    """Shrunk covariance of samples X (n, p), float64 torch: the formula of sklearn.covariance.LedoitWolf."""
    X = X - X.mean(0)
    n, p = X.shape
    emp = X.T @ X / n
    mu = torch.trace(emp) / p
    X2 = X ** 2
    delta_ = ((X.T @ X) ** 2).sum() / n ** 2
    beta = ((X2.T @ X2).sum() / n - delta_) / (p * n)
    delta = (delta_ - 2 * mu * torch.trace(emp) + p * mu ** 2) / p
    shrinkage = 0.0 if beta == 0 else float(torch.minimum(beta, delta) / delta)
    return (1 - shrinkage) * emp + shrinkage * mu * torch.eye(p, dtype=X.dtype, device=X.device)


def normalizer(eeg_fit):
    """Whitening + per-channel z-score fitted on (n, 17, samples) float64 GPU recordings; returns a function that
    turns any recordings into float32 model inputs."""
    cov = ledoit_wolf(eeg_fit.transpose(1, 2).reshape(-1, N_CHANNELS))
    w, V = torch.linalg.eigh(cov)
    W = V @ torch.diag(w.rsqrt()) @ V.T
    white = torch.einsum('cd,ndt->nct', W, eeg_fit)[:, :, :N_SAMPLES]
    mu, sd = white.mean(dim=(0, 2), keepdim=True), white.std(dim=(0, 2), keepdim=True, correction=0)
    return lambda e: ((torch.einsum('cd,ndt->nct', W, e)[:, :, :N_SAMPLES] - mu) / sd).float()


def load_challenge(sub, split='trn'):
    if split == 'trn':
        with np.load(DATA_ROOT / 'trn' / f'sub-{sub:02d}_trn.npz') as z:
            keys = sorted(z.files, key=int)
            ids, eeg = np.array([int(k) for k in keys]), np.stack([z[k] for k in keys])
    else:
        ids, eeg = None, np.load(DATA_ROOT / 'tst' / f'sub-{sub:02d}_tst.npy')
    eeg = eeg.astype(np.float64)
    eeg -= eeg.mean(axis=2, keepdims=True)
    return ids, decimate(eeg, EEG_SFREQ // MODEL_SFREQ, axis=-1, ftype='fir', zero_phase=True)


def fold_data(fit_videos, order, query, device):
    """Model inputs of all six subjects, on the GPU. fit_videos: recordings used for fitting and for the
    normalisation. query: 'test' or an array of training videos whose recordings are returned for evaluation."""
    X, slot, vid, queries = [], [], [], {}
    for sub in SUBJECTS:
        ids, eeg = RAW[sub]
        eeg = torch.as_tensor(eeg[:, order], device=device)
        fit_rows = np.isin(ids, fit_videos)
        fit_mask = torch.as_tensor(fit_rows, device=device)
        norm = normalizer(eeg[fit_mask])
        X.append(norm(eeg[fit_mask]))
        vid.append(ids[fit_rows])
        slot.append(torch.full((int(fit_rows.sum()),), CHALLENGE_SLOT[sub], device=device))
        if isinstance(query, str):
            tst = torch.as_tensor(load_challenge(sub, 'tst')[1][:, order], device=device)
            queries[sub] = (norm(tst), None)
        else:
            q_rows = np.isin(ids, query)
            queries[sub] = (norm(eeg[torch.as_tensor(q_rows, device=device)]), ids[q_rows])
    return torch.cat(X), torch.cat(slot), np.concatenate(vid), queries


def init_model(variant, seed, device):
    audio_dim = next(iter(AUDIO_RAW.values())).shape[1] if AUDIO_HEAD and variant != 'zero-shot' else None
    model = new_model(seed, device, audio_dim)
    if variant != 'scratch':
        missing, unexpected = model.load_state_dict(
            torch.load(CKPT_DIR / f'seed{seed}.pt', map_location=device, weights_only=True), strict=False)
        assert not unexpected and set(missing) <= {'audio_head.weight', 'audio_head.bias', 'audio_logit_scale'}
        with torch.no_grad():                    # new subjects start from identity mixing and the mean THINGS token
            slots = list(CHALLENGE_SLOT.values())
            model.subject_mix[slots] = torch.eye(N_CHANNELS, device=device)
            model.subject_token[slots] = model.subject_token[THINGS_SLOTS].mean(0)
    return model


def centred(cand):
    """Candidate centring (from CORTIVA): subtract the mean target over the candidates, back to unit length. What
    all candidates share then cannot decide the ranking. Uses only the candidate videos, no EEG and no labels."""
    return F.normalize(cand - cand.mean(dim=0, keepdim=True), dim=-1)


def row_z(S):
    """Standardise every row (one recording's scores over the candidates); numpy or torch."""
    return (S - S.mean(axis=1, keepdims=True)) / S.std(axis=1, keepdims=True) if isinstance(S, np.ndarray) else \
        (S - S.mean(dim=1, keepdim=True)) / S.std(dim=1, keepdim=True)


def similarities(model, queries, cand_ids, targets, device, audio_targets=None):
    """{score: {sub: similarity matrix (recordings, candidates)}} as numpy; candidates centred. Always 'visual';
    with audio_targets also 'audio' and 'visual+audio' (sum of the row-standardised scores, equal weights)."""
    rows = torch.as_tensor(ROW_OF[cand_ids], device=device)
    cand = centred(targets[rows])
    out = {'visual': {}} | ({'audio': {}, 'visual+audio': {}} if audio_targets is not None else {})
    for sub, (x, _) in queries.items():
        slot = torch.full((len(x),), CHALLENGE_SLOT[sub], device=device)
        if audio_targets is None:
            out['visual'][sub] = (embed(model, x, slot) @ cand.T).cpu().numpy()
            continue
        zv, za = embed(model, x, slot, with_audio=True)
        sv, sa = zv @ cand.T, za @ centred(audio_targets[rows]).T
        out['visual'][sub], out['audio'][sub] = sv.cpu().numpy(), sa.cpu().numpy()
        out['visual+audio'][sub] = (row_z(sv) + row_z(sa)).cpu().numpy()
    return out


def audio_targets_for(window, fit_videos, device):
    """Audio targets of all videos: centred on the fitting videos (per fold), unit length."""
    raw = AUDIO_RAW[window]
    return F.normalize(torch.as_tensor(raw - raw[ROW_OF[fit_videos]].mean(0), device=device), dim=-1)


def train_on(variant, seed, fit_videos, order, window, query, device, progress=None, name=None):
    """Train one variant on fit_videos (minus an inner early-stopping set). Returns the query recordings, the model,
    the training summary and the audio targets (None without audio head)."""
    targets = torch.as_tensor(VIDEO_TARGETS[window], device=device)
    model = init_model(variant, seed, device)
    if variant == 'zero-shot':
        _, _, _, queries = fold_data(fit_videos, order, query, device)
        return queries, model, dict(best_epoch=0, best_score=float('nan')), None
    audio_t = audio_targets_for(window, fit_videos, device) if AUDIO_HEAD else None
    rng = np.random.default_rng(INNER_SEED)                  # same inner videos for every variant and seed
    inner = np.sort(rng.choice(fit_videos, int(round(INNER_FRACTION * len(fit_videos))), replace=False))
    X, slot, vid, queries = fold_data(fit_videos, order, query, device)
    is_inner = torch.as_tensor(np.isin(vid, inner), device=device)
    y = torch.as_tensor(ROW_OF[vid], device=device)
    inner_rows = torch.as_tensor(ROW_OF[inner], device=device)
    inner_cand = centred(targets[inner_rows])
    inner_true = torch.as_tensor(np.searchsorted(inner, vid[is_inner.cpu().numpy()]), device=device)
    Xi, si = X[is_inner], slot[is_inner]

    def score(m):   # higher = better: minus the mean rank of the inner recordings among the inner videos
        if audio_t is None:
            return -float(ranks(embed(m, Xi, si), inner_cand, inner_true).float().mean())
        zv, za = embed(m, Xi, si, with_audio=True)
        S = row_z(zv @ inner_cand.T) + row_z(za @ centred(audio_t[inner_rows]).T)
        return -float((S > S[torch.arange(len(S), device=S.device), inner_true][:, None]).sum(1).add(1).float().mean())

    head = [model.audio_head.weight, model.audio_head.bias, model.audio_logit_scale] if audio_t is not None else []
    head_ids = {id(p) for p in head}
    base = [model.subject_mix, model.subject_token] if variant == 'frozen' else \
        [p for p in model.parameters() if id(p) not in head_ids]
    lr = FT_LR if variant == 'finetune' else LR
    groups = [dict(params=base, lr=lr)] + ([dict(params=head, lr=LR)] if head else [])   # new head: normal LR
    audio = dict(targets=audio_t, has_audio=torch.as_tensor(HAS_AUDIO, device=device), weight=AUDIO_WEIGHT) \
        if audio_t is not None else None
    info = fit(model, X[~is_inner], slot[~is_inner], y[~is_inner], targets, groups, lr, FT_MAX_EPOCHS, FT_PATIENCE,
               FT_BATCH, score, seed, progress=progress, name=name, audio=audio)
    return queries, model, info, audio_t


VIDEO_TARGETS = {WINDOW: np.load(FREEZE_DIR / 'features' / 'clip_targets.npy')}
AUDIO_RAW = {WINDOW: np.load(FREEZE_DIR / 'features' / 'audio_targets_raw.npy')} if AUDIO_HEAD else {}
HAS_AUDIO = np.load(FREEZE_DIR / 'features' / 'has_audio.npy') if AUDIO_HEAD else np.ones(1102, bool)
ALL_VIDEO_IDS = np.arange(1, 1 + len(VIDEO_TARGETS[WINDOW]))
ROW_OF = np.full(ALL_VIDEO_IDS.max() + 1, -1)
ROW_OF[ALL_VIDEO_IDS] = np.arange(len(ALL_VIDEO_IDS))
ORDERS = {'recorded': np.arange(N_CHANNELS)}            # = the THINGS-EEG2 order (organisers, 2026-10-18)
RAW = {}


def load_data():
    RAW.update({sub: load_challenge(sub) for sub in SUBJECTS})


def load_frozen_model(seed, device='cuda:0'):
    audio_dim = AUDIO_RAW[WINDOW].shape[1] if AUDIO_HEAD else None
    model = new_model(seed, device, audio_dim)
    model.load_state_dict(torch.load(FREEZE_DIR / 'model' / f'seed{seed}.pt', map_location=device, weights_only=True))
    return model.eval()
