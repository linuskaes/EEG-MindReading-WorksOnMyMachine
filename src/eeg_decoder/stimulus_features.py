"""Time-resolved stimulus features for EEG encoding models (50 Hz grid over the 3-s clips).

Everything here only reads stimulus videos. Statistics (standardisation, PCA) are fitted on the
1000 *training* videos and applied unchanged to the 102 candidate videos ("unsupervised
normalisation of the video features"). Pretrained models are public and pinned by revision.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

GRID_HZ = 50
GRID_T = 150                     # 3 s at 50 Hz
FRAME_GRID_FPS = 30
FRAME_GRID_N = 90
AUDIO_SR = 16000
AUDIO_SECONDS = 3.2
TRAIN_VIDEO_IDS = np.arange(1, 1001)
ALL_VIDEO_IDS = np.arange(1, 1103)
WHISPER_BASE = {"name": "openai/whisper-base", "revision": "e37978b90ca9030d5170a5c07aadb050351a65bb"}


def video_path(stimuli_root: str | Path, video_id: int) -> Path:
    root = Path(stimuli_root)
    folder = "stimuli_train" if video_id <= 1000 else "stimuli_test"
    return root / folder / f"{video_id:04d}.mp4"


# ---------------------------------------------------------------------------
# decoding
# ---------------------------------------------------------------------------
def decode_video(path: str | Path):
    """Return (gray64 (90,64,64) uint8, rgb32 (90,32,32,3) uint8) sampled-and-held at 30 fps."""
    import av
    import cv2

    frames, times = [], []
    with av.open(str(path), metadata_errors="ignore") as container:
        stream = container.streams.video[0]
        for frame in container.decode(video=0):
            t = float(frame.pts * stream.time_base) if frame.pts is not None else len(frames) / 30.0
            frames.append(frame.to_ndarray(format="rgb24"))
            times.append(t)
    if not frames:
        raise ValueError(f"No video frames in {path}")
    times = np.asarray(times) - times[0]
    grid = np.arange(FRAME_GRID_N) / FRAME_GRID_FPS
    index = np.clip(np.searchsorted(times, grid + 1e-6, side="right") - 1, 0, len(frames) - 1)
    gray = np.empty((FRAME_GRID_N, 64, 64), np.uint8)
    rgb = np.empty((FRAME_GRID_N, 32, 32, 3), np.uint8)
    for k, i in enumerate(index):
        g = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        gray[k] = cv2.resize(g, (64, 64), interpolation=cv2.INTER_AREA)
        rgb[k] = cv2.resize(frames[i], (32, 32), interpolation=cv2.INTER_AREA)
    return gray, rgb


def decode_audio(path: str | Path) -> tuple[NDArray[np.float32], bool]:
    """Mono 16 kHz waveform, zero-padded/cropped to 3.2 s; flag whether audible audio exists."""
    import av

    n = int(AUDIO_SECONDS * AUDIO_SR)
    wave = np.zeros(n, np.float32)
    with av.open(str(path), metadata_errors="ignore") as container:
        if not container.streams.audio:
            return wave, False
        resampler = av.AudioResampler(format="flt", layout="mono", rate=AUDIO_SR)
        chunks = []
        for frame in container.decode(audio=0):
            chunks += [o.to_ndarray().reshape(-1) for o in resampler.resample(frame)]
        chunks += [o.to_ndarray().reshape(-1) for o in resampler.resample(None)]
    if not chunks:
        return wave, False
    audio = np.concatenate(chunks).astype(np.float32)[:n]
    wave[: len(audio)] = audio
    return wave, bool(np.abs(audio).max() > 1e-5)


# ---------------------------------------------------------------------------
# low-level features
# ---------------------------------------------------------------------------
def _mel_filterbank(n_mels: int = 32, n_fft: int = 512, fmin: float = 50.0, fmax: float = 8000.0):
    hz2mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)  # noqa: E731
    mel2hz = lambda m: 700.0 * (10 ** (m / 2595.0) - 1.0)  # noqa: E731
    hz = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2))
    bins = np.fft.rfftfreq(n_fft, 1.0 / AUDIO_SR)
    fb = np.zeros((n_mels, len(bins)), np.float32)
    for m in range(n_mels):
        up = (bins - hz[m]) / (hz[m + 1] - hz[m])
        down = (hz[m + 2] - bins) / (hz[m + 2] - hz[m + 1])
        fb[m] = np.maximum(0, np.minimum(up, down))
    return fb


def audio_mel_envelope(wave: NDArray[np.floating], has_audio: bool):
    """Log-mel (300, 32) and RMS envelope (300,) at 100 Hz over the first 3 s."""
    fb = _mel_filterbank()
    hop = AUDIO_SR // 100
    mel = np.zeros((300, fb.shape[0]), np.float32)
    env = np.zeros(300, np.float32)
    if not has_audio:
        return mel, env
    audio = np.pad(np.asarray(wave, np.float32), (0, 512))
    window = np.hanning(512).astype(np.float32)
    for k in range(300):
        seg = audio[k * hop:k * hop + 512] * window
        mel[k] = np.log(fb @ (np.abs(np.fft.rfft(seg)) ** 2) + 1e-8)
        env[k] = np.sqrt(np.mean(audio[k * hop:(k + 1) * hop] ** 2))
    return mel, env


def lowlevel_features(gray, rgb, mel, env, has_audio, grid: int = 4):
    """(V, 150, 99) raw low-level visual + onset + audio features (see names)."""
    gray = np.asarray(gray, np.float32) / 255.0
    rgb = np.asarray(rgb, np.float32) / 255.0
    V = gray.shape[0]
    frame_index = np.minimum((np.arange(GRID_T) / GRID_HZ * FRAME_GRID_FPS).astype(int), FRAME_GRID_N - 1)
    b = 64 // grid
    lum_grid = gray.reshape(V, FRAME_GRID_N, grid, b, grid, b).mean(axis=(3, 5)).reshape(V, FRAME_GRID_N, -1)
    lum = gray.mean(axis=(2, 3))[..., None]
    con = gray.std(axis=(2, 3))[..., None]
    diff = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    diff_grid = diff.reshape(V, FRAME_GRID_N, grid, b, grid, b).mean(axis=(3, 5)).reshape(V, FRAME_GRID_N, -1)
    diff_g = diff.mean(axis=(2, 3))[..., None]
    edge = (np.abs(np.diff(gray, axis=2)).mean(axis=(2, 3)) + np.abs(np.diff(gray, axis=3)).mean(axis=(2, 3)))[..., None]
    r, g, bl = rgb[..., 0].mean((2, 3)), rgb[..., 1].mean((2, 3)), rgb[..., 2].mean((2, 3))
    color = np.stack([r - g, bl - (r + g) / 2, (r + g + bl) / 3], axis=-1)
    sat = (rgb.max(-1) - rgb.min(-1)).mean((2, 3))[..., None]
    cut = (diff_g > 0.08).astype(np.float32)
    per_frame = np.concatenate([lum, con, lum_grid, diff_g, diff_grid, edge, color, sat, cut], axis=-1)
    feats = per_frame[:, frame_index]
    first = np.concatenate([np.ones((V, 1)), lum[:, 0], con[:, 0], edge[:, 0], lum_grid[:, 0], color[:, 0], sat[:, 0]], -1)
    onset = np.zeros((V, GRID_T, first.shape[1]), np.float32)
    onset[:, 0] = first
    mel = np.asarray(mel, np.float32)
    envl = np.log(np.asarray(env, np.float32) + 1e-4)[..., None]
    step = 100 // GRID_HZ
    mel = mel[:, : GRID_T * step].reshape(V, GRID_T, step, -1).mean(2)
    envl = envl[:, : GRID_T * step].reshape(V, GRID_T, step, -1).mean(2)
    has = np.asarray(has_audio, bool)
    if has.any():
        mel = np.where(has[:, None, None], mel, mel[has].mean((0, 1)))
        envl = np.where(has[:, None, None], envl, envl[has].mean())
    onset_a = np.maximum(np.diff(envl, axis=1, prepend=envl[:, :1]), 0)
    return np.concatenate([feats, onset, envl, onset_a, mel], axis=-1).astype(np.float32)


def standardize(feats: NDArray, video_ids: NDArray, fit_ids: Iterable[int] = TRAIN_VIDEO_IDS) -> NDArray[np.float32]:
    mask = np.isin(video_ids, np.asarray(list(fit_ids)))
    mu = feats[mask].mean(axis=(0, 1), keepdims=True)
    sd = feats[mask].std(axis=(0, 1), keepdims=True) + 1e-6
    return ((feats - mu) / sd).astype(np.float32)


# ---------------------------------------------------------------------------
# pretrained audio encoder (Whisper) hidden states on the 50 Hz grid
# ---------------------------------------------------------------------------
def whisper_hidden_states(waves: NDArray, layers: Iterable[int], *, model=WHISPER_BASE, device="cuda",
                          batch: int = 16) -> dict[int, NDArray[np.float16]]:
    """Encoder hidden states (V, 150, dim) for the requested layers (0 = conv front-end output)."""
    import torch
    from huggingface_hub import snapshot_download
    from transformers import WhisperFeatureExtractor, WhisperModel

    path = snapshot_download(model["name"], revision=model["revision"],
                             allow_patterns=["*.json", "*.safetensors", "*.txt"])
    fe = WhisperFeatureExtractor.from_pretrained(path)
    dtype = torch.float16 if device == "cuda" else torch.float32
    enc = WhisperModel.from_pretrained(path, dtype=dtype).encoder.to(device).eval()
    layers = list(layers)
    out = {li: np.zeros((len(waves), GRID_T, enc.config.d_model), np.float16) for li in layers}
    with torch.no_grad():
        for b0 in range(0, len(waves), batch):
            chunk = [np.asarray(w, np.float32) for w in waves[b0:b0 + batch]]
            x = fe(chunk, sampling_rate=AUDIO_SR, return_tensors="pt").input_features.to(device, dtype)
            hs = enc(x, output_hidden_states=True).hidden_states
            for li in layers:
                arr = hs[li][:, : int(AUDIO_SECONDS * 50) + 1].float().cpu().numpy()
                for k in range(arr.shape[0]):
                    out[li][b0 + k] = _to_grid(arr[k][: int(AUDIO_SECONDS * 50)], 50.0).astype(np.float16)
    return out


def _to_grid(h: NDArray, frame_rate: float) -> NDArray:
    t_src = (np.arange(len(h)) + 0.5) / frame_rate
    t_dst = (np.arange(GRID_T) + 0.5) / GRID_HZ
    idx = np.clip(np.searchsorted(t_src, t_dst) - 1, 0, len(h) - 2)
    w = np.clip((t_dst - t_src[idx]) / (t_src[idx + 1] - t_src[idx]), 0, 1)[:, None]
    return (1 - w) * h[idx] + w * h[idx + 1]


def pca_reduce(hidden: NDArray, has_audio: NDArray, k: int = 64, train_count: int = 1000) -> NDArray[np.float32]:
    """PCA fitted on audible training-video frames; silent clips map to zero; standardised."""
    h = np.asarray(hidden, np.float32)
    has = np.asarray(has_audio, bool)
    train = h[:train_count][has[:train_count]].reshape(-1, h.shape[-1])
    mu = train.mean(0)
    _, _, vt = np.linalg.svd(train[::4] - mu, full_matrices=False)
    z = (h - mu) @ vt[:k].T
    z[~has] = 0.0
    return standardize(z, np.arange(1, len(h) + 1), np.arange(1, train_count + 1))


# ---------------------------------------------------------------------------
# pretrained image encoders (per-frame embeddings at 10 fps)
# ---------------------------------------------------------------------------
DINOV2_BASE = {"name": "facebook/dinov2-base", "revision": "f9e44c814b77203eaa57a6bdbbd535f21ede1415"}
CLIP_B32 = {"name": "openai/clip-vit-base-patch32", "revision": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"}
EMBED_FPS = 10
EMBED_N = 30


def decode_frames_10fps(path: str | Path) -> NDArray[np.uint8]:
    """(30, H, W, 3) RGB frames sampled-and-held at 10 fps from clip onset."""
    import av

    frames, times = [], []
    with av.open(str(path), metadata_errors="ignore") as container:
        stream = container.streams.video[0]
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
            times.append(float(frame.pts * stream.time_base) if frame.pts is not None else len(times) / 30.0)
    times = np.asarray(times) - times[0]
    index = np.clip(np.searchsorted(times, np.arange(EMBED_N) / EMBED_FPS + 1e-6, side="right") - 1,
                    0, len(frames) - 1)
    return np.stack([frames[i] for i in index])


def image_embeddings(stimuli_root: str | Path, model: dict, *, device: str = "cuda",
                     video_ids: Iterable[int] = ALL_VIDEO_IDS) -> NDArray[np.float32]:
    """Per-frame embeddings (V, 30, D): CLIP image_embeds or DINOv2 pooler output."""
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoImageProcessor, AutoModel, CLIPVisionModelWithProjection

    patterns = ["*.json", "*.txt", "*.safetensors"]
    path = snapshot_download(model["name"], revision=model["revision"], allow_patterns=patterns)
    if not list(Path(path).glob("*.safetensors")):
        path = snapshot_download(model["name"], revision=model["revision"], allow_patterns=patterns + ["pytorch_model.bin"])
    processor = AutoImageProcessor.from_pretrained(path)
    dtype = torch.float16 if device == "cuda" else torch.float32
    is_clip = "clip" in model["name"]
    cls = CLIPVisionModelWithProjection if is_clip else AutoModel
    net = cls.from_pretrained(path, dtype=dtype).to(device).eval()
    out = []
    with torch.no_grad():
        for vid in video_ids:
            frames = decode_frames_10fps(video_path(stimuli_root, int(vid)))
            x = processor(images=list(frames), return_tensors="pt")["pixel_values"].to(device, dtype)
            result = net(pixel_values=x)
            emb = result.image_embeds if is_clip else result.pooler_output
            out.append(emb.float().cpu().numpy())
    return np.stack(out).astype(np.float32)


def visual_pca_grid(embeddings: NDArray, k: int = 16, train_count: int = 1000) -> NDArray[np.float32]:
    """PCA-k (fitted on training-video frames), standardised, sample-and-hold to the 50 Hz grid."""
    emb = np.asarray(embeddings, np.float32)
    train = emb[:train_count].reshape(-1, emb.shape[-1])
    mu = train.mean(0)
    _, _, vt = np.linalg.svd(train - mu, full_matrices=False)
    z = (emb - mu) @ vt[:k].T
    z = (z - z[:train_count].mean((0, 1))) / z[:train_count].std((0, 1))
    grid = np.minimum((np.arange(GRID_T) / GRID_HZ * EMBED_FPS).astype(int), emb.shape[1] - 1)
    return z[:, grid].astype(np.float32)


def audio_rich_features(waves: NDArray, has_audio: NDArray, n_mels: int = 64,
                        train_count: int = 1000) -> NDArray[np.float32]:
    """(V, 150, 128): 64-band log-mel (STFT 25 ms / 10 ms, centred) and its half-wave rectified
    temporal derivative (spectral flux), averaged to 50 Hz. Silent clips get the training mean."""
    n_fft, hop, win = 512, 160, 400
    window = np.zeros(n_fft, np.float32)
    window[(n_fft - win) // 2:(n_fft - win) // 2 + win] = np.hanning(win + 1)[:-1]
    hz2mel = lambda f: 2595 * np.log10(1 + f / 700)  # noqa: E731
    mel2hz = lambda m: 700 * (10 ** (m / 2595) - 1)  # noqa: E731
    hz = mel2hz(np.linspace(hz2mel(30.0), hz2mel(8000.0), n_mels + 2))
    bins = np.fft.rfftfreq(n_fft, 1 / AUDIO_SR)
    fb = np.zeros((n_mels, len(bins)), np.float32)
    for m in range(n_mels):
        fb[m] = np.maximum(0, np.minimum((bins - hz[m]) / (hz[m + 1] - hz[m]),
                                         (hz[m + 2] - bins) / (hz[m + 2] - hz[m + 1])))
    out = np.zeros((len(waves), GRID_T, 2 * n_mels), np.float32)
    starts = np.arange(300) * hop
    for v, wave in enumerate(waves):
        x = np.pad(np.asarray(wave, np.float32), (n_fft // 2, n_fft // 2), mode="reflect")
        frames = np.stack([x[s:s + n_fft] for s in starts]) * window
        mel = np.log(np.abs(np.fft.rfft(frames, axis=1)) ** 2 @ fb.T + 1e-6)
        flux = np.maximum(np.diff(mel, axis=0, prepend=mel[:1]), 0)
        out[v] = np.concatenate([mel, flux], 1).reshape(GRID_T, 2, -1).mean(1)
    has = np.asarray(has_audio, bool)
    out[~has] = out[:train_count][has[:train_count]].mean((0, 1))
    return standardize(out, np.arange(1, len(out) + 1), np.arange(1, train_count + 1))


LOWLEVEL_VISUAL_COLUMNS = 59   # visual (39) + onset impulses (20); the remaining 40 columns are audio


def save_meta(path: Path, **meta) -> None:
    path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
