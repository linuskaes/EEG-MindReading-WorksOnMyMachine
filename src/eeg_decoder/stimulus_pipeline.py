"""Stimulus feature construction for the final v7 configuration."""
from pathlib import Path
import numpy as np
from .stimulus_features import (ALL_VIDEO_IDS, TRAIN_VIDEO_IDS, LOWLEVEL_VISUAL_COLUMNS,
    WHISPER_BASE, DINOV2_BASE, CLIP_B32, audio_mel_envelope, audio_rich_features,
    decode_audio, decode_video, image_embeddings, lowlevel_features, pca_reduce,
    standardize, video_path, visual_pca_grid, whisper_hidden_states)
IMAGE_MODELS = {"dinov2-base": DINOV2_BASE, "clip-b32": CLIP_B32}

def build_features(data_root: Path, cache: Path, spec: dict, device: str, log=print) -> np.ndarray:
    """Feature bank for one family. spec keys: lowlevel ('full' | 'visual'), whisper_layers, pca,
    visual [{model, k}], audio_rich (bool)."""
    cache.mkdir(parents=True, exist_ok=True)
    raw_path = cache / "stimulus-raw.npz"
    if not raw_path.exists():
        log("decoding 1102 stimulus videos ...")
        gray, rgb, waves, has = [], [], [], []
        for vid in ALL_VIDEO_IDS:
            path = video_path(data_root, int(vid))
            g, r = decode_video(path)
            w, h = decode_audio(path)
            gray.append(g); rgb.append(r); waves.append(w); has.append(h)
        np.savez(raw_path, gray=np.stack(gray), rgb=np.stack(rgb), wave=np.stack(waves), has_audio=np.asarray(has))
    raw = np.load(raw_path)
    has = raw["has_audio"]
    mel_env = [audio_mel_envelope(w, h) for w, h in zip(raw["wave"], has)]
    low = lowlevel_features(raw["gray"], raw["rgb"], np.stack([m for m, _ in mel_env]),
                            np.stack([e for _, e in mel_env]), has)
    low = standardize(low, ALL_VIDEO_IDS, TRAIN_VIDEO_IDS)
    parts = [low[..., :LOWLEVEL_VISUAL_COLUMNS] if spec.get("lowlevel", "full") == "visual" else low]
    layers = spec["whisper_layers"]
    wpath = cache / f"whisper-base-{'-'.join(map(str, layers))}.npz"
    if not wpath.exists():
        log("Whisper-base hidden states ...")
        hs = whisper_hidden_states(raw["wave"], layers, device=device)
        np.savez(wpath, **{f"layer{li}": h for li, h in hs.items()})
    hs = np.load(wpath)
    for li in layers:
        parts.append(pca_reduce(hs[f"layer{li}"], has, k=spec["pca"]))
    if spec.get("audio_rich"):
        apath = cache / "audio-rich.npy"
        if not apath.exists():
            np.save(apath, audio_rich_features(raw["wave"], has))
        parts.append(np.load(apath))
    for item in spec.get("visual", []):
        spec = IMAGE_MODELS[item["model"]]
        vpath = cache / f"{item['model']}-10fps.npy"
        if not vpath.exists():
            log(f"{spec['name']} frame embeddings ...")
            np.save(vpath, image_embeddings(data_root, spec, device=device).astype(np.float16))
        parts.append(visual_pca_grid(np.load(vpath).astype(np.float32), k=item.get("k", 16)))
    return np.concatenate(parts, -1).astype(np.float32)
