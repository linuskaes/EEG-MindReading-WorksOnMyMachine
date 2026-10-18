from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

av = pytest.importorskip("av")
pytest.importorskip("cv2")

from eeg_decoder.stimulus_features import (EMBED_N, FRAME_GRID_N, decode_audio, decode_frames_10fps,  # noqa: E402
                                           decode_video, visual_pca_grid)


def _write_video(path: Path, frames: int = 45, fps: int = 15) -> None:
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 48, 48, "yuv420p"
        for k in range(frames):
            img = np.full((48, 48, 3), (k * 5) % 255, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_decode_video_samples_and_holds_on_fixed_grids(tmp_path):
    path = tmp_path / "clip.mp4"
    _write_video(path)
    gray, rgb = decode_video(path)
    assert gray.shape == (FRAME_GRID_N, 64, 64) and rgb.shape == (FRAME_GRID_N, 32, 32, 3)
    # brightness increases over time; sample-and-hold keeps the sequence monotone non-decreasing
    means = gray.reshape(FRAME_GRID_N, -1).mean(1)
    assert means[-1] > means[0] and np.all(np.diff(means) >= -2)
    frames = decode_frames_10fps(path)
    assert frames.shape[0] == EMBED_N and frames.shape[-1] == 3
    wave, has_audio = decode_audio(path)
    assert wave.shape == (51200,) and not has_audio and not wave.any()


def test_visual_pca_grid_fits_on_training_rows():
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((12, EMBED_N, 8)).astype(np.float32)
    out = visual_pca_grid(emb, k=3, train_count=10)
    assert out.shape == (12, 150, 3)
    np.testing.assert_allclose(out[:10].mean((0, 1)), 0, atol=1e-4)
    shifted = emb.copy(); shifted[10:] += 50
    np.testing.assert_allclose(visual_pca_grid(shifted, k=3, train_count=10)[:10], out[:10], atol=1e-4)
