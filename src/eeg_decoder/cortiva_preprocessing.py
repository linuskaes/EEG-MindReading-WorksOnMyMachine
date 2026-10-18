"""Final internal FFT band-pass with mirror padding."""
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

def bandpass_trials(
    eeg: NDArray[np.floating],
    *,
    sample_rate_hz: float,
    low_hz: float,
    high_hz: float,
    low_transition_hz: float = 0.5,
    high_transition_hz: float = 4.0,
    pad_seconds: float = 0.5,
    chunk_size: int = 128,
) -> NDArray[np.float32]:
    """Zero-phase FFT band-pass with raised-cosine edges and mirror padding.

    ``low_hz == 0`` gives a pure low-pass. Mirror padding limits edge artefacts
    on the short three-second epochs.
    """

    values = np.asarray(eeg, dtype=np.float32)
    if values.ndim != 3 or values.shape[2] < 4:
        raise ValueError("EEG must have shape (trials, channels, samples>=4)")
    nyquist = sample_rate_hz / 2.0
    if sample_rate_hz <= 0 or not 0 <= low_hz < high_hz < nyquist:
        raise ValueError("Band edges must satisfy 0 <= low < high < Nyquist")
    if low_transition_hz <= 0 or high_transition_hz <= 0 or pad_seconds < 0 or chunk_size <= 0:
        raise ValueError("Transitions and chunk_size must be positive, pad_seconds non-negative")
    if not np.isfinite(values).all():
        raise ValueError("EEG must contain only finite values")
    samples = values.shape[2]
    pad = min(samples - 1, int(round(pad_seconds * sample_rate_hz)))
    length = samples + 2 * pad
    frequencies = np.fft.rfftfreq(length, d=1.0 / sample_rate_hz)
    gain = _raised_cosine_gain(frequencies, low_hz, high_hz, low_transition_hz, high_transition_hz)
    output = np.empty_like(values)
    for start in range(0, len(values), chunk_size):
        chunk = values[start:start + chunk_size].astype(np.float64)
        padded = np.pad(chunk, ((0, 0), (0, 0), (pad, pad)), mode="reflect") if pad else chunk
        filtered = np.fft.irfft(np.fft.rfft(padded, axis=2) * gain, n=length, axis=2)
        output[start:start + chunk_size] = filtered[:, :, pad:pad + samples]
    return output

def _raised_cosine_gain(
    frequencies: NDArray[np.float64], low: float, high: float, low_width: float, high_width: float,
) -> NDArray[np.float64]:
    gain = np.ones_like(frequencies)
    if low > 0:
        start, stop = max(low - low_width / 2, 0.0), low + low_width / 2
        gain[frequencies <= start] = 0.0
        ramp = (frequencies > start) & (frequencies < stop)
        gain[ramp] = 0.5 - 0.5 * np.cos(np.pi * (frequencies[ramp] - start) / (stop - start))
    start, stop = high - high_width / 2, high + high_width / 2
    ramp = (frequencies > start) & (frequencies < stop)
    gain[ramp] *= 0.5 + 0.5 * np.cos(np.pi * (frequencies[ramp] - start) / (stop - start))
    gain[frequencies >= stop] = 0.0
    return gain
