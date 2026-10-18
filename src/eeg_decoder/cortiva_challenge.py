from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np
from numpy.typing import NDArray
EXPECTED_CHANNELS = 17
EXPECTED_SAMPLES = 1500
EXPECTED_CANDIDATE_IDS = tuple(range(1001, 1103))

@dataclass(frozen=True)
class ChallengeTrainingSubject:
    """Training-only view that cannot expose test recordings or labels."""
    subject_id: str
    train_eeg: NDArray[np.float32]
    train_video_ids: NDArray[np.int64]

@dataclass(frozen=True)
class ChallengeTestSubject:
    """Scoring-only view that cannot expose training EEG or test labels."""
    subject_id: str
    test_eeg: NDArray[np.float32]

def load_challenge_training_subject(root: str | Path, subject_id: str, *, expected_channels: int=EXPECTED_CHANNELS, expected_samples: int=EXPECTED_SAMPLES) -> ChallengeTrainingSubject:
    """Load only official training EEG; never resolve or open test paths."""
    if not subject_id.startswith('sub-'):
        raise ValueError(f'Invalid subject ID: {subject_id}')
    train_path = Path(root) / 'trn' / f'{subject_id}_trn.npz'
    with np.load(train_path, allow_pickle=False) as archive:
        try:
            ordered_keys = sorted(archive.files, key=lambda value: int(value))
        except ValueError as error:
            raise ValueError(f'Training keys for {subject_id} must be integer video IDs') from error
        if not ordered_keys:
            raise ValueError(f'Training archive for {subject_id} is empty')
        train_video_ids = np.asarray([int(key) for key in ordered_keys], dtype=np.int64)
        if len(np.unique(train_video_ids)) != len(train_video_ids):
            raise ValueError(f'Training video IDs for {subject_id} must be unique')
        train_eeg = np.stack([np.asarray(archive[key], dtype=np.float32) for key in ordered_keys])
    expected_trial_shape = (expected_channels, expected_samples)
    if train_eeg.shape[1:] != expected_trial_shape:
        raise ValueError(f'Invalid training EEG shape for {subject_id}: {train_eeg.shape}; expected (trials, {expected_channels}, {expected_samples})')
    _require_finite(train_eeg, f'training EEG for {subject_id}')
    return ChallengeTrainingSubject(subject_id=subject_id, train_eeg=train_eeg, train_video_ids=train_video_ids)

def load_challenge_test_subject(root: str | Path, subject_id: str, *, expected_channels: int=EXPECTED_CHANNELS, expected_samples: int=EXPECTED_SAMPLES, expected_test_trials: int=102) -> ChallengeTestSubject:
    """Load only test EEG for inference; never resolve training or label paths."""
    if not subject_id.startswith('sub-'):
        raise ValueError(f'Invalid subject ID: {subject_id}')
    test_path = Path(root) / 'tst' / f'{subject_id}_tst.npy'
    test_eeg = np.asarray(np.load(test_path, allow_pickle=False), dtype=np.float32)
    expected_shape = (expected_test_trials, expected_channels, expected_samples)
    if test_eeg.shape != expected_shape:
        raise ValueError(f'Invalid test EEG shape for {subject_id}: {test_eeg.shape}; expected {expected_shape}')
    _require_finite(test_eeg, f'test EEG for {subject_id}')
    return ChallengeTestSubject(subject_id=subject_id, test_eeg=test_eeg)

def _require_finite(values: NDArray[np.floating], label: str) -> None:
    if not np.isfinite(values).all():
        raise ValueError(f'{label} must contain only finite values')
