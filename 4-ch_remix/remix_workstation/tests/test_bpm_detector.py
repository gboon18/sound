# =============================================================================
# Author  : Ho San Ko
# Email   : hko@avalanche.energy
# Project : 4-Channel Music Remix Workstation
# =============================================================================

"""Unit tests for BpmDetector (Section 11 test plan)."""

import numpy as np
import pytest

from engine.bpm_detector import BpmDetector

SAMPLE_RATE = 44100
DETECTOR = BpmDetector()


def _click_track(bpm: float, duration_s: float = 10.0) -> np.ndarray:
    """Generate a simple click track at *bpm*."""
    sr = SAMPLE_RATE
    n = int(duration_s * sr)
    audio = np.zeros((n, 2), dtype=np.float32)
    beat_samples = int(sr * 60.0 / bpm)
    for i in range(0, n, beat_samples):
        if i < n:
            audio[i, :] = 1.0
    return audio


def test_detect_120_bpm():
    """Known 120 BPM click track should be detected within ±2 BPM."""
    audio = _click_track(120.0)
    result = DETECTOR.detect(audio, SAMPLE_RATE)
    assert result is not None, "Detection returned None for a clear click track"
    assert abs(result - 120.0) <= 2.0, f"Detected {result:.1f}, expected 120 ± 2 BPM"


def test_silence_returns_none():
    """Silence input should return None (no BPM detectable)."""
    audio = np.zeros((SAMPLE_RATE * 5, 2), dtype=np.float32)
    result = DETECTOR.detect(audio, SAMPLE_RATE)
    assert result is None, f"Expected None for silence, got {result}"


def test_noise_returns_none_or_valid():
    """White noise should return None or a value in [40, 300]."""
    rng = np.random.default_rng(42)
    audio = rng.standard_normal((SAMPLE_RATE * 5, 2)).astype(np.float32)
    result = DETECTOR.detect(audio, SAMPLE_RATE)
    if result is not None:
        assert 40.0 <= result <= 300.0, f"Noise detection out of range: {result}"
