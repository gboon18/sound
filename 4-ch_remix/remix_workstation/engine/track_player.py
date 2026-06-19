# =============================================================================
# Author  : Ho San Ko
# Email   : hko@avalanche.energy
# Project : 4-Channel Music Remix Workstation
# =============================================================================

"""Per-channel audio player — loads, stretches, and streams audio chunks."""

import threading
from pathlib import Path
from typing import Optional

import numpy as np

from constants import SAMPLE_RATE, SUPPORTED_FORMATS


class TrackPlayer:
    """Holds raw + stretched audio for one channel and maintains the playhead.

    Attributes (Section 6 class diagram):
        _audio_data      Raw PCM float32 stereo array from disk.
        _stretched_data  Time-stretched copy at the current ratio.
        _playhead        Current read position in stretched_data (sample index).
        _track_bpm       BPM detected (or manually set) for this track.
        _playing         Playback state flag.
        _lock            Protects _stretched_data, _playhead, _playing.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, buffer_size: int = 512) -> None:
        self._sample_rate = sample_rate
        self._buffer_size = buffer_size

        self._audio_data: Optional[np.ndarray] = None       # (N, 2) float32
        self._stretched_data: Optional[np.ndarray] = None   # (M, 2) float32
        self._playhead: int = 0
        self._track_bpm: float = 0.0
        self._playing: bool = False
        self._lock: threading.Lock = threading.Lock()

        self._stretch_ratio: float = 1.0

    # ── File loading ──────────────────────────────────────────────────────────

    def load_file(self, path: str) -> None:
        """Load an audio file, detect BPM, and perform initial time-stretch.

        Steps (Section 7.3):
        1. Read via soundfile → float32 stereo.
        2. Resample if sr != SAMPLE_RATE.
        3. Detect BPM via BpmDetector.
        4. Stretch to master BPM ratio.
        """
        raise NotImplementedError

    def stretch(self, ratio: float) -> None:
        """Re-stretch the raw audio to *ratio* and update loop boundaries.

        Steps (Section 7.3):
        1. pyrubberband.time_stretch(audio_data, sr, ratio).
        2. Store as _stretched_data.
        3. Recalculate beat grid and loop boundary positions.
        """
        raise NotImplementedError

    # ── Playback ──────────────────────────────────────────────────────────────

    def advance(self, frames: int) -> np.ndarray:
        """Return the next *frames* samples and advance the playhead.

        Returns silence (zeros) if no track is loaded or playback is paused.
        Handles loop crossfade at the loop boundary (2 ms, ~88 samples).
        """
        with self._lock:
            if not self._playing or self._stretched_data is None:
                return np.zeros((frames, 2), dtype=np.float32)
            raise NotImplementedError

    def set_playhead(self, sample: int) -> None:
        """Jump the playhead to an absolute sample position in stretched coords."""
        with self._lock:
            if self._stretched_data is not None:
                self._playhead = max(0, min(sample, len(self._stretched_data) - 1))

    def get_playhead(self) -> int:
        with self._lock:
            return self._playhead

    # ── Transport ─────────────────────────────────────────────────────────────

    def play(self) -> None:
        with self._lock:
            self._playing = True

    def pause(self) -> None:
        with self._lock:
            self._playing = False

    def stop(self) -> None:
        with self._lock:
            self._playing = False
            self._playhead = 0

    # ── BPM ───────────────────────────────────────────────────────────────────

    def get_track_bpm(self) -> float:
        return self._track_bpm

    def set_track_bpm(self, bpm: float) -> None:
        self._track_bpm = bpm
