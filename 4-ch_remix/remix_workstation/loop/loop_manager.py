# =============================================================================
# Author  : Ho San Ko
# Email   : hko@avalanche.energy
# Project : 4-Channel Music Remix Workstation
# =============================================================================

"""Loop in/out management, hot cues, and playhead wrapping (Section 7.9)."""

import threading
from typing import Optional

from constants import HOT_CUES_PER_CHANNEL, SAMPLE_RATE


class LoopManager:
    """Manages loop boundaries and hot cues for one channel.

    All positions are in sample indices of the *stretched* audio so they
    remain valid across BPM changes as long as the stretch is reapplied.

    Attributes (Section 6 class diagram):
        _loop_in        Loop-in point (sample index) or None.
        _loop_out       Loop-out point (sample index) or None.
        _loop_active    True when the loop is engaged.
        _hot_cues       List of 4 absolute sample positions (or None).
        _lock           Protects all mutable state.
    """

    _MIN_LOOP_SAMPLES: int = 512  # guard: zero-length loop (edge case 18)

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self._sample_rate = sample_rate
        self._loop_in: Optional[int] = None
        self._loop_out: Optional[int] = None
        self._loop_active: bool = False
        self._hot_cues: list[Optional[int]] = [None] * HOT_CUES_PER_CHANNEL
        self._lock: threading.Lock = threading.Lock()

    # ── Loop points ───────────────────────────────────────────────────────────

    def set_loop_in(self, playhead: int) -> None:
        """Set the loop-in point, quantized to the nearest beat.

        Steps (Section 7.9):
        1. Quantize to beat grid.
        2. If loop_out is set and loop_out <= quantized_in, swap.
        """
        raise NotImplementedError

    def set_loop_out(self, playhead: int) -> None:
        """Set the loop-out point and optionally activate/escape the loop.

        Steps (Section 7.9):
        - First press: set out, auto-swap if out <= in, activate loop.
        - Second press (loop already active): escape loop.
        """
        raise NotImplementedError

    def toggle_loop(self) -> None:
        """Toggle loop active state without changing the in/out points."""
        with self._lock:
            if self._loop_in is not None and self._loop_out is not None:
                self._loop_active = not self._loop_active

    def escape_loop(self) -> None:
        """Deactivate the loop without clearing the in/out points."""
        with self._lock:
            self._loop_active = False

    def is_loop_active(self) -> bool:
        with self._lock:
            return self._loop_active

    def get_loop_bounds(self) -> Optional[tuple[int, int]]:
        """Return (loop_in, loop_out) or None if not both set."""
        with self._lock:
            if self._loop_in is not None and self._loop_out is not None:
                return (self._loop_in, self._loop_out)
            return None

    def wrap_playhead(self, playhead: int) -> int:
        """Return the wrapped playhead position if loop is active.

        Steps (Section 7.9): if playhead >= loop_out → loop_in + (ph - loop_out).
        """
        with self._lock:
            if not self._loop_active or self._loop_in is None or self._loop_out is None:
                return playhead
            if playhead >= self._loop_out:
                return self._loop_in + (playhead - self._loop_out)
            return playhead

    def get_loop_length_bars(self, bpm: float, sr: int) -> float:
        """Return the loop length expressed in bars (4/4 time)."""
        with self._lock:
            if self._loop_in is None or self._loop_out is None:
                return 0.0
            samples = self._loop_out - self._loop_in
            samples_per_beat = sr * 60.0 / bpm
            beats = samples / samples_per_beat
            return beats / 4.0  # 4 beats per bar

    # ── Hot cues ──────────────────────────────────────────────────────────────

    def set_hot_cue(self, idx: int, position: int) -> None:
        """Store an absolute stretched-time position as hot cue *idx*."""
        with self._lock:
            self._hot_cues[idx] = position

    def recall_hot_cue(self, idx: int) -> Optional[int]:
        """Return the stored hot cue position, or None if not set."""
        with self._lock:
            return self._hot_cues[idx]
