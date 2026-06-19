# =============================================================================
# Author  : Ho San Ko
# Email   : hko@avalanche.energy
# Project : 4-Channel Music Remix Workstation
# =============================================================================

"""Loop in/out management, beat-grid quantization, and hot cues (Section 7.9).

All positions are in sample indices of the *stretched* audio so they remain
valid after the global time-stretch.  When the master BPM changes, the caller
must update loop bounds via set_loop_in / set_loop_out after the new stretch
is applied (or call clear() to start fresh).

Beat-grid quantization uses master_bpm / sample_rate to snap loop points to
the nearest beat boundary in stretched coordinates.  Because pyrubberband
pre-stretches to the master BPM, one beat = sample_rate * 60 / master_bpm
samples in the stretched buffer regardless of the original track BPM.
"""

import threading
from typing import Optional

from constants import DEFAULT_MASTER_BPM, HOT_CUES_PER_CHANNEL, SAMPLE_RATE


class LoopManager:
    """Manages loop in/out, loop activation, and hot cues for one channel.

    Attributes (Section 6 class diagram):
        _loop_in        Loop-in sample position (or None).
        _loop_out       Loop-out sample position (or None).
        _loop_active    True when the loop is currently engaged.
        _hot_cues       Four absolute sample positions (or None each).
        _lock           Protects all state.
    """

    _MIN_LOOP_SAMPLES: int = 256  # edge case 18: reject zero/near-zero loops

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        master_bpm: float = DEFAULT_MASTER_BPM,
    ) -> None:
        self._sample_rate = sample_rate
        self._master_bpm: float = master_bpm

        self._loop_in: Optional[int] = None
        self._loop_out: Optional[int] = None
        self._loop_active: bool = False
        self._hot_cues: list[Optional[int]] = [None] * HOT_CUES_PER_CHANNEL
        self._lock: threading.Lock = threading.Lock()

    # ── BPM (needed for quantization) ─────────────────────────────────────────

    def set_master_bpm(self, bpm: float) -> None:
        with self._lock:
            self._master_bpm = bpm

    # ── Loop points (Section 7.9) ─────────────────────────────────────────────

    def set_loop_in(self, playhead: int) -> None:
        """Set the loop-in point, snapped to the nearest beat.

        Steps (Section 7.9):
        1. Quantize playhead to nearest beat.
        2. Store _loop_in.
        3. If _loop_out is already set and out <= in: swap them.
        """
        with self._lock:
            q = self._quantize(playhead)
            self._loop_in = q
            if self._loop_out is not None and self._loop_out <= q:
                self._loop_in, self._loop_out = self._loop_out, q

    def set_loop_out(self, playhead: int) -> None:
        """Set loop-out point (first press) or escape loop (second press).

        Steps (Section 7.9):
        1. No-op if _loop_in is None.
        2. Quantize.
        3. First press (_loop_out is None OR loop inactive):
           - Store _loop_out.
           - If out <= in: swap.
           - Guard minimum loop length.
           - Activate loop.
        4. Second press (_loop_out set AND loop active): escape loop.
        """
        with self._lock:
            if self._loop_in is None:
                return
            q = self._quantize(playhead)

            if self._loop_out is None or not self._loop_active:
                # First press: set the out point
                loop_in = self._loop_in
                loop_out = q
                if loop_out <= loop_in:
                    loop_in, loop_out = loop_out, loop_in
                # Edge case 18: minimum loop length
                if loop_out - loop_in < self._MIN_LOOP_SAMPLES:
                    return
                self._loop_in = loop_in
                self._loop_out = loop_out
                self._loop_active = True
            else:
                # Second press: escape loop
                self._loop_active = False

    def toggle_loop(self) -> None:
        """Toggle active state without changing in/out points."""
        with self._lock:
            if self._loop_in is not None and self._loop_out is not None:
                self._loop_active = not self._loop_active

    def escape_loop(self) -> None:
        """Deactivate loop, keeping in/out points for later re-engagement."""
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
        """Return playhead wrapped back to loop_in if it crossed loop_out.

        Steps (Section 7.9): if ph >= loop_out → loop_in + (ph - loop_out).
        """
        with self._lock:
            if (
                not self._loop_active
                or self._loop_in is None
                or self._loop_out is None
            ):
                return playhead
            if playhead >= self._loop_out:
                return self._loop_in + (playhead - self._loop_out)
            return playhead

    def get_loop_length_bars(self, bpm: float, sr: int) -> float:
        """Return loop length expressed in 4/4 bars."""
        with self._lock:
            if self._loop_in is None or self._loop_out is None:
                return 0.0
            samples = self._loop_out - self._loop_in
            samples_per_beat = sr * 60.0 / bpm
            return samples / samples_per_beat / 4.0

    # ── Hot cues (Section 7.9) ────────────────────────────────────────────────

    def set_hot_cue(self, idx: int, position: int) -> None:
        """Store *position* (absolute stretched-time sample) as hot cue *idx*."""
        with self._lock:
            if 0 <= idx < HOT_CUES_PER_CHANNEL:
                self._hot_cues[idx] = position

    def recall_hot_cue(self, idx: int) -> Optional[int]:
        """Return stored hot-cue position, or None if not set."""
        with self._lock:
            return self._hot_cues[idx] if 0 <= idx < HOT_CUES_PER_CHANNEL else None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _quantize(self, sample: int) -> int:
        """Snap *sample* to the nearest beat boundary in stretched coordinates.

        In the pre-stretched audio, one beat = sample_rate * 60 / master_bpm samples,
        regardless of the original track BPM (pyrubberband normalised everything).
        """
        if self._master_bpm <= 0:
            return sample
        samples_per_beat = self._sample_rate * 60.0 / self._master_bpm
        beat_index = round(sample / samples_per_beat)
        return int(beat_index * samples_per_beat)
