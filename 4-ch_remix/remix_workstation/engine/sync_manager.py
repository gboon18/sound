# =============================================================================
# Author  : Ho San Ko
# Email   : hko@avalanche.energy
# Project : 4-Channel Music Remix Workstation
# =============================================================================

"""Drift-correcting sync manager — keeps all four channels locked to master BPM.

Architecture (Section 7.5):
- recalculate_ratios() is called whenever master BPM changes or a track is loaded.
  It re-stretches all players to the new ratio.  This blocks the worker thread
  (not the audio callback) for however long pyrubberband needs.
- advance_all() is called from the real-time audio callback every buffer tick.
  It advances each player and optionally nudges the playhead by ±1 sample to
  compensate fractional drift for non-integer ratios.
- correct_drift() implements the per-channel accumulator described in Section 7.5.

Drift rationale:
  With OFFLINE pyrubberband pre-stretching, the stretched buffer is an exact
  integer-length array: no fractional sample ever accumulates during sequential
  reads.  The drift correction therefore has no effect in normal operation but
  provides a safety net for two edge cases:
    (a) Very long sessions where floating-point accumulation in ratio bookkeeping
        could shift the apparent beat position.
    (b) Future migration to real-time (on-the-fly) stretching where fractional
        drift is unavoidable.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from engine.master_clock import MasterClock
    from engine.track_player import TrackPlayer


class SyncManager:
    """Calculates per-channel stretch ratios and applies drift correction.

    Attributes (Section 6 class diagram):
        _master_clock    Single shared MasterClock.
        _players         List of four TrackPlayer instances.
        _ratios          Current stretch ratio per channel.
        _drift_accum     Fractional-sample error accumulator per channel.
    """

    _DRIFT_THRESHOLD_SAMPLES: float = 0.5
    _CORRECTION_PERIOD_BEATS: int = 4   # only correct every N beats for non-int ratios

    def __init__(
        self,
        master_clock: "MasterClock",
        players: "list[TrackPlayer]",
    ) -> None:
        self._master_clock = master_clock
        self._players = players

        n = len(players)
        self._ratios: list[float] = [1.0] * n
        self._drift_accum: list[float] = [0.0] * n

        # Elapsed sample counter used by the drift corrector.
        # Incremented in advance_all() — audio-callback thread only; no lock needed.
        self._elapsed_samples: int = 0
        # Beat counter: elapsed_samples / samples_per_beat
        self._lock: threading.Lock = threading.Lock()

    # ── Public API (Section 6 class diagram) ─────────────────────────────────

    def recalculate_ratios(self) -> None:
        """Recompute stretch ratios and trigger re-stretch on all loaded players.

        Steps (Section 7.5):
        ratio[i] = master_bpm / track_bpm[i]
        Then call player[i].stretch(ratio[i]).

        Called from UI/MIDI thread after a BPM change or after a track load.
        The stretch itself is CPU-heavy — call this from a QThreadPool worker
        whenever possible, or accept a brief stutter.
        """
        master_bpm = self._master_clock.get_bpm()
        for i, player in enumerate(self._players):
            track_bpm = player.get_track_bpm()
            if track_bpm > 0:
                ratio = master_bpm / track_bpm
                with self._lock:
                    self._ratios[i] = ratio
                player.stretch(ratio)
            # If track_bpm == 0 (not loaded or detection failed), leave ratio as-is.

    def advance_all(self, frames: int) -> list[np.ndarray]:
        """Advance every player by *frames* samples and return their buffers.

        Called from the real-time sounddevice audio callback on every tick.
        Must be fast: no I/O, no heavy allocations.

        Steps:
        1. For each channel: apply drift nudge (±1 sample) if needed.
        2. Call player.advance(frames).
        3. Accumulate elapsed_samples counter for next drift check.

        Returns:
            List of (frames, 2) float32 arrays, one per channel.
        """
        buffers: list[np.ndarray] = []
        for i, player in enumerate(self._players):
            nudge = self.correct_drift(i)
            if nudge != 0:
                ph = player.get_playhead()
                player.set_playhead(ph + nudge)
            buffers.append(player.advance(frames))

        self._elapsed_samples += frames
        return buffers

    def correct_drift(self, channel: int) -> int:
        """Return a ±1 sample nudge if accumulated drift exceeds the threshold.

        Algorithm (Section 7.5):
        - ideal   = elapsed_samples × ratio[channel]
                    (where the playhead should be in stretched coordinates)
        - actual  = player.get_playhead()
        - delta   = ideal - actual
        - If |delta| > _DRIFT_THRESHOLD_SAMPLES (0.5): return round(delta) and
          reset the accumulator; else return 0.

        With offline pre-stretching this always returns 0, but the accumulator
        is maintained so that BPM-change transitions are corrected on the next
        beat.

        Note: No lock is needed here — called only from the audio-callback thread,
        the same thread that drives _elapsed_samples.

        Args:
            channel: Channel index 0–3.
        Returns:
            Integer sample nudge (typically 0, occasionally ±1).
        """
        if channel >= len(self._players):
            return 0

        with self._lock:
            ratio = self._ratios[channel]

        # Skip correction for near-integer ratios (no drift possible)
        fractional_part = abs(ratio - round(ratio))
        if fractional_part < 1e-6:
            self._drift_accum[channel] = 0.0
            return 0

        ideal = self._elapsed_samples * ratio
        actual = self._players[channel].get_playhead()
        delta = ideal - actual
        self._drift_accum[channel] += delta - round(delta)  # fractional remainder

        if abs(self._drift_accum[channel]) >= self._DRIFT_THRESHOLD_SAMPLES:
            nudge = int(round(self._drift_accum[channel]))
            self._drift_accum[channel] -= nudge
            return nudge

        return 0

    def get_ratio(self, channel: int) -> float:
        """Return current stretch ratio for *channel*."""
        with self._lock:
            return self._ratios[channel] if channel < len(self._ratios) else 1.0

    def reset_elapsed(self) -> None:
        """Reset the elapsed-samples counter (call when all players are stopped/rewound)."""
        self._elapsed_samples = 0
        self._drift_accum = [0.0] * len(self._players)
