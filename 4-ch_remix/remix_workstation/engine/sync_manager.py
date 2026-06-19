# =============================================================================
# Author  : Ho San Ko
# Email   : hko@avalanche.energy
# Project : 4-Channel Music Remix Workstation
# =============================================================================

"""Drift-correcting sync manager — keeps all four channels locked to master BPM."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engine.master_clock import MasterClock
    from engine.track_player import TrackPlayer


class SyncManager:
    """Calculates per-channel stretch ratios and applies drift correction.

    Attributes (Section 6 class diagram):
        _master_clock   Reference to the single MasterClock instance.
        _players        List of four TrackPlayer instances.
        _ratios         Current stretch ratio per channel.
        _drift_accum    Fractional-sample drift accumulator per channel.
    """

    _DRIFT_THRESHOLD_SAMPLES: float = 0.5
    _CORRECTION_PERIOD_BEATS: int = 4

    def __init__(
        self,
        master_clock: "MasterClock",
        players: "list[TrackPlayer]",
    ) -> None:
        self._master_clock = master_clock
        self._players = players
        self._ratios: list[float] = [1.0] * len(players)
        self._drift_accum: list[float] = [0.0] * len(players)
        self._beat_counter: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def recalculate_ratios(self) -> None:
        """Recompute stretch ratios for all channels and trigger re-stretch.

        Steps (Section 7.5):
        ratio[i] = master_bpm / track_bpm[i]
        Then call player[i].stretch(ratio[i]).
        """
        master_bpm = self._master_clock.get_bpm()
        for i, player in enumerate(self._players):
            track_bpm = player.get_track_bpm()
            if track_bpm > 0:
                self._ratios[i] = master_bpm / track_bpm
                player.stretch(self._ratios[i])

    def advance_all(self, frames: int) -> None:
        """Advance all players by *frames* samples (called from audio callback)."""
        raise NotImplementedError

    def correct_drift(self, channel: int) -> int:
        """Return a sample nudge for *channel* to compensate accumulated drift.

        Steps (Section 7.5):
        - Compute delta = ideal_pos - actual_pos.
        - If |delta| > threshold: return round(delta), reset accumulator.
        - Otherwise: return 0.
        """
        raise NotImplementedError

    def get_ratio(self, channel: int) -> float:
        """Return current stretch ratio for *channel*."""
        return self._ratios[channel]
