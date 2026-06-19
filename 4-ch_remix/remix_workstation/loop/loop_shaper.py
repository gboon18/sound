# =============================================================================
# Author  : Ho San Ko
# Email   : hko@avalanche.energy
# Project : 4-Channel Music Remix Workstation
# =============================================================================

"""Loop Shaper — gesture automation recorder/replayer (Section 7.10).

The centrepiece creative feature: records knob movements mapped to the loop
timeline (0.0–1.0 normalized position) and replays them in sync every loop.

Bug-fix log vs. initial scaffold:
  1. rescale(): clamp-then-filter was a no-op — fixed to multiply-then-discard.
  2. _overdub_point(): appended at end (unsorted) — fixed with bisect.insort().
  3. arm_record() / arm_overdub() shared ARMED state — added _overdub_pending
     flag so on_loop_wrap() knows which transition to make.
"""

from __future__ import annotations

import bisect
import threading
from typing import TYPE_CHECKING, Optional

from constants import (
    AutomationMode,
    LoopShaperState,
    Param,
    PARAM_RANGES,
)

if TYPE_CHECKING:
    from loop.loop_manager import LoopManager


class LoopShaper:
    """Records, overdubs, and replays parameter automation for one channel.

    Data structure (Section 7.10):
        _lanes: dict[Param, list[tuple[float, float]]]
            Key   → Param enum
            Value → SORTED list of (normalized_loop_position, value)
                    norm_pos: 0.0 = loop start, 1.0 = loop end

    Attributes (Section 6 class diagram):
        _state           Current LoopShaperState.
        _mode            AutomationMode (ABSOLUTE or ADDITIVE).
        _lanes           Automation data per parameter (always sorted by norm_pos).
        _touched_params  Params modified in the current record/overdub pass.
        _loop_manager    Reference to the channel's LoopManager.
        _lock            Protects _lanes, _state, _mode, _touched_params.
    """

    # ±0.005 normalised window: replace existing overdub points within this radius
    _OVERDUB_TOLERANCE: float = 0.005
    # Minimum inter-point gap in seconds (edge case 19: rate-limit dense recordings)
    _RECORD_RATE_LIMIT_S: float = 0.001
    # Hard cap on lane size (edge case 19)
    _MAX_LANE_POINTS: int = 10_000

    def __init__(self, loop_manager: "LoopManager") -> None:
        self._loop_manager = loop_manager
        self._state: LoopShaperState = LoopShaperState.IDLE
        self._mode: AutomationMode = AutomationMode.ABSOLUTE
        self._lanes: dict[Param, list[tuple[float, float]]] = {p: [] for p in Param}
        self._touched_params: set[Param] = set()
        self._lock: threading.Lock = threading.Lock()
        self._last_record_time: dict[Param, float] = {}
        # Distinguishes arm_record() from arm_overdub() while in ARMED state
        self._overdub_pending: bool = False

    # ── ARMED → RECORDING / OVERDUBBING transition ────────────────────────────

    def on_loop_wrap(self) -> None:
        """Called by the audio callback each time the playhead crosses loop_out.

        Triggers the ARMED → RECORDING or ARMED → OVERDUBBING transition so
        recording always starts clean at the loop boundary (edge case 13).
        """
        with self._lock:
            if self._state != LoopShaperState.ARMED:
                return
            if self._overdub_pending:
                self._state = LoopShaperState.OVERDUBBING
                self._touched_params.clear()
                self._last_record_time.clear()
            else:
                self._state = LoopShaperState.RECORDING
                self._touched_params.clear()
                self._last_record_time.clear()
                self._clear_all_lanes()

    # ── Record API ────────────────────────────────────────────────────────────

    def arm_record(self) -> None:
        """Queue a fresh record to begin at the next loop wrap.

        Clears existing lanes immediately so they don't bleed into the
        interval between arm and the actual start.
        """
        with self._lock:
            self._overdub_pending = False
            self._state = LoopShaperState.ARMED
            self._clear_all_lanes()

    def start_record(self) -> None:
        """Begin recording now (use arm_record() for loop-boundary alignment)."""
        with self._lock:
            self._overdub_pending = False
            self._state = LoopShaperState.RECORDING
            self._touched_params.clear()
            self._last_record_time.clear()
            self._clear_all_lanes()

    def stop_record(self) -> None:
        """Finalise: sort all lanes and transition to PLAYING."""
        with self._lock:
            for lane in self._lanes.values():
                lane.sort(key=lambda pt: pt[0])
            self._state = LoopShaperState.PLAYING

    def record_point(
        self,
        param: Param,
        norm_pos: float,
        value: float,
        *,
        force: bool = False,
    ) -> None:
        """Append (norm_pos, value) to *param*'s lane.

        Steps (Section 7.10):
        1. Guard: state must be RECORDING or OVERDUBBING.
        2. Rate-limit: skip if < 1 ms since last point for this param
           (edge case 19 — prevents memory bloat from rapid knob movement).
           Pass force=True to bypass the rate-limit (used by unit tests).
        3. For RECORDING: append (lanes are sorted at stop_record()).
        4. For OVERDUBBING: replace points within ±tolerance via _overdub_point()
           which maintains sorted order with bisect.insort().
        """
        import time
        with self._lock:
            if self._state not in (LoopShaperState.RECORDING, LoopShaperState.OVERDUBBING):
                return
            if not force:
                now = time.perf_counter()
                if param in self._last_record_time:
                    if now - self._last_record_time[param] < self._RECORD_RATE_LIMIT_S:
                        return
                self._last_record_time[param] = now

            if self._state == LoopShaperState.OVERDUBBING:
                self._overdub_point(param, norm_pos, value)
            else:
                if len(self._lanes[param]) < self._MAX_LANE_POINTS:
                    self._lanes[param].append((norm_pos, value))
            self._touched_params.add(param)

    # ── Overdub API ───────────────────────────────────────────────────────────

    def arm_overdub(self) -> None:
        """Queue an overdub pass to begin at the next loop wrap.

        Existing lanes are preserved — only touched params will be modified.
        """
        with self._lock:
            self._overdub_pending = True
            self._state = LoopShaperState.ARMED

    def start_overdub(self) -> None:
        """Begin overdub now (use arm_overdub() for loop-boundary alignment)."""
        with self._lock:
            self._overdub_pending = True
            self._state = LoopShaperState.OVERDUBBING
            self._touched_params.clear()
            self._last_record_time.clear()

    # ── Evaluate (called from audio callback) ─────────────────────────────────

    def evaluate(self, norm_pos: float) -> dict[Param, float]:
        """Return interpolated automation values at *norm_pos* for all active lanes.

        Steps (Section 7.10):
        1. Guard: only active in PLAYING, RECORDING, or OVERDUBBING states.
        2. Binary search + linear interpolation per lane (O(log n) each).
        3. Returns raw interpolated values; caller applies ABSOLUTE/ADDITIVE logic.

        Performance budget (Section 12): < 0.1 ms per channel per tick.
        """
        with self._lock:
            if self._state not in (
                LoopShaperState.PLAYING,
                LoopShaperState.RECORDING,
                LoopShaperState.OVERDUBBING,
            ):
                return {}

            result: dict[Param, float] = {}
            for param, lane in self._lanes.items():
                if not lane:
                    continue
                result[param] = _interpolate(lane, norm_pos)
            return result

    # ── Lane management ───────────────────────────────────────────────────────

    def clear_lane(self, param: Param) -> None:
        with self._lock:
            self._lanes[param] = []

    def clear_all(self) -> None:
        with self._lock:
            self._clear_all_lanes()
            self._state = LoopShaperState.IDLE

    def rescale(self, old_length: float, new_length: float) -> None:
        """Proportionally rescale all norm_pos values after a loop-length change.

        Algorithm (Section 7.10):
            scale_factor = old_length / new_length
            new_pos = old_pos * scale_factor

        SHRINK (new < old, scale > 1): points near the END of the old loop
            get new_pos > 1.0 and are DISCARDED — they fall outside the shorter loop.
            Points in the first (new_length / old_length) fraction expand toward 1.0.

        EXPAND (new > old, scale < 1): all points compress toward 0.0 and
            remain in [0.0, scale] ≤ 1.0, so nothing is discarded.
            The tail region (scale … 1.0) has no automation (flat at last value).

        Critical fix vs. scaffold:  do NOT clamp before filtering.
            The original code did max(0, min(1, pos*scale)) which prevented any
            point from ever being discarded.  The correct order is:
                multiply → compare → discard if outside [0, 1] → sort.
        """
        if new_length <= 0:
            return
        scale = old_length / new_length
        with self._lock:
            for param in self._lanes:
                rescaled: list[tuple[float, float]] = [
                    (pos * scale, val)
                    for pos, val in self._lanes[param]
                ]
                # Discard out-of-range; do NOT clamp (clamping would hide
                # the discard and stack points at the boundary incorrectly)
                self._lanes[param] = [
                    (pos, val) for pos, val in rescaled if 0.0 <= pos <= 1.0
                ]
                self._lanes[param].sort(key=lambda pt: pt[0])

    def set_mode(self, mode: AutomationMode) -> None:
        with self._lock:
            self._mode = mode

    def get_mode(self) -> AutomationMode:
        with self._lock:
            return self._mode

    def get_lane(self, param: Param) -> list[tuple[float, float]]:
        with self._lock:
            return list(self._lanes[param])

    def has_automation(self, param: Param) -> bool:
        with self._lock:
            return bool(self._lanes[param])

    def get_state(self) -> LoopShaperState:
        with self._lock:
            return self._state

    # ── Internal ──────────────────────────────────────────────────────────────

    def _clear_all_lanes(self) -> None:
        for param in Param:
            self._lanes[param] = []

    def _overdub_point(self, param: Param, norm_pos: float, value: float) -> None:
        """Remove points within ±tolerance of norm_pos, then insert the new one.

        Overdub merge algorithm (Section 7.10 / edge case 3):
        - Strip all existing points whose norm_pos is within _OVERDUB_TOLERANCE
          of the incoming norm_pos.  This replaces the gesture in that window
          while leaving the rest of the lane intact.
        - Insert the new (norm_pos, value) pair in sorted position using
          bisect.insort() so the lane remains sorted for evaluate().

        Called under _lock — do NOT re-acquire _lock inside.
        """
        tolerance = self._OVERDUB_TOLERANCE
        lane = self._lanes[param]

        # Build new lane without points in the tolerance window (O(n) scan)
        filtered = [pt for pt in lane if abs(pt[0] - norm_pos) > tolerance]

        # Insert new point in sorted position (O(log n) search + O(n) insert)
        # bisect.insort compares tuples lexicographically: first by pos, then val.
        bisect.insort(filtered, (norm_pos, value))
        self._lanes[param] = filtered


# ── Module-level interpolation helper ────────────────────────────────────────

def _interpolate(lane: list[tuple[float, float]], norm_pos: float) -> float:
    """Binary-search linear interpolation on a SORTED (pos, val) list.

    Performance: O(log n) bisect + O(1) arithmetic.
    Clamps at the endpoints — no extrapolation.

    Precondition: lane is sorted by pos (ascending).  Always true after
    stop_record() sorts it and _overdub_point() uses bisect.insort().
    """
    if not lane:
        return 0.0
    if norm_pos <= lane[0][0]:
        return lane[0][1]
    if norm_pos >= lane[-1][0]:
        return lane[-1][1]

    # Build a temporary positions list for bisect — O(n) but unavoidable without
    # a separate index structure.  Flagged for Phase 10 optimisation if profiling
    # shows it hot (Section 12: target < 0.1 ms per channel per tick).
    positions = [pt[0] for pt in lane]
    idx = bisect.bisect_right(positions, norm_pos)  # first pos > norm_pos
    left_pos, left_val = lane[idx - 1]
    right_pos, right_val = lane[idx]

    span = right_pos - left_pos
    if span == 0.0:
        return left_val
    t = (norm_pos - left_pos) / span
    return left_val + t * (right_val - left_val)
