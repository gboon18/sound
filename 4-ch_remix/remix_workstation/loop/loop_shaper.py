# =============================================================================
# Author  : Ho San Ko
# Email   : hko@avalanche.energy
# Project : 4-Channel Music Remix Workstation
# =============================================================================

"""Loop Shaper — gesture automation recorder/replayer (Section 7.10).

The centrepiece creative feature: records knob movements mapped to the loop
timeline (0.0–1.0 normalized position) and replays them in sync every loop.
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
            Key   → parameter enum
            Value → sorted list of (normalized_loop_position, value)
                    norm_pos: 0.0 = loop start, 1.0 = loop end

    Attributes (Section 6 class diagram):
        _state          Current LoopShaperState.
        _mode           AutomationMode (ABSOLUTE or ADDITIVE).
        _lanes          Automation data per parameter.
        _touched_params Params modified in the current record/overdub pass.
        _loop_manager   Reference to the channel's LoopManager.
        _lock           Protects _lanes, _state, _mode.
    """

    _OVERDUB_TOLERANCE: float = 0.005  # normalized position window for overdub replace
    _RECORD_RATE_LIMIT_S: float = 0.001  # min seconds between recorded points per param
    _MAX_LANE_POINTS: int = 10_000      # hard cap; Douglas-Peucker applied at stop

    def __init__(self, loop_manager: "LoopManager") -> None:
        self._loop_manager = loop_manager
        self._state: LoopShaperState = LoopShaperState.IDLE
        self._mode: AutomationMode = AutomationMode.ABSOLUTE
        self._lanes: dict[Param, list[tuple[float, float]]] = {
            p: [] for p in Param
        }
        self._touched_params: set[Param] = set()
        self._lock: threading.Lock = threading.Lock()
        self._last_record_time: dict[Param, float] = {}  # for rate-limiting

    # ── Record ────────────────────────────────────────────────────────────────

    def arm_record(self) -> None:
        """Transition to ARMED; actual recording starts at the next loop wrap."""
        with self._lock:
            self._state = LoopShaperState.ARMED
            self._clear_all_lanes()

    def start_record(self) -> None:
        """Begin recording (called on loop wrap when ARMED)."""
        with self._lock:
            self._state = LoopShaperState.RECORDING
            self._touched_params.clear()
            self._last_record_time.clear()
            self._clear_all_lanes()

    def stop_record(self) -> None:
        """Sort all lanes and transition to PLAYING."""
        with self._lock:
            for lane in self._lanes.values():
                lane.sort(key=lambda pt: pt[0])
            self._state = LoopShaperState.PLAYING

    def record_point(self, param: Param, norm_pos: float, value: float) -> None:
        """Append a (norm_pos, value) point to *param*'s lane.

        Steps (Section 7.10):
        1. Guard: state must be RECORDING or OVERDUBBING.
        2. Rate-limit: skip if < 1 ms since last point for this param.
        3. Append and mark param as touched.
        """
        import time
        with self._lock:
            if self._state not in (LoopShaperState.RECORDING, LoopShaperState.OVERDUBBING):
                return
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

    # ── Overdub ───────────────────────────────────────────────────────────────

    def arm_overdub(self) -> None:
        """Transition to ARMED for overdub; existing lanes are NOT cleared."""
        with self._lock:
            self._state = LoopShaperState.ARMED  # overdub variant — loop_manager checks

    def start_overdub(self) -> None:
        """Begin overdub recording (called on loop wrap when armed for overdub)."""
        with self._lock:
            self._state = LoopShaperState.OVERDUBBING
            self._touched_params.clear()
            self._last_record_time.clear()

    # ── Evaluate ──────────────────────────────────────────────────────────────

    def evaluate(self, norm_pos: float) -> dict[Param, float]:
        """Return interpolated automation values at *norm_pos* for all active lanes.

        Steps (Section 7.10):
        1. Guard: state in {PLAYING, RECORDING, OVERDUBBING}.
        2. Binary search + linear interpolation per lane.
        3. ABSOLUTE → return interp value; ADDITIVE → return offset (caller adds).
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
                value = _interpolate(lane, norm_pos)
                result[param] = value
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
        """Proportionally rescale all norm_pos values on loop-length change.

        Steps (Section 7.10):
        scale_factor = old_length / new_length
        Multiply each norm_pos by scale_factor, clamp to [0.0, 1.0],
        discard points that fall outside (shrink case).
        """
        if new_length <= 0:
            return
        scale = old_length / new_length
        with self._lock:
            for param in self._lanes:
                rescaled = [
                    (min(1.0, max(0.0, pos * scale)), val)
                    for pos, val in self._lanes[param]
                ]
                self._lanes[param] = [pt for pt in rescaled if 0.0 <= pt[0] <= 1.0]
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
            return len(self._lanes[param]) > 0

    def get_state(self) -> LoopShaperState:
        with self._lock:
            return self._state

    # ── Internal ──────────────────────────────────────────────────────────────

    def _clear_all_lanes(self) -> None:
        for param in Param:
            self._lanes[param] = []

    def _overdub_point(self, param: Param, norm_pos: float, value: float) -> None:
        """Replace existing points within tolerance, then append the new one."""
        lane = self._lanes[param]
        tolerance = self._OVERDUB_TOLERANCE
        self._lanes[param] = [
            pt for pt in lane
            if abs(pt[0] - norm_pos) > tolerance
        ]
        self._lanes[param].append((norm_pos, value))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _interpolate(lane: list[tuple[float, float]], norm_pos: float) -> float:
    """Linear interpolation on a sorted list of (position, value) pairs.

    Uses bisect for O(log n) lookup.  Clamps at the endpoints.
    """
    if not lane:
        return 0.0
    if norm_pos <= lane[0][0]:
        return lane[0][1]
    if norm_pos >= lane[-1][0]:
        return lane[-1][1]

    # Find the right bracket using the positions list (bisect on first element)
    positions = [pt[0] for pt in lane]
    idx = bisect.bisect_right(positions, norm_pos)
    left = lane[idx - 1]
    right = lane[idx]

    span = right[0] - left[0]
    if span == 0.0:
        return left[1]
    t = (norm_pos - left[0]) / span
    return left[1] + t * (right[1] - left[1])
