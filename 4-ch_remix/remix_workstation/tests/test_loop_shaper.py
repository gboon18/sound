# =============================================================================
# Author  : Ho San Ko
# Email   : hko@avalanche.energy
# Project : 4-Channel Music Remix Workstation
# =============================================================================

"""Unit tests for LoopShaper (Section 11 test plan).

rescale() semantics (verified against spec Section 7.10):
    scale_factor = old_length / new_length
    new_pos      = old_pos * scale_factor

    SHRINK (old > new, scale > 1): positions expand toward 1.0; points
        whose new_pos > 1.0 are DISCARDED (they fall outside the shorter loop).

    EXPAND (old < new, scale < 1): positions compress toward 0.0; all points
        remain in [0.0, 1.0]; the tail [scale..1.0] is unautomated.

    Critical: do NOT clamp before filtering — clamping would silently stack
    discarded points at the boundary instead of removing them.
"""

import pytest
from unittest.mock import MagicMock

from constants import Param, AutomationMode, LoopShaperState, PARAM_RANGES
from loop.loop_shaper import LoopShaper, _interpolate


@pytest.fixture
def shaper():
    """Fresh shaper already in RECORDING state."""
    loop_manager = MagicMock()
    s = LoopShaper(loop_manager)
    s.start_record()
    return s


# ── Interpolation helper ───────────────────────────────────────────────────────

def test_interpolate_midpoint():
    """Linear interpolation between two known points."""
    lane = [(0.0, 0.0), (1.0, 1.0)]
    assert abs(_interpolate(lane, 0.5) - 0.5) < 1e-9


def test_interpolate_clamp_below():
    lane = [(0.3, 10.0), (0.7, 20.0)]
    assert _interpolate(lane, 0.0) == 10.0   # clamps to first value


def test_interpolate_clamp_above():
    lane = [(0.3, 10.0), (0.7, 20.0)]
    assert _interpolate(lane, 1.0) == 20.0   # clamps to last value


# ── Record and evaluate ───────────────────────────────────────────────────────

def test_record_and_interpolate(shaper):
    """Record three colinear points; mid-point should interpolate correctly."""
    shaper.record_point(Param.EQ_HIGH, 0.0, 0.0, force=True)
    shaper.record_point(Param.EQ_HIGH, 0.5, 0.5, force=True)
    shaper.record_point(Param.EQ_HIGH, 1.0, 1.0, force=True)
    shaper.stop_record()

    result = shaper.evaluate(0.25)
    assert Param.EQ_HIGH in result
    assert abs(result[Param.EQ_HIGH] - 0.25) < 1e-6, (
        f"Expected 0.25, got {result[Param.EQ_HIGH]}"
    )


def test_evaluate_idle_returns_empty(shaper):
    """evaluate() must return {} when state is IDLE."""
    shaper.clear_all()  # → IDLE
    assert shaper.evaluate(0.5) == {}


def test_evaluate_playing_after_stop(shaper):
    """evaluate() returns values in PLAYING state."""
    shaper.record_point(Param.EQ_LOW, 0.0, 0.3)
    shaper.record_point(Param.EQ_LOW, 1.0, 0.7)
    shaper.stop_record()
    result = shaper.evaluate(0.0)
    assert abs(result.get(Param.EQ_LOW, -1) - 0.3) < 1e-6


# ── Overdub merge ─────────────────────────────────────────────────────────────

def test_overdub_preserves_untouched_lanes(shaper):
    """Overdubbing EQ_HIGH must not modify the EQ_LOW lane."""
    shaper.record_point(Param.EQ_HIGH, 0.0, 1.0)
    shaper.record_point(Param.EQ_LOW,  0.0, 0.5)
    shaper.stop_record()

    shaper.start_overdub()
    shaper.record_point(Param.EQ_HIGH, 0.0, 0.9)
    shaper.stop_record()

    result = shaper.evaluate(0.0)
    assert abs(result.get(Param.EQ_LOW, -1) - 0.5) < 1e-6, (
        "EQ_LOW lane was modified by overdub"
    )


def test_overdub_replaces_within_tolerance(shaper):
    """Overdub at norm_pos=0.5 should replace existing point at 0.5."""
    shaper.record_point(Param.EQ_MID, 0.5, 0.2)
    shaper.stop_record()

    shaper.start_overdub()
    shaper.record_point(Param.EQ_MID, 0.5, 0.8)
    shaper.stop_record()

    lane = shaper.get_lane(Param.EQ_MID)
    vals_at_0_5 = [v for p, v in lane if abs(p - 0.5) <= LoopShaper._OVERDUB_TOLERANCE]
    assert len(vals_at_0_5) == 1 and abs(vals_at_0_5[0] - 0.8) < 1e-6, (
        "Overdub did not replace the point at 0.5"
    )


def test_overdub_preserves_outside_tolerance(shaper):
    """Points well outside the overdub position should survive."""
    shaper.record_point(Param.EQ_MID, 0.1, 0.3, force=True)
    shaper.record_point(Param.EQ_MID, 0.9, 0.7, force=True)
    shaper.stop_record()

    shaper.start_overdub()
    shaper.record_point(Param.EQ_MID, 0.5, 0.5, force=True)
    shaper.stop_record()

    lane = shaper.get_lane(Param.EQ_MID)
    positions = [p for p, _ in lane]
    assert 0.1 in [round(p, 6) for p in positions], "Point at 0.1 was unexpectedly removed"
    assert 0.9 in [round(p, 6) for p in positions], "Point at 0.9 was unexpectedly removed"


def test_overdub_lane_remains_sorted(shaper):
    """Lane must be sorted by position after overdub (required for bisect search)."""
    for pos in [0.8, 0.2, 0.5]:
        shaper.record_point(Param.REVERB_MIX, pos, pos)
    shaper.stop_record()

    shaper.start_overdub()
    shaper.record_point(Param.REVERB_MIX, 0.4, 0.9)
    shaper.stop_record()

    lane = shaper.get_lane(Param.REVERB_MIX)
    positions = [p for p, _ in lane]
    assert positions == sorted(positions), f"Lane not sorted: {positions}"


# ── rescale ───────────────────────────────────────────────────────────────────

def test_rescale_shrink_keeps_point_in_range(shaper):
    """SHRINK 8→4 bars (scale=2): point at 0.25 → 0.5 (kept)."""
    shaper.record_point(Param.EQ_MID, 0.25, 0.6)
    shaper.stop_record()

    shaper.rescale(old_length=8.0, new_length=4.0)  # scale = 2.0

    lane = shaper.get_lane(Param.EQ_MID)
    assert len(lane) == 1, f"Expected 1 point, got {len(lane)}"
    assert abs(lane[0][0] - 0.5) < 1e-9, f"Expected 0.5, got {lane[0][0]}"


def test_rescale_shrink_discards_out_of_range(shaper):
    """SHRINK 8→4 bars: point at 0.6 → 1.2 (>1.0) must be discarded."""
    shaper.record_point(Param.EQ_MID, 0.6, 0.5)
    shaper.stop_record()

    shaper.rescale(old_length=8.0, new_length=4.0)  # scale = 2.0; 0.6*2 = 1.2

    lane = shaper.get_lane(Param.EQ_MID)
    assert len(lane) == 0, (
        f"Expected 0 points after discard, got {len(lane)}: {lane}"
    )


def test_rescale_shrink_boundary_kept(shaper):
    """SHRINK 8→4: point at exactly 0.5 → 1.0 (boundary — must be kept)."""
    shaper.record_point(Param.EQ_MID, 0.5, 0.7)
    shaper.stop_record()

    shaper.rescale(old_length=8.0, new_length=4.0)  # 0.5 * 2 = 1.0

    lane = shaper.get_lane(Param.EQ_MID)
    assert len(lane) == 1, "Boundary point at new_pos=1.0 should be kept"
    assert abs(lane[0][0] - 1.0) < 1e-9


def test_rescale_expand_compresses_toward_zero(shaper):
    """EXPAND 4→8 bars (scale=0.5): point at 0.5 → 0.25 (compressed toward 0)."""
    shaper.record_point(Param.EQ_MID, 0.5, 0.8)
    shaper.stop_record()

    shaper.rescale(old_length=4.0, new_length=8.0)  # scale = 0.5; 0.5*0.5 = 0.25

    lane = shaper.get_lane(Param.EQ_MID)
    assert len(lane) == 1, f"Expected 1 point, got {len(lane)}"
    assert abs(lane[0][0] - 0.25) < 1e-9, f"Expected 0.25, got {lane[0][0]}"


def test_rescale_expand_keeps_all_points(shaper):
    """EXPAND: all points compress toward 0 — none are discarded."""
    for pos in [0.2, 0.5, 0.8, 1.0]:
        shaper.record_point(Param.EQ_MID, pos, 0.5, force=True)
    shaper.stop_record()

    shaper.rescale(old_length=4.0, new_length=8.0)  # scale = 0.5

    lane = shaper.get_lane(Param.EQ_MID)
    assert len(lane) == 4, f"Expected 4 points after expand, got {len(lane)}"
    positions = [p for p, _ in lane]
    assert all(0.0 <= p <= 1.0 for p in positions)
    assert positions == sorted(positions)


def test_rescale_preserves_zero_position(shaper):
    """Point at norm_pos=0.0 must map to 0.0 under any scale."""
    shaper.record_point(Param.VOLUME, 0.0, 0.5)
    shaper.stop_record()

    shaper.rescale(old_length=4.0, new_length=8.0)
    lane = shaper.get_lane(Param.VOLUME)
    assert len(lane) == 1
    assert lane[0][0] == 0.0


# ── Lane management ───────────────────────────────────────────────────────────

def test_clear_lane(shaper):
    """clear_lane clears one param without affecting others."""
    shaper.record_point(Param.EQ_HIGH, 0.0, 1.0)
    shaper.record_point(Param.EQ_LOW,  0.0, 0.5)
    shaper.stop_record()

    shaper.clear_lane(Param.EQ_HIGH)

    assert not shaper.has_automation(Param.EQ_HIGH)
    assert shaper.has_automation(Param.EQ_LOW)


def test_clear_all_sets_idle(shaper):
    """clear_all() empties all lanes and returns to IDLE."""
    shaper.record_point(Param.EQ_MID, 0.5, 0.5)
    shaper.stop_record()
    shaper.clear_all()

    assert shaper.get_state() == LoopShaperState.IDLE
    assert shaper.evaluate(0.5) == {}


# ── Additive mode clamping (edge case 15) ─────────────────────────────────────

def test_additive_mode_clamp_upper(shaper):
    """Additive automation + manual value must not exceed param max."""
    r = PARAM_RANGES[Param.EQ_HIGH]
    shaper.record_point(Param.EQ_HIGH, 0.0, r.max_val)
    shaper.record_point(Param.EQ_HIGH, 1.0, r.max_val)
    shaper.stop_record()
    shaper.set_mode(AutomationMode.ADDITIVE)

    interp_val = shaper.evaluate(0.5).get(Param.EQ_HIGH, 0.0)
    manual = r.max_val
    combined = max(r.min_val, min(r.max_val, manual + interp_val))
    assert combined == r.max_val, "Additive overflow not clamped to max"


def test_additive_mode_clamp_lower(shaper):
    """Additive automation + manual value must not go below param min."""
    r = PARAM_RANGES[Param.EQ_HIGH]
    shaper.record_point(Param.EQ_HIGH, 0.0, r.min_val)
    shaper.record_point(Param.EQ_HIGH, 1.0, r.min_val)
    shaper.stop_record()
    shaper.set_mode(AutomationMode.ADDITIVE)

    interp_val = shaper.evaluate(0.5).get(Param.EQ_HIGH, 0.0)
    manual = r.min_val
    combined = max(r.min_val, min(r.max_val, manual + interp_val))
    assert combined == r.min_val, "Additive underflow not clamped to min"


# ── ARMED state machine ───────────────────────────────────────────────────────

def test_arm_record_starts_on_wrap():
    """arm_record() → on_loop_wrap() → RECORDING (edge case 13)."""
    lm = MagicMock()
    s = LoopShaper(lm)
    s.arm_record()
    assert s.get_state() == LoopShaperState.ARMED
    s.on_loop_wrap()
    assert s.get_state() == LoopShaperState.RECORDING


def test_arm_overdub_starts_on_wrap_preserves_lanes():
    """arm_overdub() → on_loop_wrap() → OVERDUBBING; existing lanes survive."""
    lm = MagicMock()
    s = LoopShaper(lm)
    s.start_record()
    s.record_point(Param.EQ_LOW, 0.3, 0.5)
    s.stop_record()

    s.arm_overdub()
    s.on_loop_wrap()
    assert s.get_state() == LoopShaperState.OVERDUBBING
    # Lane still has the original point
    assert s.has_automation(Param.EQ_LOW)
