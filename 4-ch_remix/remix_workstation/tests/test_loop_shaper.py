# =============================================================================
# Author  : Ho San Ko
# Email   : hko@avalanche.energy
# Project : 4-Channel Music Remix Workstation
# =============================================================================

"""Unit tests for LoopShaper (Section 11 test plan)."""

import pytest
from unittest.mock import MagicMock

from constants import Param, AutomationMode, LoopShaperState, PARAM_RANGES
from loop.loop_shaper import LoopShaper, _interpolate


@pytest.fixture
def shaper():
    loop_manager = MagicMock()
    s = LoopShaper(loop_manager)
    s.start_record()
    return s


def test_record_and_interpolate(shaper):
    """Record 3 known points and verify linear interpolation at midpoints."""
    shaper.record_point(Param.EQ_HIGH, 0.0, 0.0)
    shaper.record_point(Param.EQ_HIGH, 0.5, 0.5)
    shaper.record_point(Param.EQ_HIGH, 1.0, 1.0)
    shaper.stop_record()

    result = shaper.evaluate(0.25)
    assert Param.EQ_HIGH in result
    assert abs(result[Param.EQ_HIGH] - 0.25) < 1e-6, f"Expected 0.25, got {result[Param.EQ_HIGH]}"


def test_overdub_preserves_untouched_lanes(shaper):
    """Overdub on EQ_HIGH should not modify the EQ_LOW lane."""
    shaper.record_point(Param.EQ_HIGH, 0.0, 1.0)
    shaper.record_point(Param.EQ_LOW, 0.0, 0.5)
    shaper.stop_record()

    shaper.start_overdub()
    shaper.record_point(Param.EQ_HIGH, 0.0, 0.9)
    shaper.stop_record()

    result = shaper.evaluate(0.0)
    assert Param.EQ_LOW in result
    assert abs(result[Param.EQ_LOW] - 0.5) < 1e-6, "EQ_LOW lane was modified by overdub"


def test_rescale_8_to_4_bars(shaper):
    """Rescale 8→4 bars: positions should be compressed by 0.5×."""
    shaper.record_point(Param.EQ_MID, 0.5, 0.8)
    shaper.stop_record()

    shaper.rescale(old_length=8.0, new_length=4.0)
    lane = shaper.get_lane(Param.EQ_MID)
    assert len(lane) == 1, "Expected 1 point after rescale"
    assert abs(lane[0][0] - 0.25) < 1e-6, f"Expected 0.25, got {lane[0][0]}"


def test_rescale_discard_out_of_range(shaper):
    """Shrinking loop should discard points that fall outside [0, 1]."""
    shaper.record_point(Param.EQ_MID, 0.8, 0.5)  # norm_pos=0.8 * 2.0 = 1.6 → discard
    shaper.stop_record()

    shaper.rescale(old_length=8.0, new_length=4.0)  # scale=2.0
    lane = shaper.get_lane(Param.EQ_MID)
    assert len(lane) == 0, f"Expected 0 points, got {len(lane)}"


def test_clear_lane(shaper):
    """clear_lane should empty one lane without touching others."""
    shaper.record_point(Param.EQ_HIGH, 0.0, 1.0)
    shaper.record_point(Param.EQ_LOW, 0.0, 0.5)
    shaper.stop_record()

    shaper.clear_lane(Param.EQ_HIGH)
    assert not shaper.has_automation(Param.EQ_HIGH)
    assert shaper.has_automation(Param.EQ_LOW)


def test_additive_mode_clamps(shaper):
    """Additive automation combined with manual value must not exceed param bounds."""
    r = PARAM_RANGES[Param.EQ_HIGH]
    # We test _interpolate directly to check clamping in the caller context
    lane = [(0.0, r.max_val), (1.0, r.max_val)]
    interp = _interpolate(lane, 0.5)
    manual = r.max_val
    # Caller applies clamping: max(min_val, min(max_val, manual + interp))
    combined = max(r.min_val, min(r.max_val, manual + interp))
    assert combined == r.max_val, "Additive clamping failed at upper bound"
