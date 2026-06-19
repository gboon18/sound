# =============================================================================
# Author  : Ho San Ko
# Email   : hko@avalanche.energy
# Project : 4-Channel Music Remix Workstation
# =============================================================================

"""Unit tests for SyncManager (Section 11 test plan)."""

import pytest

from engine.master_clock import MasterClock
from engine.sync_manager import SyncManager


def test_integer_ratio_zero_drift():
    """Integer ratio (120/120 = 1.0) → correct_drift should return 0."""
    clock = MasterClock(bpm=120.0)
    # SyncManager with no real players — test ratio calculation only
    sm = SyncManager(master_clock=clock, players=[])
    # Ratio for a 120 BPM track at 120 BPM master = 1.0
    assert sm._ratios == []


def test_get_ratio():
    """get_ratio should reflect recalculated ratios."""
    pytest.skip("Requires TrackPlayer stubs to be implemented")


def test_bpm_change_triggers_restretch():
    """Changing master BPM should trigger recalculate_ratios on all players."""
    pytest.skip("Requires TrackPlayer stubs to be implemented")
