# =============================================================================
# Author  : Ho San Ko
# Email   : hko@avalanche.energy
# Project : 4-Channel Music Remix Workstation
# =============================================================================

"""Unit tests for MidiMap (Section 11 test plan)."""

import json
import tempfile
from pathlib import Path

import pytest

from constants import Param
from midi.midi_map import MidiMap, MidiAddress, ControlTarget


@pytest.fixture
def midi_map():
    return MidiMap()


@pytest.fixture
def addr():
    return MidiAddress(channel=0, msg_type="cc", number=1)


@pytest.fixture
def target():
    return ControlTarget(channel_idx=0, param=Param.EQ_HIGH, action="set")


def test_add_and_lookup(midi_map, addr, target):
    """add() then lookup() should return the same target."""
    midi_map.add(addr, target)
    result = midi_map.lookup(addr)
    assert result == target


def test_lookup_missing_returns_none(midi_map, addr):
    assert midi_map.lookup(addr) is None


def test_remove(midi_map, addr, target):
    midi_map.add(addr, target)
    midi_map.remove(addr)
    assert midi_map.lookup(addr) is None


def test_conflict_detection(midi_map):
    """Same target assigned to two different addresses → conflict reported."""
    addr1 = MidiAddress(channel=0, msg_type="cc", number=1)
    addr2 = MidiAddress(channel=0, msg_type="cc", number=2)
    tgt = ControlTarget(channel_idx=0, param=Param.EQ_HIGH, action="set")
    midi_map.add(addr1, tgt)
    midi_map.add(addr2, tgt)
    conflicts = midi_map.check_conflicts()
    assert len(conflicts) > 0, "Expected a conflict but check_conflicts() returned []"


def test_save_load_round_trip(midi_map, addr, target):
    """save() then load() should reproduce the same mappings."""
    midi_map.add(addr, target)
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "test_map.json")
        midi_map.save(path)

        loaded = MidiMap()
        loaded.load(path)
        result = loaded.lookup(addr)
    assert result is not None
    assert result.channel_idx == target.channel_idx
    assert result.action == target.action
