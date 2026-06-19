# =============================================================================
# Author  : Ho San Ko
# Email   : hko@avalanche.energy
# Project : 4-Channel Music Remix Workstation
# =============================================================================

"""Unit tests for MixBus (Section 11 test plan)."""

import numpy as np
import pytest

from engine.mix_bus import MixBus

FRAMES = 512


@pytest.fixture
def bus():
    return MixBus(num_channels=4)


def _silence(frames=FRAMES):
    return np.zeros((frames, 2), dtype=np.float32)


def _full(frames=FRAMES, value=1.0):
    return np.full((frames, 2), value, dtype=np.float32)


def test_mix_silence(bus):
    """4 silent buffers should produce a silent output."""
    result = bus.mix([_silence()] * 4)
    assert result.shape == (FRAMES, 2)
    np.testing.assert_array_equal(result, 0.0)


def test_mix_sum_amplitude(bus):
    """4 identical unit-amplitude buffers summed → 4.0, clipped to 1.0."""
    result = bus.mix([_full(value=1.0)] * 4)
    np.testing.assert_array_less(result, 1.001)  # clipped
    np.testing.assert_array_less(-result, 1.001)


def test_mix_four_unit_without_clip():
    """When master_volume = 0.25, 4× unit buffers → 1.0 (no clip)."""
    bus = MixBus(num_channels=4)
    bus.set_master_vol(0.25)
    result = bus.mix([_full(value=1.0)] * 4)
    np.testing.assert_allclose(result, 1.0, atol=1e-6)


def test_master_volume_halves_output(bus):
    """Setting master volume to 0.5 should halve the output."""
    bus.set_master_vol(0.5)
    result = bus.mix([_full(value=0.5)] * 1)
    # 0.5 * 0.5 = 0.25
    np.testing.assert_allclose(result, 0.25, atol=1e-6)


def test_clipping(bus):
    """Sum exceeding 1.0 must be clipped to exactly 1.0."""
    bus.set_master_vol(1.0)
    result = bus.mix([_full(value=1.0)] * 4)
    assert float(result.max()) <= 1.0, "Output exceeds 1.0 — clipping failed"
    assert float(result.min()) >= -1.0, "Output below -1.0 — clipping failed"


def test_output_dtype(bus):
    """Output dtype must be float32."""
    result = bus.mix([_silence()])
    assert result.dtype == np.float32
