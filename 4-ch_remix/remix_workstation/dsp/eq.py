# =============================================================================
# Author  : Ho San Ko
# Email   : hko@avalanche.energy
# Project : 4-Channel Music Remix Workstation
# =============================================================================

"""3-band EQ wrapper around pedalboard shelf/peak filters."""

import pedalboard

from constants import Param


class EQProcessor:
    """Wraps pedalboard.LowShelfFilter + PeakFilter + HighShelfFilter.

    Parameters: EQ_LOW (dB), EQ_MID (dB), EQ_HIGH (dB).
    Each gain is applied at a fixed centre frequency:
        low  → 100 Hz shelf
        mid  → 1 kHz peak  (Q = 1.0)
        high → 10 kHz shelf
    """

    _LOW_FREQ: float = 100.0
    _MID_FREQ: float = 1000.0
    _HIGH_FREQ: float = 10000.0
    _MID_Q: float = 1.0

    def __init__(self, sample_rate: int) -> None:
        self._sample_rate = sample_rate
        self._low_shelf = pedalboard.LowShelfFilter(
            cutoff_frequency_hz=self._LOW_FREQ, gain_db=0.0
        )
        self._mid_peak = pedalboard.PeakFilter(
            cutoff_frequency_hz=self._MID_FREQ, gain_db=0.0, q=self._MID_Q
        )
        self._high_shelf = pedalboard.HighShelfFilter(
            cutoff_frequency_hz=self._HIGH_FREQ, gain_db=0.0
        )

    def update(self, params: dict[Param, float]) -> None:
        """Apply new gain values to all three EQ bands."""
        if Param.EQ_LOW in params:
            self._low_shelf.gain_db = params[Param.EQ_LOW]
        if Param.EQ_MID in params:
            self._mid_peak.gain_db = params[Param.EQ_MID]
        if Param.EQ_HIGH in params:
            self._high_shelf.gain_db = params[Param.EQ_HIGH]

    def get_plugins(self) -> list[pedalboard.Plugin]:
        """Return ordered plugin list for inclusion in the pedalboard chain."""
        return [self._low_shelf, self._mid_peak, self._high_shelf]
