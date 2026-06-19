# =============================================================================
# Author  : Ho San Ko
# Email   : hko@avalanche.energy
# Project : 4-Channel Music Remix Workstation
# =============================================================================

"""Pitch shift wrapper around pedalboard.PitchShift."""

import pedalboard

from constants import Param


class PitchProcessor:
    """Thin wrapper. Parameters: PITCH_SEMITONE, PITCH_CENTS.

    Total shift = semitones + cents / 100.0.
    """

    def __init__(self) -> None:
        self._plugin = pedalboard.PitchShift(semitones=0.0)
        self._semitones: float = 0.0
        self._cents: float = 0.0

    def update(self, params: dict[Param, float]) -> None:
        if Param.PITCH_SEMITONE in params:
            self._semitones = float(params[Param.PITCH_SEMITONE])
        if Param.PITCH_CENTS in params:
            self._cents = float(params[Param.PITCH_CENTS])
        self._plugin.semitones = self._semitones + self._cents / 100.0

    def get_plugins(self) -> list[pedalboard.Plugin]:
        return [self._plugin]
