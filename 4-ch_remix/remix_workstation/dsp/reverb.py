# =============================================================================
# Author  : Ho San Ko
# Email   : hko@avalanche.energy
# Project : 4-Channel Music Remix Workstation
# =============================================================================

"""Reverb wrapper around pedalboard.Reverb."""

import pedalboard

from constants import Param


class ReverbProcessor:
    """Thin wrapper. Parameters: REVERB_SIZE, REVERB_DAMP, REVERB_MIX.

    wet_level + dry_level are kept summing to 1.0 so total gain stays unity.
    """

    def __init__(self) -> None:
        self._reverb = pedalboard.Reverb(
            room_size=0.3, damping=0.5, wet_level=0.0, dry_level=1.0,
        )

    def update(self, params: dict[Param, float]) -> None:
        if Param.REVERB_SIZE in params:
            self._reverb.room_size = float(params[Param.REVERB_SIZE])
        if Param.REVERB_DAMP in params:
            self._reverb.damping = float(params[Param.REVERB_DAMP])
        if Param.REVERB_MIX in params:
            mix = float(params[Param.REVERB_MIX])
            self._reverb.wet_level = mix
            self._reverb.dry_level = 1.0 - mix

    def get_plugins(self) -> list[pedalboard.Plugin]:
        return [self._reverb]
