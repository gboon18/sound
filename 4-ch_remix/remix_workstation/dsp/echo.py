# =============================================================================
# Author  : Ho San Ko
# Email   : hko@avalanche.energy
# Project : 4-Channel Music Remix Workstation
# =============================================================================

"""Echo/delay wrapper around pedalboard.Delay."""

import pedalboard

from constants import Param


class EchoProcessor:
    """Thin wrapper. Parameters: ECHO_TIME (ms), ECHO_FEEDBACK (0–0.95), ECHO_MIX."""

    def __init__(self) -> None:
        self._delay = pedalboard.Delay(delay_seconds=0.5, feedback=0.3, mix=0.0)

    def update(self, params: dict[Param, float]) -> None:
        if Param.ECHO_TIME in params:
            self._delay.delay_seconds = float(params[Param.ECHO_TIME]) / 1000.0
        if Param.ECHO_FEEDBACK in params:
            self._delay.feedback = float(params[Param.ECHO_FEEDBACK])
        if Param.ECHO_MIX in params:
            self._delay.mix = float(params[Param.ECHO_MIX])

    def get_plugins(self) -> list[pedalboard.Plugin]:
        return [self._delay]
