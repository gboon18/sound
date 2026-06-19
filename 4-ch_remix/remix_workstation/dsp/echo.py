# =============================================================================
# Author  : Ho San Ko
# Email   : hko@avalanche.energy
# Project : 4-Channel Music Remix Workstation
# =============================================================================

"""Echo/delay wrapper around pedalboard.Delay."""

import pedalboard

from constants import Param


class EchoProcessor:
    """Thin wrapper around pedalboard.Delay.

    Parameters: ECHO_TIME (ms → seconds), ECHO_FEEDBACK (0–0.95), ECHO_MIX (0–1).
    """

    def __init__(self) -> None:
        self._delay = pedalboard.Delay(
            delay_seconds=0.5,
            feedback=0.3,
            mix=0.0,
        )

    def update(self, params: dict[Param, float]) -> None:
        if Param.ECHO_TIME in params:
            self._delay.delay_seconds = params[Param.ECHO_TIME] / 1000.0  # ms → s
        if Param.ECHO_FEEDBACK in params:
            self._delay.feedback = params[Param.ECHO_FEEDBACK]
        if Param.ECHO_MIX in params:
            self._delay.mix = params[Param.ECHO_MIX]

    def get_plugin(self) -> pedalboard.Delay:
        return self._delay
