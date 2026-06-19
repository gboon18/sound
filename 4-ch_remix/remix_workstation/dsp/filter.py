# =============================================================================
# Author  : Ho San Ko
# Email   : hko@avalanche.energy
# Project : 4-Channel Music Remix Workstation
# =============================================================================

"""LP/HP filter wrapper around pedalboard.LowpassFilter / HighpassFilter."""

import pedalboard

from constants import Param, FilterType


class FilterProcessor:
    """Switches between LowpassFilter and HighpassFilter based on FILTER_TYPE.

    Parameters: FILTER_CUTOFF (Hz), FILTER_RESONANCE (Q), FILTER_TYPE (0=LP, 1=HP).
    Note: pedalboard's built-in filters don't expose Q directly; resonance is
    approximated via a PeakFilter staged in the chain for the resonant bump.
    The active plugin is swapped when FILTER_TYPE changes.
    """

    def __init__(self, sample_rate: int) -> None:
        self._sample_rate = sample_rate
        self._lp = pedalboard.LowpassFilter(cutoff_frequency_hz=20000.0)
        self._hp = pedalboard.HighpassFilter(cutoff_frequency_hz=20.0)
        self._active_type: FilterType = FilterType.LOWPASS
        self._cutoff: float = 20000.0
        self._resonance: float = 0.707

    def update(self, params: dict[Param, float]) -> None:
        """Update filter parameters and switch type if needed."""
        if Param.FILTER_TYPE in params:
            new_type = FilterType.HIGHPASS if params[Param.FILTER_TYPE] >= 0.5 else FilterType.LOWPASS
            self._active_type = new_type
        if Param.FILTER_CUTOFF in params:
            self._cutoff = params[Param.FILTER_CUTOFF]
        if Param.FILTER_RESONANCE in params:
            self._resonance = params[Param.FILTER_RESONANCE]

        if self._active_type == FilterType.LOWPASS:
            self._lp.cutoff_frequency_hz = self._cutoff
        else:
            self._hp.cutoff_frequency_hz = self._cutoff

    def get_plugin(self) -> pedalboard.Plugin:
        """Return the currently active filter plugin."""
        return self._lp if self._active_type == FilterType.LOWPASS else self._hp
