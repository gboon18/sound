# =============================================================================
# Author  : Ho San Ko
# Email   : hko@avalanche.energy
# Project : 4-Channel Music Remix Workstation
# =============================================================================

"""LP/HP filter wrapper — both plugins always live in the Pedalboard chain.

The key design decision: keeping LP and HP in the board simultaneously avoids
rebuilding the Pedalboard (an expensive Python-object swap) on every filter-type
toggle.  Instead:
  • LP mode  → LP cutoff = user value,  HP cutoff = 20 Hz  (acoustic pass-through)
  • HP mode  → HP cutoff = user value,  LP cutoff = 20 kHz (acoustic pass-through)

A LowpassFilter at 20 kHz passes virtually all audible content; a HighpassFilter
at 20 Hz removes only DC — both behave as identity transforms in practice.
"""

import pedalboard

from constants import FilterType, Param


_LP_PASSTHROUGH_HZ: float = 20_000.0
_HP_PASSTHROUGH_HZ: float = 20.0


class FilterProcessor:
    """Maintains both LP and HP plugins; swaps their cutoffs to select the type.

    Parameters: FILTER_CUTOFF (Hz), FILTER_RESONANCE (Q), FILTER_TYPE (0=LP, 1=HP).

    Note: pedalboard's LowpassFilter / HighpassFilter do not expose a Q / resonance
    parameter; a resonant bump requires a separate PeakFilter at the cutoff frequency.
    Resonance is stored in _params and passed forward for Phase 4 completeness, but
    the pedalboard API only gains Q support if moved to a Biquad formulation later.
    """

    def __init__(self, sample_rate: int) -> None:
        self._sample_rate = sample_rate
        self._lp = pedalboard.LowpassFilter(cutoff_frequency_hz=_LP_PASSTHROUGH_HZ)
        self._hp = pedalboard.HighpassFilter(cutoff_frequency_hz=_HP_PASSTHROUGH_HZ)
        self._active_type: FilterType = FilterType.LOWPASS
        self._cutoff: float = _LP_PASSTHROUGH_HZ
        self._resonance: float = 0.707

    def update(self, params: dict[Param, float]) -> None:
        """Apply new filter parameters in-place."""
        changed_type = False
        if Param.FILTER_TYPE in params:
            new_type = (
                FilterType.HIGHPASS if params[Param.FILTER_TYPE] >= 0.5
                else FilterType.LOWPASS
            )
            if new_type != self._active_type:
                self._active_type = new_type
                changed_type = True
        if Param.FILTER_CUTOFF in params:
            self._cutoff = params[Param.FILTER_CUTOFF]
        if Param.FILTER_RESONANCE in params:
            self._resonance = params[Param.FILTER_RESONANCE]

        self._apply_cutoffs()

    def _apply_cutoffs(self) -> None:
        if self._active_type == FilterType.LOWPASS:
            self._lp.cutoff_frequency_hz = self._cutoff
            self._hp.cutoff_frequency_hz = _HP_PASSTHROUGH_HZ
        else:
            self._hp.cutoff_frequency_hz = self._cutoff
            self._lp.cutoff_frequency_hz = _LP_PASSTHROUGH_HZ

    def get_plugins(self) -> list[pedalboard.Plugin]:
        """Return [LP, HP] — both are always included in the chain."""
        return [self._lp, self._hp]
