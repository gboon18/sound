# =============================================================================
# Author  : Ho San Ko
# Email   : hko@avalanche.energy
# Project : 4-Channel Music Remix Workstation
# =============================================================================

"""Per-channel DSP chain orchestrator using pedalboard (Section 7.7).

Performance contract (Section 12):
  process() must complete in < 5.8 ms (half of a 512-frame buffer period at 44100 Hz).

Design: the Pedalboard object is created ONCE in __init__ and never rebuilt.
Plugin parameters are updated IN-PLACE through each sub-processor's update()
method.  Since all plugin objects are shared between the Pedalboard list and
the sub-processor, modifying e.g. self._reverb._reverb.room_size immediately
affects the next board() call — no rebuild needed.

Thread safety: _lock is held throughout both process() and set_param().
The audio callback thread and MIDI/UI thread serialise on this one lock.
Lock-hold time for set_param() is O(1); for process() it is the full
pedalboard processing time but this is expected (the callback budget is theirs).
"""

import math
import threading

import numpy as np
import pedalboard

from constants import Param, ParamRange, PARAM_RANGES, SAMPLE_RATE
from dsp.eq import EQProcessor
from dsp.filter import FilterProcessor
from dsp.reverb import ReverbProcessor
from dsp.echo import EchoProcessor
from dsp.pitch import PitchProcessor


class ChannelDSP:
    """Orchestrates the full FX chain for one channel.

    Chain order (Section 7.7):  EQ → Filter → Reverb → Echo → Pitch → Vol/Pan

    Attributes (Section 6 class diagram):
        _board    Pedalboard instance — created once, never rebuilt.
        _eq       EQProcessor (3 pedalboard plugins).
        _filter   FilterProcessor (2 plugins — LP and HP always present).
        _reverb   ReverbProcessor.
        _echo     EchoProcessor.
        _pitch    PitchProcessor.
        _volume   Cached from _params[Param.VOLUME] for fast process().
        _pan      Cached from _params[Param.PAN]    for fast process().
        _params   Dict of all 16 current Param values.
        _lock     Serialises process() and set_param() across threads.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self._sample_rate = sample_rate
        self._lock: threading.Lock = threading.Lock()

        # Sub-processors — own the plugin objects
        self._eq = EQProcessor(sample_rate)
        self._filter = FilterProcessor(sample_rate)
        self._reverb = ReverbProcessor()
        self._echo = EchoProcessor()
        self._pitch = PitchProcessor()

        # Default param state
        self._params: dict[Param, float] = {
            p: r.default for p, r in PARAM_RANGES.items()
        }

        # Build the board ONCE: collect plugin refs in chain order
        all_plugins: list[pedalboard.Plugin] = []
        all_plugins.extend(self._eq.get_plugins())        # 3 plugins: low/mid/high shelf
        all_plugins.extend(self._filter.get_plugins())    # 2 plugins: LP + HP
        all_plugins.extend(self._reverb.get_plugins())    # 1 plugin
        all_plugins.extend(self._echo.get_plugins())      # 1 plugin
        all_plugins.extend(self._pitch.get_plugins())     # 1 plugin
        self._board: pedalboard.Pedalboard = pedalboard.Pedalboard(all_plugins)

        # Sync sub-processors to the default param state (sets plugin attrs)
        self._sync_all()

    # ── Public API ────────────────────────────────────────────────────────────

    def process(self, buffer: np.ndarray) -> np.ndarray:
        """Run *buffer* through the FX chain, then apply volume and pan.

        Args:
            buffer: (frames, 2) float32 stereo array.
        Returns:
            Processed (frames, 2) float32 array.

        pedalboard expects (channels, samples) — we transpose in and out.
        """
        with self._lock:
            audio_in = buffer.T.astype(np.float32, copy=False)  # (2, frames)
            audio_out: np.ndarray = self._board(audio_in, self._sample_rate)
            result = audio_out.T.copy()  # (frames, 2), writeable

            # Volume — simple scalar multiply
            result *= self._params[Param.VOLUME]

            # Constant-power pan law (Section 7.7)
            # angle = 45° when pan=0 (centre) → equal power L+R
            pan = self._params[Param.PAN]
            angle = pan * (math.pi / 4.0) + (math.pi / 4.0)
            result[:, 0] *= math.cos(angle)
            result[:, 1] *= math.sin(angle)

            return result.astype(np.float32, copy=False)

    def set_param(self, param: Param, value: float) -> None:
        """Clamp *value* to the param's range and update the plugin in-place.

        Dispatches only to the affected sub-processor — O(1) per call.
        Volume and Pan are stored in _params and applied during process().
        """
        r: ParamRange = PARAM_RANGES[param]
        value = max(r.min_val, min(r.max_val, float(value)))
        with self._lock:
            self._params[param] = value
            # Dispatch to the affected sub-processor only
            if param in (Param.EQ_LOW, Param.EQ_MID, Param.EQ_HIGH):
                self._eq.update({param: value})
            elif param in (Param.FILTER_CUTOFF, Param.FILTER_RESONANCE, Param.FILTER_TYPE):
                self._filter.update({param: value})
            elif param in (Param.REVERB_SIZE, Param.REVERB_DAMP, Param.REVERB_MIX):
                self._reverb.update({param: value})
            elif param in (Param.ECHO_TIME, Param.ECHO_FEEDBACK, Param.ECHO_MIX):
                self._echo.update({param: value})
            elif param in (Param.PITCH_SEMITONE, Param.PITCH_CENTS):
                self._pitch.update({param: value})
            # VOLUME and PAN: stored in _params, applied directly in process()

    def get_param(self, param: Param) -> float:
        with self._lock:
            return self._params[param]

    def get_all_params(self) -> dict[Param, float]:
        with self._lock:
            return dict(self._params)

    def reset_to_defaults(self) -> None:
        with self._lock:
            self._params = {p: r.default for p, r in PARAM_RANGES.items()}
            self._sync_all()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _sync_all(self) -> None:
        """Push ALL current _params into every sub-processor (used on init/reset)."""
        self._eq.update(self._params)
        self._filter.update(self._params)
        self._reverb.update(self._params)
        self._echo.update(self._params)
        self._pitch.update(self._params)
