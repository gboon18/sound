# =============================================================================
# Author  : Ho San Ko
# Email   : hko@avalanche.energy
# Project : 4-Channel Music Remix Workstation
# =============================================================================

"""Per-channel DSP chain orchestrator using pedalboard."""

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
        _board      The pedalboard.Pedalboard instance (rebuilt on param change).
        _eq         EQProcessor instance.
        _filter     FilterProcessor instance.
        _reverb     ReverbProcessor instance.
        _echo       EchoProcessor instance.
        _pitch      PitchProcessor instance.
        _volume     Current linear volume (0–1).
        _pan        Current pan position (-1=L, +1=R).
        _params     Dict of all 16 Param values.
        _lock       Protects _params and the pedalboard chain.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self._sample_rate = sample_rate
        self._lock: threading.Lock = threading.Lock()

        # Sub-processors
        self._eq = EQProcessor(sample_rate)
        self._filter = FilterProcessor(sample_rate)
        self._reverb = ReverbProcessor()
        self._echo = EchoProcessor()
        self._pitch = PitchProcessor()

        # Initialize params to defaults
        self._params: dict[Param, float] = {
            p: r.default for p, r in PARAM_RANGES.items()
        }

        self._board: pedalboard.Pedalboard = self._build_board()

    # ── Public API ────────────────────────────────────────────────────────────

    def process(self, buffer: np.ndarray) -> np.ndarray:
        """Run *buffer* through the full FX chain and apply volume + pan.

        Args:
            buffer: (frames, 2) float32 stereo array.
        Returns:
            Processed (frames, 2) float32 array.
        """
        with self._lock:
            # Pedalboard expects (channels, samples) — transpose in/out
            audio_in = buffer.T.astype(np.float32)
            audio_out = self._board(audio_in, self._sample_rate)
            result = audio_out.T  # back to (frames, 2)

            # Volume
            result = result * self._params[Param.VOLUME]

            # Constant-power pan law (Section 7.7)
            pan = self._params[Param.PAN]
            angle = pan * math.pi / 4.0 + math.pi / 4.0
            left_gain = math.cos(angle)
            right_gain = math.sin(angle)
            result[:, 0] *= left_gain
            result[:, 1] *= right_gain

            return result.astype(np.float32)

    def set_param(self, param: Param, value: float) -> None:
        """Set a single parameter value and update the FX chain."""
        r: ParamRange = PARAM_RANGES[param]
        value = max(r.min_val, min(r.max_val, float(value)))
        with self._lock:
            self._params[param] = value
            self._sync_processors()
            self._board = self._build_board()

    def get_param(self, param: Param) -> float:
        with self._lock:
            return self._params[param]

    def get_all_params(self) -> dict[Param, float]:
        with self._lock:
            return dict(self._params)

    def reset_to_defaults(self) -> None:
        with self._lock:
            self._params = {p: r.default for p, r in PARAM_RANGES.items()}
            self._sync_processors()
            self._board = self._build_board()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _sync_processors(self) -> None:
        """Push current _params into each sub-processor (called under lock)."""
        self._eq.update(self._params)
        self._filter.update(self._params)
        self._reverb.update(self._params)
        self._echo.update(self._params)
        self._pitch.update(self._params)

    def _build_board(self) -> pedalboard.Pedalboard:
        """Assemble a fresh Pedalboard from the sub-processor plugins."""
        plugins: list[pedalboard.Plugin] = []
        plugins.extend(self._eq.get_plugins())
        plugins.append(self._filter.get_plugin())
        plugins.append(self._reverb.get_plugin())
        plugins.append(self._echo.get_plugin())
        plugins.append(self._pitch.get_plugin())
        return pedalboard.Pedalboard(plugins)
