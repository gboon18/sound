# =============================================================================
# Author  : Ho San Ko
# Email   : hko@avalanche.energy
# Project : 4-Channel Music Remix Workstation
# =============================================================================

"""Mix bus — sums all channel outputs and applies master volume (Section 7.6).

Phase 10 optimisation: mix() accepts an optional pre-allocated *out* array so
the audio callback can reuse a single buffer across callbacks instead of
allocating a new one each time.
"""

from typing import Optional

import numpy as np

from constants import NUM_CHANNELS


class MixBus:
    """Sums N stereo channel buffers, applies master volume, hard-clips.

    Attributes (Section 6 class diagram):
        _master_volume  Linear gain applied after summing (0.0–1.0).
        _num_channels   Number of channels to mix.
    """

    def __init__(self, num_channels: int = NUM_CHANNELS) -> None:
        self._num_channels = num_channels
        self._master_volume: float = 1.0

    # ── Public API ────────────────────────────────────────────────────────────

    def mix(
        self,
        buffers: list[np.ndarray],
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Sum *buffers*, apply master volume, clip to [-1, 1].

        Steps (Section 7.6):
        1. np.stack → (N, frames, 2).
        2. sum along axis 0 → (frames, 2).
        3. multiply by master volume.
        4. np.clip to [-1.0, 1.0] (in-place when *out* is provided).
        5. Return float32 result.

        Args:
            buffers: List of (frames, 2) float32 arrays, one per channel.
            out:     Optional pre-allocated (frames, 2) float32 array.
                     When supplied, results are written in-place and *out*
                     is returned — avoids one heap allocation per callback.
        Returns:
            Mixed (frames, 2) float32 array.
        """
        if not buffers:
            raise ValueError("mix() requires at least one buffer")

        stacked: np.ndarray = np.stack(buffers)           # (N, frames, 2)
        mixed: np.ndarray   = stacked.sum(axis=0)         # (frames, 2)
        mixed *= self._master_volume

        if out is not None:
            np.clip(mixed, -1.0, 1.0, out=out)
            return out
        else:
            return np.clip(mixed, -1.0, 1.0).astype(np.float32)

    def set_master_vol(self, volume: float) -> None:
        """Set master volume, clamped to [0.0, 1.0]."""
        self._master_volume = max(0.0, min(1.0, float(volume)))

    def get_master_vol(self) -> float:
        return self._master_volume
