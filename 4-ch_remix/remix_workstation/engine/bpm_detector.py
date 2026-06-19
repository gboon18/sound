# =============================================================================
# Author  : Ho San Ko
# Email   : hko@avalanche.energy
# Project : 4-Channel Music Remix Workstation
# =============================================================================

"""Stateless BPM detection utility using librosa."""

from typing import Optional

import numpy as np


class BpmDetector:
    """Stateless utility class — instantiate once and share across all channels.

    Class diagram (Section 6): stateless, exposes only detect().
    """

    def detect(self, audio: np.ndarray, sr: int) -> Optional[float]:
        """Detect BPM from a float32 numpy audio array.

        Steps (Section 7.4):
        1. Convert stereo to mono via mean.
        2. librosa.beat.beat_track(y=mono, sr=sr).
        3. Return None if result is 0.0 or outside [40, 300].
        4. Otherwise return detected BPM as float.

        Returns:
            Detected BPM, or None on failure (caller should prompt manual entry).
        """
        import librosa  # deferred: only needed at load time

        if audio.ndim == 2:
            mono = audio.mean(axis=1)
        else:
            mono = audio

        tempo, _ = librosa.beat.beat_track(y=mono, sr=sr)
        # librosa may return an array; extract scalar
        if hasattr(tempo, "__len__"):
            tempo = float(tempo[0]) if len(tempo) > 0 else 0.0
        else:
            tempo = float(tempo)

        if tempo == 0.0 or not (40.0 <= tempo <= 300.0):
            return None
        return tempo
