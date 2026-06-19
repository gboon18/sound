# =============================================================================
# Author  : Ho San Ko
# Email   : hko@avalanche.energy
# Project : 4-Channel Music Remix Workstation
# =============================================================================

"""Stateless BPM detection utility using librosa (Section 7.4)."""

from typing import Optional

import numpy as np


class BpmDetector:
    """Stateless utility — instantiate once and share across channels.

    Class diagram (Section 6): stateless, exposes only detect().
    """

    _BPM_MIN: float = 40.0
    _BPM_MAX: float = 300.0

    def detect(self, audio: np.ndarray, sr: int) -> Optional[float]:
        """Detect BPM from a float32 numpy audio array.

        Steps (Section 7.4):
        1. Convert stereo to mono via channel mean.
        2. librosa.beat.beat_track(y=mono, sr=sr).
        3. Return None if result is 0.0 or outside [40, 300].
        4. Otherwise return detected BPM as float.

        Returns:
            Detected BPM, or None on failure (UI should prompt manual entry).
        """
        import librosa

        # Step 1: mono
        if audio.ndim == 2:
            mono = audio.mean(axis=1).astype(np.float32)
        else:
            mono = audio.astype(np.float32)

        if len(mono) == 0:
            return None

        # Step 2: detect
        # librosa < 0.10 : beat_track(y, sr)  → (scalar, beats)
        # librosa >= 0.10: beat_track(y=, sr=) → (array or scalar, beats)
        try:
            tempo, _ = librosa.beat.beat_track(y=mono, sr=sr)
        except TypeError:
            # Fallback for older librosa signatures
            tempo, _ = librosa.beat.beat_track(mono, sr)  # type: ignore[call-arg]

        # Normalise to plain float (librosa may return 0-d array or 1-d array)
        tempo_f: float
        if hasattr(tempo, "__len__"):
            tempo_f = float(tempo[0]) if len(tempo) > 0 else 0.0
        else:
            tempo_f = float(tempo)

        # Step 3: validate
        if tempo_f == 0.0 or not (self._BPM_MIN <= tempo_f <= self._BPM_MAX):
            return None

        # Step 4: return
        return tempo_f
