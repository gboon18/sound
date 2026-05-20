from __future__ import annotations

import numpy as np


class PlaybackError(RuntimeError):
    pass


def play_audio(samples: np.ndarray, sample_rate: int, *, blocking: bool = False) -> None:
    """Play audio through sounddevice if available."""

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")

    try:
        import sounddevice as sd
    except Exception as exc:  # noqa: BLE001
        raise PlaybackError("sounddevice is not installed or PortAudio is unavailable.") from exc

    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim == 2 and audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
        audio = audio.T
    elif audio.ndim > 2:
        raise ValueError("samples must be 1-D or 2-D.")

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak

    sd.play(audio, samplerate=int(sample_rate), blocking=blocking)


def stop_audio() -> None:
    try:
        import sounddevice as sd
    except Exception as exc:  # noqa: BLE001
        raise PlaybackError("sounddevice is not installed or PortAudio is unavailable.") from exc
    sd.stop()
