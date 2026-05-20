from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import subprocess
import tempfile
from typing import Optional

import numpy as np


SUPPORTED_INPUT_SUFFIXES = {".wav", ".mp3", ".m4a"}


@dataclass(frozen=True)
class AudioData:
    """Audio stored as float32 samples with shape (channels, samples)."""

    samples: np.ndarray
    sample_rate: int
    path: Optional[Path] = None

    def __post_init__(self) -> None:
        if self.samples.ndim != 2:
            raise ValueError("AudioData.samples must have shape (channels, samples).")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")

    @property
    def duration_seconds(self) -> float:
        return float(self.samples.shape[1]) / float(self.sample_rate)

    @property
    def channel_count(self) -> int:
        return int(self.samples.shape[0])

    def mono(self) -> np.ndarray:
        if self.samples.shape[0] == 1:
            return np.asarray(self.samples[0], dtype=np.float32)
        return np.asarray(np.mean(self.samples, axis=0), dtype=np.float32)


def load_audio(path: str | Path, *, mono: bool = False, target_sample_rate: int | None = None) -> AudioData:
    """Load a .wav, .mp3, or .m4a file as float32 audio.

    The loader tries soundfile first, then PyAV, then an ffmpeg CLI fallback.
    """

    file_path = Path(path).expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_INPUT_SUFFIXES:
        allowed = ", ".join(sorted(SUPPORTED_INPUT_SUFFIXES))
        raise ValueError(f"Unsupported audio extension {suffix!r}. Supported: {allowed}.")

    errors: list[str] = []
    samples: np.ndarray | None = None
    sample_rate: int | None = None

    for decoder in (_decode_with_soundfile, _decode_with_pyav, _decode_with_ffmpeg_cli):
        try:
            samples, sample_rate = decoder(file_path)
            break
        except Exception as exc:  # noqa: BLE001 - collect decoder failures for useful error message.
            errors.append(f"{decoder.__name__}: {exc}")

    if samples is None or sample_rate is None:
        joined = "\n".join(errors)
        raise RuntimeError(f"Could not decode audio file: {file_path}\n{joined}")

    samples = _ensure_channels_first_float32(samples)

    if target_sample_rate is not None and int(target_sample_rate) != int(sample_rate):
        samples = resample_audio(samples, sample_rate, int(target_sample_rate))
        sample_rate = int(target_sample_rate)

    if mono:
        samples = np.mean(samples, axis=0, keepdims=True).astype(np.float32)

    return AudioData(samples=samples, sample_rate=int(sample_rate), path=file_path)


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int) -> None:
    """Write float audio to a WAV file.

    samples may be shape (samples,), (channels, samples), or (samples, channels).
    """

    import soundfile as sf

    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim == 1:
        data = audio
    elif audio.ndim == 2:
        if audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
            data = audio.T
        else:
            data = audio
    else:
        raise ValueError("samples must be 1-D or 2-D audio data.")

    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak > 1.0:
        data = data / peak

    sf.write(output_path, data, int(sample_rate))


def resample_audio(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample channels-first audio with scipy.signal.resample_poly."""

    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("source_rate and target_rate must be positive.")
    if source_rate == target_rate:
        return np.asarray(samples, dtype=np.float32)

    from scipy.signal import resample_poly

    gcd = math.gcd(int(source_rate), int(target_rate))
    up = int(target_rate) // gcd
    down = int(source_rate) // gcd
    return np.asarray(resample_poly(samples, up, down, axis=1), dtype=np.float32)


def _decode_with_soundfile(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    # soundfile returns shape (samples, channels). Store as (channels, samples).
    return np.ascontiguousarray(data.T), int(sample_rate)


def _decode_with_pyav(path: Path) -> tuple[np.ndarray, int]:
    import av

    container = av.open(str(path))
    try:
        stream = next((item for item in container.streams if item.type == "audio"), None)
        if stream is None:
            raise ValueError("No audio stream found.")

        chunks: list[np.ndarray] = []
        sample_rate = int(stream.rate or 0)

        for frame in container.decode(stream):
            if sample_rate <= 0 and frame.sample_rate:
                sample_rate = int(frame.sample_rate)
            chunks.append(_pyav_frame_to_channels_first_float32(frame))

        if not chunks:
            raise ValueError("No audio frames decoded.")
        if sample_rate <= 0:
            raise ValueError("Could not determine sample rate.")

        channel_count = max(chunk.shape[0] for chunk in chunks)
        normalized_chunks = [_match_channel_count(chunk, channel_count) for chunk in chunks]
        return np.concatenate(normalized_chunks, axis=1).astype(np.float32), sample_rate
    finally:
        container.close()


def _decode_with_ffmpeg_cli(path: Path) -> tuple[np.ndarray, int]:
    """Use an installed ffmpeg binary as a last-resort decoder."""

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        command = [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(path),
            "-acodec",
            "pcm_f32le",
            tmp.name,
        ]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return _decode_with_soundfile(Path(tmp.name))


def _pyav_frame_to_channels_first_float32(frame: object) -> np.ndarray:
    arr = np.asarray(frame.to_ndarray())
    channels = int(getattr(frame.layout, "nb_channels", 0) or 1)

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim == 2:
        if arr.shape[0] == channels:
            pass
        elif arr.shape[1] == channels:
            arr = arr.T
        elif arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
            pass
        else:
            arr = arr.T
    else:
        raise ValueError(f"Unexpected audio frame array shape: {arr.shape}")

    if np.issubdtype(arr.dtype, np.floating):
        data = arr.astype(np.float32, copy=False)
    elif np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        scale = max(abs(info.min), abs(info.max))
        data = arr.astype(np.float32) / float(scale)
    else:
        data = arr.astype(np.float32)

    return np.ascontiguousarray(data)


def _match_channel_count(chunk: np.ndarray, channel_count: int) -> np.ndarray:
    if chunk.shape[0] == channel_count:
        return chunk
    if chunk.shape[0] == 1:
        return np.repeat(chunk, channel_count, axis=0)
    if chunk.shape[0] > channel_count:
        return chunk[:channel_count]
    pad_count = channel_count - chunk.shape[0]
    padding = np.repeat(chunk[-1:, :], pad_count, axis=0)
    return np.concatenate([chunk, padding], axis=0)


def _ensure_channels_first_float32(samples: np.ndarray) -> np.ndarray:
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim == 1:
        return np.ascontiguousarray(audio.reshape(1, -1))
    if audio.ndim != 2:
        raise ValueError("Decoded audio must be 1-D or 2-D.")
    if audio.shape[0] <= 8 and audio.shape[1] >= audio.shape[0]:
        return np.ascontiguousarray(audio)
    if audio.shape[1] <= 8 and audio.shape[0] > audio.shape[1]:
        return np.ascontiguousarray(audio.T)
    return np.ascontiguousarray(audio)
