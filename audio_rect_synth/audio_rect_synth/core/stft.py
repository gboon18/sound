from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class STFTConfig:
    sample_rate: int
    n_fft: int = 4096
    hop_length: int = 1024
    window: str = "hann"

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        if self.n_fft <= 0:
            raise ValueError("n_fft must be positive.")
        if self.hop_length <= 0:
            raise ValueError("hop_length must be positive.")
        if self.hop_length > self.n_fft:
            raise ValueError("hop_length must be <= n_fft.")

    @property
    def noverlap(self) -> int:
        return int(self.n_fft - self.hop_length)

    @property
    def frame_duration_seconds(self) -> float:
        return float(self.hop_length) / float(self.sample_rate)

    @property
    def fft_bin_width_hz(self) -> float:
        return float(self.sample_rate) / float(self.n_fft)


def compute_stft(samples: np.ndarray, config: STFTConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return freqs, times, complex STFT for a mono waveform."""

    waveform = np.asarray(samples, dtype=np.float32)
    if waveform.ndim != 1:
        raise ValueError("compute_stft expects a mono 1-D waveform.")

    freqs, times, zxx = signal.stft(
        waveform,
        fs=config.sample_rate,
        window=config.window,
        nperseg=config.n_fft,
        noverlap=config.noverlap,
        nfft=config.n_fft,
        detrend=False,
        return_onesided=True,
        boundary="zeros",
        padded=True,
    )
    return freqs.astype(np.float64), times.astype(np.float64), zxx.astype(np.complex64)


def invert_stft(zxx: np.ndarray, config: STFTConfig, *, target_length: int | None = None) -> np.ndarray:
    """Invert a complex one-sided STFT and optionally trim/pad to target_length."""

    matrix = np.asarray(zxx, dtype=np.complex64)
    if matrix.ndim != 2:
        raise ValueError("zxx must be a 2-D complex STFT matrix.")

    _, waveform = signal.istft(
        matrix,
        fs=config.sample_rate,
        window=config.window,
        nperseg=config.n_fft,
        noverlap=config.noverlap,
        nfft=config.n_fft,
        input_onesided=True,
        boundary=True,
    )
    result = np.asarray(waveform, dtype=np.float32)

    if target_length is not None:
        target = int(target_length)
        if target < 0:
            raise ValueError("target_length must be non-negative.")
        if result.shape[0] > target:
            result = result[:target]
        elif result.shape[0] < target:
            result = np.pad(result, (0, target - result.shape[0]))

    return np.ascontiguousarray(result, dtype=np.float32)


def magnitude_to_db(magnitude: np.ndarray, *, floor_db: float = -120.0) -> np.ndarray:
    """Convert a non-negative magnitude spectrogram to decibels."""

    mag = np.asarray(magnitude, dtype=np.float32)
    if np.any(mag < 0):
        raise ValueError("magnitude values must be non-negative.")
    reference = max(float(np.max(mag)), 1e-12)
    db = 20.0 * np.log10(np.maximum(mag, 1e-12) / reference)
    return np.maximum(db, float(floor_db)).astype(np.float32)


def stft_to_db(zxx: np.ndarray, *, floor_db: float = -120.0) -> np.ndarray:
    return magnitude_to_db(np.abs(zxx), floor_db=floor_db)


def db_to_magnitude(db: np.ndarray, reference: float = 1.0) -> np.ndarray:
    return np.asarray(reference * np.power(10.0, np.asarray(db, dtype=np.float32) / 20.0), dtype=np.float32)


def time_bounds_to_frame_slice(times: np.ndarray, t_start: float, t_end: float) -> slice:
    """Return a non-empty frame slice covering [t_start, t_end]."""

    if t_end < t_start:
        t_start, t_end = t_end, t_start
    time_axis = np.asarray(times, dtype=np.float64)
    if time_axis.ndim != 1 or time_axis.size == 0:
        raise ValueError("times must be a non-empty 1-D array.")

    start = int(np.searchsorted(time_axis, max(float(t_start), float(time_axis[0])), side="left"))
    end = int(np.searchsorted(time_axis, min(float(t_end), float(time_axis[-1])), side="right"))
    start = max(0, min(start, time_axis.size - 1))
    end = max(start + 1, min(end, time_axis.size))
    return slice(start, end)


def freq_bounds_to_bin_slice(freqs: np.ndarray, f_low: float, f_high: float) -> slice:
    """Return a non-empty frequency-bin slice covering [f_low, f_high]."""

    if f_high < f_low:
        f_low, f_high = f_high, f_low
    freq_axis = np.asarray(freqs, dtype=np.float64)
    if freq_axis.ndim != 1 or freq_axis.size == 0:
        raise ValueError("freqs must be a non-empty 1-D array.")

    start = int(np.searchsorted(freq_axis, max(float(f_low), float(freq_axis[0])), side="left"))
    end = int(np.searchsorted(freq_axis, min(float(f_high), float(freq_axis[-1])), side="right"))
    start = max(0, min(start, freq_axis.size - 1))
    end = max(start + 1, min(end, freq_axis.size))
    return slice(start, end)


def slice_center_time(times: np.ndarray, frame_slice: slice) -> Tuple[float, float]:
    axis = np.asarray(times, dtype=np.float64)
    start = frame_slice.start or 0
    stop = frame_slice.stop or axis.size
    start = max(0, min(start, axis.size - 1))
    stop = max(start + 1, min(stop, axis.size))
    return float(axis[start]), float(axis[stop - 1])


def slice_freq_bounds(freqs: np.ndarray, bin_slice: slice) -> Tuple[float, float]:
    axis = np.asarray(freqs, dtype=np.float64)
    start = bin_slice.start or 0
    stop = bin_slice.stop or axis.size
    start = max(0, min(start, axis.size - 1))
    stop = max(start + 1, min(stop, axis.size))
    return float(axis[start]), float(axis[stop - 1])
