from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np

from audio_rect_synth.core.rectangle_fit import render_rectangle_magnitude
from audio_rect_synth.core.rectangle_model import RectangleFunction, RectangleModel
from audio_rect_synth.core.stft import STFTConfig, freq_bounds_to_bin_slice, invert_stft, time_bounds_to_frame_slice

ReconstructionMode = Literal["rectangles", "masked_source", "mix", "remove"]


@dataclass(frozen=True)
class ReconstructionResult:
    waveform: np.ndarray
    stft: np.ndarray
    magnitude: np.ndarray
    mode: ReconstructionMode


def rectangles_to_complex_stft(
    rectangles: Iterable[RectangleFunction],
    source_zxx: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    *,
    aggregation: str = "max",
) -> tuple[np.ndarray, np.ndarray]:
    """Convert rectangle functions into a complex STFT using source phase."""

    source = np.asarray(source_zxx, dtype=np.complex64)
    magnitude = render_rectangle_magnitude(rectangles, freqs, times, aggregation=aggregation)
    if magnitude.shape != source.shape:
        raise ValueError("Rendered rectangle magnitude shape must match source_zxx shape.")

    phase = np.exp(1j * np.angle(source)).astype(np.complex64)
    return (magnitude * phase).astype(np.complex64), magnitude


def reconstruct_from_rectangles(
    model: RectangleModel,
    source_zxx: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    *,
    target_length: int | None = None,
    mode: ReconstructionMode = "rectangles",
    normalize: bool = True,
) -> ReconstructionResult:
    """Reconstruct waveform from rectangle functions.

    modes:
    - rectangles: synthesize only fitted rectangle STFT values.
    - masked_source: keep original complex source values only where rectangles are active.
    - mix: add fitted rectangles to the original STFT.
    - remove: remove rectangle-active bins from the original STFT.
    """

    model.validate()
    config = STFTConfig(
        sample_rate=model.sample_rate,
        n_fft=model.n_fft,
        hop_length=model.hop_length,
        window=model.window,
    )

    source = np.asarray(source_zxx, dtype=np.complex64)
    rectangle_stft, rectangle_magnitude = rectangles_to_complex_stft(model.rectangles, source, freqs, times)
    active = rectangle_magnitude > 0.0

    if mode == "rectangles":
        output_stft = rectangle_stft
    elif mode == "masked_source":
        output_stft = np.where(active, source, np.complex64(0.0)).astype(np.complex64)
    elif mode == "mix":
        output_stft = (source + rectangle_stft).astype(np.complex64)
    elif mode == "remove":
        output_stft = np.where(active, np.complex64(0.0), source).astype(np.complex64)
    else:
        raise ValueError(f"Unsupported reconstruction mode: {mode!r}")

    waveform = invert_stft(output_stft, config, target_length=target_length)
    if normalize:
        waveform = normalize_peak(waveform)

    return ReconstructionResult(
        waveform=waveform,
        stft=output_stft,
        magnitude=rectangle_magnitude,
        mode=mode,
    )


def normalize_peak(samples: np.ndarray, *, peak: float = 0.98) -> np.ndarray:
    waveform = np.asarray(samples, dtype=np.float32)
    max_abs = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if max_abs <= 0.0 or max_abs <= peak:
        return np.ascontiguousarray(waveform, dtype=np.float32)
    return np.ascontiguousarray(waveform * (float(peak) / max_abs), dtype=np.float32)


def synthesize_rectangles_oscillator_bank(
    rectangles: Iterable[RectangleFunction],
    *,
    sample_rate: int,
    duration_seconds: float,
    tones_per_rectangle: int = 8,
    normalize: bool = True,
) -> np.ndarray:
    """Experimental direct rectangle-to-wave synthesis.

    Each rectangle becomes a short additive waveform spread across its frequency band.
    This is useful for testing a purely synthetic interpretation of rectangle functions.
    The STFT-phase reconstruction path generally sounds more natural.
    """

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    if duration_seconds <= 0.0:
        raise ValueError("duration_seconds must be positive.")
    if tones_per_rectangle <= 0:
        raise ValueError("tones_per_rectangle must be positive.")

    total_samples = int(round(sample_rate * duration_seconds))
    output = np.zeros(total_samples, dtype=np.float32)

    for rectangle in rectangles:
        rect = rectangle.normalized()
        start_sample = max(0, int(round(rect.t_start * sample_rate)))
        end_sample = min(total_samples, int(round(rect.t_end * sample_rate)))
        if end_sample <= start_sample:
            continue

        length = end_sample - start_sample
        local_time = np.arange(length, dtype=np.float32) / float(sample_rate)
        envelope = _raised_cosine_envelope(length)
        frequency_count = max(1, int(tones_per_rectangle))
        if rect.f_high <= rect.f_low:
            frequencies = np.array([rect.f_low], dtype=np.float32)
        else:
            frequencies = np.linspace(rect.f_low, rect.f_high, frequency_count, dtype=np.float32)

        chunk = np.zeros(length, dtype=np.float32)
        for index, frequency in enumerate(frequencies):
            if frequency <= 0.0 or frequency >= sample_rate / 2.0:
                continue
            phase = _deterministic_phase(rect, index)
            chunk += np.sin(2.0 * np.pi * float(frequency) * local_time + phase).astype(np.float32)

        chunk *= (float(rect.amplitude) / max(1, frequency_count)) * envelope
        output[start_sample:end_sample] += chunk.astype(np.float32)

    if normalize:
        output = normalize_peak(output)
    return np.ascontiguousarray(output, dtype=np.float32)


def active_rectangle_mask(
    rectangles: Iterable[RectangleFunction],
    freqs: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    mask = np.zeros((len(freqs), len(times)), dtype=bool)
    for rectangle in rectangles:
        rect = rectangle.normalized()
        f_slice = freq_bounds_to_bin_slice(freqs, rect.f_low, rect.f_high)
        t_slice = time_bounds_to_frame_slice(times, rect.t_start, rect.t_end)
        mask[f_slice, t_slice] = True
    return mask


def _raised_cosine_envelope(length: int) -> np.ndarray:
    if length <= 1:
        return np.ones(max(0, length), dtype=np.float32)
    fade = min(length // 2, max(1, int(round(length * 0.1))))
    envelope = np.ones(length, dtype=np.float32)
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, fade, dtype=np.float32))
    envelope[:fade] = ramp
    envelope[-fade:] = ramp[::-1]
    return envelope


def _deterministic_phase(rectangle: RectangleFunction, index: int) -> float:
    seed = (
        int(round(rectangle.t_start * 1000.0))
        ^ int(round(rectangle.t_end * 1000.0))
        ^ int(round(rectangle.f_low))
        ^ int(round(rectangle.f_high))
        ^ int(index * 2654435761)
    ) & 0xFFFFFFFF
    return float((seed / 0xFFFFFFFF) * 2.0 * np.pi)
