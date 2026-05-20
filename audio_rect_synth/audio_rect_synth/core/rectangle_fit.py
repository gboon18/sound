from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from audio_rect_synth.core.rectangle_model import RectangleFunction, RectangleModel, TimeFrequencySelection
from audio_rect_synth.core.stft import STFTConfig, freq_bounds_to_bin_slice, time_bounds_to_frame_slice


@dataclass(frozen=True)
class RectangleFitSettings:
    min_rects: int = 1
    max_rects: int = 6
    slice_duration_ms: float = 40.0
    slice_overlap: float = 0.5
    threshold_quantile: float = 0.75
    min_peak_fraction: float = 0.08

    def validate(self) -> None:
        if self.min_rects <= 0:
            raise ValueError("min_rects must be positive.")
        if self.max_rects < self.min_rects:
            raise ValueError("max_rects must be >= min_rects.")
        if self.slice_duration_ms <= 0.0:
            raise ValueError("slice_duration_ms must be positive.")
        if not 0.0 <= self.slice_overlap < 1.0:
            raise ValueError("slice_overlap must be in [0, 1).")
        if not 0.0 <= self.threshold_quantile <= 1.0:
            raise ValueError("threshold_quantile must be in [0, 1].")
        if self.min_peak_fraction < 0.0:
            raise ValueError("min_peak_fraction must be non-negative.")


@dataclass(frozen=True)
class RectangleFitResult:
    model: RectangleModel
    fitted_magnitude: np.ndarray
    source_magnitude: np.ndarray
    mean_squared_error: float

    @property
    def rectangle_count(self) -> int:
        return len(self.model.rectangles)


def fit_rectangle_model(
    zxx: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    config: STFTConfig,
    selections: Iterable[TimeFrequencySelection],
    settings: RectangleFitSettings,
    *,
    source_path: str | None = None,
) -> RectangleFitResult:
    """Fit axis-aligned magnitude rectangles inside selected time-frequency regions."""

    settings.validate()
    complex_stft = np.asarray(zxx, dtype=np.complex64)
    if complex_stft.ndim != 2:
        raise ValueError("zxx must be a 2-D complex STFT matrix.")

    freq_axis = np.asarray(freqs, dtype=np.float64)
    time_axis = np.asarray(times, dtype=np.float64)
    if complex_stft.shape != (freq_axis.size, time_axis.size):
        raise ValueError("zxx shape must be (len(freqs), len(times)).")

    source_magnitude = np.abs(complex_stft).astype(np.float32)
    rectangles: list[RectangleFunction] = []

    for selection in selections:
        normalized = selection.normalized()
        normalized.validate()
        rectangles.extend(
            _fit_single_selection(
                source_magnitude,
                freq_axis,
                time_axis,
                normalized,
                settings,
            )
        )

    fitted_magnitude = render_rectangle_magnitude(rectangles, freq_axis, time_axis)
    active_mask = fitted_magnitude > 0.0
    if np.any(active_mask):
        mse = float(np.mean((source_magnitude[active_mask] - fitted_magnitude[active_mask]) ** 2))
    else:
        mse = float(np.mean(source_magnitude**2)) if source_magnitude.size else 0.0

    model = RectangleModel(
        sample_rate=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        window=config.window,
        rectangles=rectangles,
        source_path=source_path,
    )
    model.validate()
    return RectangleFitResult(
        model=model,
        fitted_magnitude=fitted_magnitude,
        source_magnitude=source_magnitude,
        mean_squared_error=mse,
    )


def render_rectangle_magnitude(
    rectangles: Iterable[RectangleFunction],
    freqs: np.ndarray,
    times: np.ndarray,
    *,
    aggregation: str = "max",
) -> np.ndarray:
    """Render rectangle amplitudes into an STFT-shaped magnitude matrix."""

    if aggregation not in {"max", "sum"}:
        raise ValueError("aggregation must be 'max' or 'sum'.")

    freq_axis = np.asarray(freqs, dtype=np.float64)
    time_axis = np.asarray(times, dtype=np.float64)
    magnitude = np.zeros((freq_axis.size, time_axis.size), dtype=np.float32)

    for rectangle in rectangles:
        rect = rectangle.normalized()
        f_slice = freq_bounds_to_bin_slice(freq_axis, rect.f_low, rect.f_high)
        t_slice = time_bounds_to_frame_slice(time_axis, rect.t_start, rect.t_end)
        amplitude = np.float32(max(0.0, rect.amplitude))
        if aggregation == "max":
            magnitude[f_slice, t_slice] = np.maximum(magnitude[f_slice, t_slice], amplitude)
        else:
            magnitude[f_slice, t_slice] += amplitude

    return magnitude


def _fit_single_selection(
    magnitude: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    selection: TimeFrequencySelection,
    settings: RectangleFitSettings,
) -> list[RectangleFunction]:
    f_slice = freq_bounds_to_bin_slice(freqs, selection.f_low, selection.f_high)
    t_slice = time_bounds_to_frame_slice(times, selection.t_start, selection.t_end)

    freq_count = int(f_slice.stop - f_slice.start)
    if freq_count <= 0:
        return []

    effective_min = max(1, min(settings.min_rects, freq_count))
    effective_max = max(effective_min, min(settings.max_rects, freq_count))

    rectangles: list[RectangleFunction] = []
    frame_slices = list(_iter_time_slices(times, t_slice, settings.slice_duration_ms, settings.slice_overlap))

    for slice_index, frame_slice in enumerate(frame_slices):
        patch = magnitude[f_slice, frame_slice]
        if patch.size == 0:
            continue

        spectrum = np.mean(patch, axis=1)
        band_slices = _pick_frequency_bands(
            spectrum,
            min_bands=effective_min,
            max_bands=effective_max,
            threshold_quantile=settings.threshold_quantile,
            min_peak_fraction=settings.min_peak_fraction,
        )

        t_start, t_end = _frame_slice_bounds(times, frame_slice, fallback_duration_ms=settings.slice_duration_ms)
        t_start = max(t_start, selection.t_start)
        t_end = min(max(t_end, t_start + 1e-9), selection.t_end)

        for band in band_slices:
            local_start = band.start or 0
            local_stop = band.stop or local_start + 1
            global_start = int(f_slice.start + local_start)
            global_stop = int(f_slice.start + local_stop)
            global_stop = max(global_start + 1, min(global_stop, freqs.size))

            band_patch = patch[local_start:local_stop, :]
            if band_patch.size == 0:
                continue

            amplitude = float(np.mean(band_patch))
            error = float(np.mean((band_patch - amplitude) ** 2))
            f_low = float(freqs[global_start])
            f_high = float(freqs[global_stop - 1])
            if f_high <= f_low and freqs.size > 1:
                f_high = f_low + float(np.median(np.diff(freqs)))

            rectangles.append(
                RectangleFunction(
                    t_start=float(t_start),
                    t_end=float(t_end),
                    f_low=f_low,
                    f_high=f_high,
                    amplitude=max(0.0, amplitude),
                    source_region_id=selection.region_id,
                    slice_index=slice_index,
                    error=error,
                ).normalized()
            )

    return rectangles


def _iter_time_slices(times: np.ndarray, frame_slice: slice, slice_duration_ms: float, overlap: float) -> Iterable[slice]:
    start = int(frame_slice.start or 0)
    stop = int(frame_slice.stop or times.size)
    start = max(0, min(start, times.size - 1))
    stop = max(start + 1, min(stop, times.size))

    if stop - start <= 1:
        yield slice(start, stop)
        return

    diffs = np.diff(times)
    positive_diffs = diffs[diffs > 0]
    frame_step_seconds = float(np.median(positive_diffs)) if positive_diffs.size else 1.0
    slice_seconds = float(slice_duration_ms) / 1000.0
    frames_per_slice = max(1, int(round(slice_seconds / frame_step_seconds)))
    frames_per_slice = min(frames_per_slice, stop - start)
    hop = max(1, int(round(frames_per_slice * (1.0 - overlap))))

    cursor = start
    while cursor < stop:
        current_stop = min(cursor + frames_per_slice, stop)
        yield slice(cursor, current_stop)
        if current_stop >= stop:
            break
        cursor += hop


def _pick_frequency_bands(
    spectrum: np.ndarray,
    *,
    min_bands: int,
    max_bands: int,
    threshold_quantile: float,
    min_peak_fraction: float,
) -> list[slice]:
    values = np.asarray(spectrum, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError("spectrum must be 1-D.")
    if values.size == 0:
        return []

    min_bands = max(1, min(min_bands, values.size))
    max_bands = max(min_bands, min(max_bands, values.size))

    smoothed = _smooth_spectrum(values)
    peak = float(np.max(smoothed)) if smoothed.size else 0.0

    if peak <= 0.0 or not np.isfinite(peak):
        bands = [slice(index, index + 1) for index in range(min_bands)]
        return bands

    quantile_threshold = float(np.quantile(smoothed, threshold_quantile))
    peak_threshold = peak * float(min_peak_fraction)
    threshold = max(quantile_threshold, peak_threshold)

    active = smoothed >= threshold
    bands = _connected_true_runs(active)
    bands = _drop_zero_energy_bands(bands, values)

    if not bands:
        bands = [_expand_peak_to_band(values, int(np.argmax(values)))]

    bands = _deduplicate_and_sort_bands(bands, values.size)

    while len(bands) < min_bands:
        bands = _add_or_split_band(bands, values)
        bands = _deduplicate_and_sort_bands(bands, values.size)
        if len(bands) >= values.size:
            break

    if len(bands) > max_bands:
        bands = _keep_top_energy_bands(bands, values, max_bands)

    # Final guard: order by frequency, not energy, for predictable visual overlays.
    return _deduplicate_and_sort_bands(bands, values.size)[:max_bands]


def _smooth_spectrum(values: np.ndarray) -> np.ndarray:
    if values.size < 5:
        return values.astype(np.float32)
    window_size = max(3, int(round(values.size / 128.0)))
    if window_size % 2 == 0:
        window_size += 1
    window_size = min(window_size, values.size if values.size % 2 == 1 else values.size - 1)
    if window_size <= 1:
        return values.astype(np.float32)
    kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
    padded = np.pad(values, (window_size // 2, window_size // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _connected_true_runs(mask: np.ndarray) -> list[slice]:
    runs: list[slice] = []
    start: int | None = None
    for index, flag in enumerate(mask):
        if bool(flag) and start is None:
            start = index
        elif not bool(flag) and start is not None:
            runs.append(slice(start, index))
            start = None
    if start is not None:
        runs.append(slice(start, mask.size))
    return runs


def _drop_zero_energy_bands(bands: list[slice], values: np.ndarray) -> list[slice]:
    result: list[slice] = []
    for band in bands:
        start = int(band.start or 0)
        stop = int(band.stop or start + 1)
        if float(np.sum(values[start:stop])) > 0.0:
            result.append(slice(start, stop))
    return result


def _expand_peak_to_band(values: np.ndarray, peak_index: int) -> slice:
    peak_index = max(0, min(int(peak_index), values.size - 1))
    peak = float(values[peak_index])
    if peak <= 0.0:
        return slice(peak_index, peak_index + 1)

    threshold = peak * 0.5
    start = peak_index
    while start > 0 and float(values[start - 1]) >= threshold:
        start -= 1
    stop = peak_index + 1
    while stop < values.size and float(values[stop]) >= threshold:
        stop += 1
    return slice(start, stop)


def _add_or_split_band(bands: list[slice], values: np.ndarray) -> list[slice]:
    existing = _deduplicate_and_sort_bands(bands, values.size)

    # Prefer splitting the widest/highest-energy existing band.
    splittable = [band for band in existing if int(band.stop or 0) - int(band.start or 0) >= 2]
    if splittable:
        chosen = max(splittable, key=lambda band: _band_energy(band, values))
        midpoint = int(math.floor(((chosen.start or 0) + (chosen.stop or 0)) / 2.0))
        output: list[slice] = []
        for band in existing:
            if band == chosen:
                output.append(slice(chosen.start, midpoint))
                output.append(slice(midpoint, chosen.stop))
            else:
                output.append(band)
        return output

    # If all bands are one bin, add the strongest uncovered bin.
    covered = np.zeros(values.size, dtype=bool)
    for band in existing:
        covered[band] = True
    uncovered_indices = np.flatnonzero(~covered)
    if uncovered_indices.size == 0:
        return existing
    strongest = int(uncovered_indices[np.argmax(values[uncovered_indices])])
    return existing + [slice(strongest, strongest + 1)]


def _keep_top_energy_bands(bands: list[slice], values: np.ndarray, count: int) -> list[slice]:
    ranked = sorted(bands, key=lambda band: _band_energy(band, values), reverse=True)
    return ranked[:count]


def _band_energy(band: slice, values: np.ndarray) -> float:
    start = int(band.start or 0)
    stop = int(band.stop or start + 1)
    start = max(0, min(start, values.size - 1))
    stop = max(start + 1, min(stop, values.size))
    return float(np.sum(values[start:stop]))


def _deduplicate_and_sort_bands(bands: list[slice], size: int) -> list[slice]:
    cleaned: list[tuple[int, int]] = []
    for band in bands:
        start = int(band.start or 0)
        stop = int(band.stop or start + 1)
        start = max(0, min(start, size - 1))
        stop = max(start + 1, min(stop, size))
        cleaned.append((start, stop))
    cleaned = sorted(set(cleaned))

    merged: list[tuple[int, int]] = []
    for start, stop in cleaned:
        if not merged:
            merged.append((start, stop))
            continue
        prev_start, prev_stop = merged[-1]
        if start < prev_stop:
            merged[-1] = (prev_start, max(prev_stop, stop))
        else:
            merged.append((start, stop))
    return [slice(start, stop) for start, stop in merged]


def _frame_slice_bounds(times: np.ndarray, frame_slice: slice, *, fallback_duration_ms: float) -> tuple[float, float]:
    start = int(frame_slice.start or 0)
    stop = int(frame_slice.stop or start + 1)
    start = max(0, min(start, times.size - 1))
    stop = max(start + 1, min(stop, times.size))

    t_start = float(times[start])
    t_end = float(times[stop - 1])
    if t_end <= t_start:
        diffs = np.diff(times)
        positive = diffs[diffs > 0]
        duration = float(np.median(positive)) if positive.size else float(fallback_duration_ms) / 1000.0
        t_end = t_start + max(duration, 1e-6)
    return t_start, t_end
