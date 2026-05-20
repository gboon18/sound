from __future__ import annotations

import json

import numpy as np

from audio_rect_synth.core.rectangle_fit import RectangleFitSettings, fit_rectangle_model
from audio_rect_synth.core.rectangle_model import RectangleModel, TimeFrequencySelection
from audio_rect_synth.core.reconstruct import reconstruct_from_rectangles, synthesize_rectangles_oscillator_bank
from audio_rect_synth.core.stft import STFTConfig, compute_stft, invert_stft


def _synthetic_signal(sample_rate: int = 8000, duration: float = 1.0) -> np.ndarray:
    t = np.arange(int(sample_rate * duration), dtype=np.float32) / float(sample_rate)
    envelope = np.ones_like(t)
    envelope[:200] = np.linspace(0, 1, 200, dtype=np.float32)
    envelope[-200:] = np.linspace(1, 0, 200, dtype=np.float32)
    return (0.4 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 880 * t)) * envelope


def test_stft_roundtrip() -> None:
    sample_rate = 8000
    x = _synthetic_signal(sample_rate)
    config = STFTConfig(sample_rate=sample_rate, n_fft=512, hop_length=128)
    freqs, times, zxx = compute_stft(x, config)
    assert zxx.shape == (freqs.size, times.size)
    y = invert_stft(zxx, config, target_length=x.shape[0])
    rmse = float(np.sqrt(np.mean((x - y) ** 2)))
    assert rmse < 1e-4


def test_rectangle_fit_and_reconstruct() -> None:
    sample_rate = 8000
    x = _synthetic_signal(sample_rate)
    config = STFTConfig(sample_rate=sample_rate, n_fft=512, hop_length=128)
    freqs, times, zxx = compute_stft(x, config)
    selection = TimeFrequencySelection(t_start=0.0, t_end=0.8, f_low=300.0, f_high=1200.0, region_id="test")
    settings = RectangleFitSettings(min_rects=1, max_rects=4, slice_duration_ms=50.0, slice_overlap=0.5)
    fit = fit_rectangle_model(zxx, freqs, times, config, [selection], settings)
    assert fit.rectangle_count > 0
    assert fit.fitted_magnitude.shape == zxx.shape
    assert fit.mean_squared_error >= 0.0

    reconstruction = reconstruct_from_rectangles(fit.model, zxx, freqs, times, target_length=x.shape[0])
    assert reconstruction.waveform.shape == x.shape
    assert np.all(np.isfinite(reconstruction.waveform))


def test_rectangle_model_json_roundtrip() -> None:
    sample_rate = 8000
    x = _synthetic_signal(sample_rate)
    config = STFTConfig(sample_rate=sample_rate, n_fft=512, hop_length=128)
    freqs, times, zxx = compute_stft(x, config)
    selection = TimeFrequencySelection(t_start=0.0, t_end=0.2, f_low=300.0, f_high=1200.0, region_id="json")
    settings = RectangleFitSettings(min_rects=1, max_rects=2, slice_duration_ms=50.0, slice_overlap=0.0)
    fit = fit_rectangle_model(zxx, freqs, times, config, [selection], settings)
    data = fit.model.to_dict()
    encoded = json.dumps(data)
    decoded = RectangleModel.from_dict(json.loads(encoded))
    assert len(decoded.rectangles) == len(fit.model.rectangles)
    assert decoded.sample_rate == sample_rate


def test_experimental_oscillator_synthesis() -> None:
    sample_rate = 8000
    x = _synthetic_signal(sample_rate)
    config = STFTConfig(sample_rate=sample_rate, n_fft=512, hop_length=128)
    freqs, times, zxx = compute_stft(x, config)
    selection = TimeFrequencySelection(t_start=0.0, t_end=0.2, f_low=300.0, f_high=1200.0, region_id="osc")
    settings = RectangleFitSettings(min_rects=1, max_rects=2, slice_duration_ms=50.0, slice_overlap=0.0)
    fit = fit_rectangle_model(zxx, freqs, times, config, [selection], settings)
    y = synthesize_rectangles_oscillator_bank(
        fit.model.rectangles,
        sample_rate=sample_rate,
        duration_seconds=1.0,
        tones_per_rectangle=4,
    )
    assert y.shape == x.shape
    assert np.all(np.isfinite(y))
