from __future__ import annotations

import argparse
from pathlib import Path

from audio_rect_synth.core.audio_io import load_audio, write_wav
from audio_rect_synth.core.rectangle_fit import RectangleFitSettings, fit_rectangle_model
from audio_rect_synth.core.rectangle_model import TimeFrequencySelection, save_rectangle_model
from audio_rect_synth.core.reconstruct import reconstruct_from_rectangles
from audio_rect_synth.core.stft import STFTConfig, compute_stft


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit time-frequency rectangles and reconstruct audio.")
    parser.add_argument("input", type=Path, help="Input .wav, .mp3, or .m4a file.")
    parser.add_argument("output", type=Path, help="Output .wav file.")
    parser.add_argument("--model-output", type=Path, default=None, help="Optional output JSON model path.")
    parser.add_argument("--t-start", type=float, default=0.0, help="Selection start time in seconds.")
    parser.add_argument("--t-end", type=float, default=None, help="Selection end time in seconds. Default: file duration.")
    parser.add_argument("--f-low", type=float, default=20.0, help="Selection low frequency in Hz.")
    parser.add_argument("--f-high", type=float, default=None, help="Selection high frequency in Hz. Default: Nyquist.")
    parser.add_argument("--min-rects", type=int, default=1, help="Minimum rectangles per time slice.")
    parser.add_argument("--max-rects", type=int, default=6, help="Maximum rectangles per time slice.")
    parser.add_argument("--slice-ms", type=float, default=40.0, help="Time slice duration in milliseconds.")
    parser.add_argument("--slice-overlap", type=float, default=0.5, help="Time slice overlap fraction in [0, 1).")
    parser.add_argument("--n-fft", type=int, default=4096, help="STFT FFT/window size.")
    parser.add_argument("--hop-length", type=int, default=1024, help="STFT hop length in samples.")
    parser.add_argument(
        "--mode",
        choices=["rectangles", "masked_source", "mix", "remove"],
        default="rectangles",
        help="Reconstruction mode.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    audio = load_audio(args.input, mono=True)
    mono = audio.mono()
    config = STFTConfig(sample_rate=audio.sample_rate, n_fft=args.n_fft, hop_length=args.hop_length)
    freqs, times, zxx = compute_stft(mono, config)

    selection = TimeFrequencySelection(
        t_start=args.t_start,
        t_end=args.t_end if args.t_end is not None else audio.duration_seconds,
        f_low=args.f_low,
        f_high=args.f_high if args.f_high is not None else audio.sample_rate / 2.0,
        region_id="cli-selection",
    )
    settings = RectangleFitSettings(
        min_rects=args.min_rects,
        max_rects=args.max_rects,
        slice_duration_ms=args.slice_ms,
        slice_overlap=args.slice_overlap,
    )

    result = fit_rectangle_model(
        zxx,
        freqs,
        times,
        config,
        [selection],
        settings,
        source_path=str(audio.path) if audio.path is not None else None,
    )
    reconstruction = reconstruct_from_rectangles(
        result.model,
        zxx,
        freqs,
        times,
        target_length=mono.shape[0],
        mode=args.mode,
    )

    write_wav(args.output, reconstruction.waveform, audio.sample_rate)
    model_output = args.model_output or args.output.with_suffix(".rectangles.json")
    save_rectangle_model(model_output, result.model)

    print(f"Wrote audio: {args.output}")
    print(f"Wrote model: {model_output}")
    print(f"Rectangles: {result.rectangle_count}")
    print(f"MSE over active rectangles: {result.mean_squared_error:.8g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
