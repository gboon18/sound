# Audio Rect Synth

Interactive spectrogram rectangle fitting and reconstruction tool.

This MVP lets you:

1. Load `.wav`, `.mp3`, and `.m4a` audio files.
2. Plot a dB spectrogram.
3. Draw time-frequency rectangular selections.
4. Split selections into time slices.
5. Fit each slice with a user-bounded number of frequency-band rectangle functions.
6. Reconstruct audio from fitted rectangles using the original STFT phase.
7. Preview and export reconstructed `.wav` audio and a `.json` rectangle model.

## Recommended environment

Python 3.10 through 3.14 and Poetry.

```bash
poetry install
```

This installs the package itself and creates the `audio-rect-synth` and
`rect-synth-cli` command wrappers inside Poetry's virtual environment. If either
command is not recognized, run `poetry install` again from this directory.

Poetry installs dependencies from `pyproject.toml`, not directly from
`requirements.txt`. This project already has the current `requirements.txt`
entries in `pyproject.toml`. If you update `requirements.txt` and want to import
those entries into Poetry, run this from the project directory:

```powershell
Get-Content requirements.txt |
  Where-Object { $_.Trim() -and -not $_.Trim().StartsWith("#") } |
  ForEach-Object { poetry add $_ }
```

On macOS or Linux:

```bash
grep -vE '^\s*(#|$)' requirements.txt | xargs poetry add
```

On Linux, audio playback may require PortAudio system libraries for `sounddevice`.
If playback is unavailable, export still works.

## Run the desktop app

```bash
poetry run audio-rect-synth
```

or from the source tree:

```bash
poetry run python -m audio_rect_synth.app.main
```

## CLI smoke test

The CLI is useful for batch testing without the GUI. It fits a rectangular region and exports audio/model files.

```bash
poetry run rect-synth-cli input.wav output.wav --t-start 0 --t-end 3 --f-low 100 --f-high 4000 --min-rects 1 --max-rects 6
```

## Reconstruction model

The exported JSON stores the STFT settings and each fitted rectangle:

```json
{
  "sample_rate": 44100,
  "n_fft": 4096,
  "hop_length": 1024,
  "rectangles": [
    {
      "t_start": 0.25,
      "t_end": 0.29,
      "f_low": 430.66,
      "f_high": 861.33,
      "amplitude": 0.04,
      "source_region_id": "region-1",
      "slice_index": 0
    }
  ]
}
```

## Notes on rectangle fitting

The first fitting algorithm is intentionally conservative and explainable:

- Time selections are split into slices, defaulting to 40 ms with 50% overlap.
- Each slice is reduced to a mean spectrum.
- High-energy contiguous frequency bands are identified.
- The band count is constrained by `min_rects` and `max_rects`.
- Rectangle amplitudes are estimated from the source magnitude spectrogram.
- Reconstruction uses original STFT phase for natural-sounding output.

This produces a practical baseline. Later versions can add chirp-aware slanted rectangles, complex-valued fitting, neural phase recovery, or fully synthetic oscillator-bank synthesis.
