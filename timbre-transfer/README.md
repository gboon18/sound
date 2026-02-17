# STFT Timbre Transfer (Offline)

Offline timbre transfer using **STFT** (Short-Time Fourier Transform). The tool extracts a **time-varying spectral envelope** (“timbre”) from an input audio file and applies it to **white noise**, producing a new sound that preserves timbral color while changing the excitation.

## Inputs

- `.wav` (mono or stereo)
- `.m4a` (and other formats supported by **ffmpeg**)

If the input is not `.wav`, it is converted to `.wav` internally using ffmpeg.

## Outputs

- `.wav` (same channel count as the input unless you force channels)

---

# Quick Start

## Poetry (Python 3.12)

```bash
mkdir timbre-transfer
cd timbre-transfer
poetry init -n
poetry env use 3.12
poetry add numpy scipy soundfile
```

## ffmpeg (required for .m4a)

**macOS**
```bash
brew install ffmpeg
```

**Ubuntu / Debian**
```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

**Windows (winget)**
```powershell
winget install --id Gyan.FFmpeg -e
```

If ffmpeg is installed but not found, add its `bin` to PATH and restart your terminal/VSCode.

## Run

```bash
poetry run python scripts/timbre_transfer_stft_stereo.py \
  --in_audio "input.m4a" \
  --out_wav "out.wav"
```

---

# Parameter Explanation

These are the key defaults you listed:

## `--n_fft 2048`

FFT size (also the analysis window length in samples).

- Larger `n_fft` → better frequency detail, smoother envelopes, but worse time detail
- Smaller `n_fft` → better time detail, but noisier envelopes

At `sr = 48000`, `n_fft = 2048` corresponds to:

- Window duration ≈ `2048 / 48000 = 0.04267 s` (≈ 42.7 ms)

## `--hop 256`

Hop size (frame step) in samples.

- Smaller `hop` → more frames (more time resolution, more compute)
- Larger `hop` → fewer frames (less time resolution, less compute)

At `sr = 48000`, `hop = 256` corresponds to:

- Frame step ≈ `256 / 48000 = 0.00533 s` (≈ 5.33 ms)

## `--lifter_q 30`

Cepstral lifter cutoff in **quefrency bins** (controls how smooth the spectral envelope is).

- Smaller `lifter_q` → *smoother* envelope (more “formant-like”, less detail)
- Larger `lifter_q` → envelope keeps more fine structure (can start leaking pitch/harmonics)

Typical useful range: `20–60`.

## `--stereo_mode shared_envelope`

Stereo handling when the input has 2 channels:

- `shared_envelope` (recommended for colored noise):  
  Extract envelope from the **mid** signal `(L+R)/2`, then apply the same envelope to both channels.
  This keeps a coherent “timbre” across stereo output.

- `per_channel`:  
  Extract/apply envelopes separately to left and right.

---

# Rectangles (Time–Frequency Region Selection)

If you want “rectangles” (blocks) in the time–frequency plane, the simplest STFT-only approach is:

1. Compute STFT of the source: `Zs[f, t]`
2. Build a mask `M[f, t]` that is 1 inside your selected rectangles and 0 elsewhere
3. Extract the timbre envelope using only the masked region:

```
mag_s = abs(Zs) * M
```

Then apply the extracted envelope to noise as usual.

## Rectangle definition (recommended)

Define each rectangle in **seconds and Hz**:

- `(t0_sec, t1_sec, f0_hz, f1_hz)`

Example:
- Use 0.50–1.20 seconds
- Use 300–3000 Hz

```
(0.50, 1.20, 300.0, 3000.0)
```

## Converting rectangles to STFT indices

Let:

- sample rate `sr`
- FFT size `n_fft`
- hop size `hop`
- frequency bins count `F = n_fft/2 + 1`

Then:

**Time frame index**
```
t_idx = floor(t_sec * sr / hop)
```

**Frequency bin index**
```
f_idx = floor(f_hz * n_fft / sr)
```

Clamp indices into valid ranges:
- `0 <= t_idx < num_frames`
- `0 <= f_idx < F`

## Minimal code to add rectangles (drop-in snippet)

Add this helper:

```python
from typing import List, Tuple
RectHz = Tuple[float, float, float, float]  # (t0, t1, f0, f1)

def rects_hz_to_mask(rects: List[RectHz], sr: int, n_fft: int, hop: int, F: int, T: int) -> np.ndarray:
    mask = np.zeros((F, T), dtype=np.float32)
    for t0, t1, f0, f1 in rects:
        t0i = int(np.floor(t0 * sr / hop))
        t1i = int(np.floor(t1 * sr / hop))
        f0i = int(np.floor(f0 * n_fft / sr))
        f1i = int(np.floor(f1 * n_fft / sr))

        t0i = max(0, min(T, t0i))
        t1i = max(0, min(T, t1i))
        f0i = max(0, min(F, f0i))
        f1i = max(0, min(F, f1i))

        if t1i > t0i and f1i > f0i:
            mask[f0i:f1i, t0i:t1i] = 1.0
    return mask
```

Then, inside `timbre_transfer_envelope_mono` after computing `Zs`:

```python
F, T = Zs.shape
rects = [(0.50, 1.20, 300.0, 3000.0)]  # example
M = rects_hz_to_mask(rects, sr, n_fft, hop, F, T)

mag_s = np.abs(Zs) * M
mag_s = mag_s + 1e-8 * (1.0 - M)  # avoid log(0) outside rectangles
env_s = spectral_envelope_cepstrum(mag_s, lifter_quefrency_bins)
```

If you supply **no rectangles**, set `M = 1` everywhere to use the full signal.

## How to pick rectangle values without UI

Practical method:

1. Start with broad frequency bounds (e.g. 100–8000 Hz).
2. Sweep time bounds in seconds (e.g. select a short region where the timbre is clear).
3. Listen to outputs and refine.

You can also print the STFT frame count to understand time resolution:
- Each frame is about `hop/sr` seconds (≈ 5.33 ms at default settings).

---

# Notes

- The tool accepts `.wav` and `.m4a` input.
- By default, it keeps the original channel count (mono stays mono; stereo stays stereo).
- You can force output channels during conversion using `--force_channels 1` or `--force_channels 2`.

