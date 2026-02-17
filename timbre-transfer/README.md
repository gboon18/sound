# STFT Timbre Transfer (Offline)

Offline timbre transfer using **STFT** (Short-Time Fourier Transform). The tool extracts a **time-varying spectral envelope** (“timbre”) from an input audio file and applies it to **white noise**, producing a new sound that preserves timbral color while changing the excitation.

## Inputs

- `.wav` (mono or stereo)
- `.m4a` (and other formats supported by **ffmpeg**)

If the input is not `.wav`, it is converted to `.wav` internally using ffmpeg.

## Outputs

- `.wav` (same channel count as the input unless you force channels)

---

# Poetry setup (Python 3.12)

```bash
poetry init -n
poetry env use 3.12
poetry add numpy scipy soundfile
```

## Optional (for spectrogram plotting)

If you use `--plot`, install:

```bash
poetry add matplotlib pillow
```

---

# ffmpeg / ffprobe (required for .m4a)

Install ffmpeg (ffprobe is included).

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

---

# Sample rate policy (best practice)

- If you pass a `.wav`, its sample rate is used as-is.
- If you pass a non-wav input (e.g. `.m4a`), the tool converts it to `.wav` **keeping the original sample rate** (detected via `ffprobe`), unless you explicitly request resampling.

To resample during conversion:

```bash
poetry run python scripts/timbre_transfer_stft_stereo.py --in_audio input.m4a --out_wav out.wav --sr 44100
```

Default behavior is `--sr 0` which means “keep original”.

---

# Run (basic)

```bash
poetry run python scripts/timbre_transfer_stft_stereo.py \
  --in_audio "input.m4a" \
  --out_wav "out.wav"
```

---

# Key parameters

## `--n_fft 2048`
FFT size / window length in samples. Larger = better frequency detail, worse time detail.

At 48 kHz: ~42.7 ms window.  
At 44.1 kHz: ~46.4 ms window.

## `--hop 256`
Frame step in samples. Smaller = more time resolution, more compute.

At 48 kHz: ~5.33 ms step.  
At 44.1 kHz: ~5.80 ms step.

## `--lifter_q 30`
Cepstral lifter cutoff (envelope smoothness). Typical range: 20–60.

Smaller = smoother envelope.  
Larger = more fine structure (can leak harmonics).

## `--stereo_mode shared_envelope`
If stereo input:

- `shared_envelope` (recommended): extract envelope from mid (L+R)/2 and apply to both channels.
- `per_channel`: extract/apply independently per channel.

---

# Rectangle selection (STFT-only)

You can restrict **where** timbre is extracted from using rectangles in the time–frequency plane.

Rectangles are defined in **seconds and Hz**:

- `t0,t1,f0,f1`

Example: use only 0.50–1.20 s and 300–3000 Hz for timbre extraction:

```bash
poetry run python scripts/timbre_transfer_stft_stereo.py \
  --in_audio input.wav \
  --out_wav out.wav \
  --rect "0.50,1.20,300,3000"
```

You can repeat `--rect` multiple times; selections are unioned:

```bash
--rect "0.50,1.20,300,3000" --rect "2.00,2.50,800,6000"
```

If you supply **no** rectangles, the full signal is used.

---

# Spectrogram canvas (single PNG)

To visualize:

- input spectrogram + rectangle overlays
- excitation spectrogram (white noise)
- output spectrogram (after transfer)

Use `--plot`:

```bash
poetry run python scripts/timbre_transfer_stft_stereo.py \
  --in_audio input.m4a \
  --out_wav out.wav \
  --rect "0.50,1.20,300,3000" \
  --plot --plot_png spectrogram_canvas.png
```

This writes **one** image file `spectrogram_canvas.png` that contains the three spectrograms stacked vertically.

