# STFT Timbre Transfer (Offline)

This project performs offline timbre transfer using STFT (Short-Time Fourier Transform).

It extracts the time-varying spectral envelope (timbre) from an input audio file and applies it to white noise, producing a new sound that has the same timbral structure but different excitation.

---

# Supported Input Formats

✔ .wav  
✔ .m4a  
✔ Any format supported by ffmpeg  

If input is not .wav, it is automatically converted using ffmpeg.

---

# Mono and Stereo Support

The program automatically detects:

- Mono → outputs mono
- Stereo → outputs stereo

Stereo modes:

- shared_envelope (default): extract envelope from mid (L+R)/2 and apply to both channels
- per_channel: process left and right independently

---

# How It Works

1. Convert input to WAV if necessary
2. Compute STFT
3. Extract spectral envelope using cepstral liftering
4. Generate white noise
5. Apply envelope to noise magnitude spectrum
6. Inverse STFT to reconstruct audio

Envelope application:

|Y(f,t)| = |Noise(f,t)| × Envelope(f,t)

---

# Poetry Setup (Python 3.12)

## Create project

    mkdir timbre-transfer
    cd timbre-transfer
    poetry init -n

## Use Python 3.12

    poetry env use 3.12

## Install dependencies

    poetry add numpy scipy soundfile

---

# Install ffmpeg (Required)

## macOS

    brew install ffmpeg

## Ubuntu / Debian

    sudo apt-get update
    sudo apt-get install -y ffmpeg

## Windows

    winget install --id Gyan.FFmpeg -e

Restart terminal after installation.

---

# Run

    poetry run python scripts/timbre_transfer_stft_stereo.py --in_audio "input.m4a" --out_wav "out.wav"

---

# Important Parameters

--n_fft        FFT size (default 2048)  
--hop          Hop size (default 256)  
--lifter_q     Envelope smoothness (20–60 typical)  
--mix          Envelope strength (0.0–1.0)  
--stereo_mode  shared_envelope | per_channel  

---

# Recommended Defaults

    --n_fft 2048
    --hop 256
    --lifter_q 30
    --stereo_mode shared_envelope
