from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import stft, istft


def ffprobe_audio_info(in_path: Path) -> Tuple[int, int]:
    """
    Returns (sample_rate_hz, channels) for the first audio stream.
    Requires ffprobe installed (bundled with ffmpeg).
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate,channels",
        "-of", "json",
        str(in_path),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe failed:\n{p.stderr}")
    data = json.loads(p.stdout)
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError("ffprobe returned no audio streams.")
    sr = int(streams[0]["sample_rate"])
    ch = int(streams[0]["channels"])
    return sr, ch


def convert_to_wav_ffmpeg(
    in_path: Path,
    out_path: Path,
    sr: Optional[int] = None,
    channels: Optional[int] = None,
) -> None:
    """
    Convert audio to PCM WAV via ffmpeg.

    Best-practice defaults:
      - If sr is None: keep the original sample rate.
      - If channels is None: keep the original channel count.

    If you pass sr and/or channels, ffmpeg will resample / remix accordingly.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", str(in_path)]
    if channels is not None:
        cmd += ["-ac", str(int(channels))]
    if sr is not None:
        cmd += ["-ar", str(int(sr))]
    cmd += ["-c:a", "pcm_s24le", str(out_path)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{p.stderr}")


def load_audio_wav(path: Path) -> Tuple[np.ndarray, int]:
    """
    Load WAV via soundfile.

    Returns:
      x: (n,) for mono or (n, c) for multichannel
      sr
    """
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim == 1:
        return x.astype(np.float32, copy=False), sr
    return x.astype(np.float32, copy=False), sr


def write_wav(path: Path, x: np.ndarray, sr: int) -> None:
    x = np.asarray(x, dtype=np.float32)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 0.999:
        x = x / peak * 0.999
    sf.write(path, x, sr, subtype="PCM_24")


def make_excitation_white(n: int, channels: int) -> np.ndarray:
    """
    White noise excitation matching input channel count.
      mono: (n,)
      stereo: (n, 2)
    """
    rng = np.random.default_rng(0)
    w = rng.standard_normal((n, channels)).astype(np.float32)
    if channels == 1:
        return w[:, 0]
    return w


def _safe_log(a: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return np.log(np.maximum(a, eps))


def spectral_envelope_cepstrum(mag: np.ndarray, lifter_quefrency_bins: int) -> np.ndarray:
    """
    Cepstral liftering envelope:
      mag: (freq_bins, frames)
      lifter_quefrency_bins: ~20–60 typical (smaller => smoother)
    """
    log_mag = _safe_log(mag)
    cep = np.fft.irfft(log_mag, axis=0)
    cep_lift = np.zeros_like(cep)
    q = min(lifter_quefrency_bins, cep.shape[0])
    cep_lift[:q, :] = cep[:q, :]
    log_env = np.fft.rfft(cep_lift, axis=0).real
    return np.exp(log_env)


def timbre_transfer_envelope_mono(
    source: np.ndarray,
    excitation: np.ndarray,
    sr: int,
    n_fft: int,
    hop: int,
    lifter_quefrency_bins: int,
    mix_env_amount: float,
    use_excitation_phase: bool = True,
) -> np.ndarray:
    """
    Offline timbre coloring (mono): apply source spectral envelope to excitation.
    """
    n = min(source.size, excitation.size)
    source = source[:n]
    excitation = excitation[:n]

    noverlap = n_fft - hop
    _, _, Zs = stft(source, fs=sr, nperseg=n_fft, noverlap=noverlap, window="hann")
    _, _, Ze = stft(excitation, fs=sr, nperseg=n_fft, noverlap=noverlap, window="hann")

    frames = min(Zs.shape[1], Ze.shape[1])
    Zs = Zs[:, :frames]
    Ze = Ze[:, :frames]

    env_s = spectral_envelope_cepstrum(np.abs(Zs), lifter_quefrency_bins)
    env_s = env_s / (np.mean(env_s, axis=0, keepdims=True) + 1e-8)

    env_apply = np.power(env_s, float(np.clip(mix_env_amount, 0.0, 1.0)))
    new_mag = np.abs(Ze) * env_apply

    if use_excitation_phase:
        phase = np.angle(Ze)
    else:
        rng = np.random.default_rng(1)
        phase = rng.uniform(-np.pi, np.pi, size=new_mag.shape).astype(np.float32)

    Znew = new_mag * np.exp(1j * phase)
    _, y = istft(Znew, fs=sr, nperseg=n_fft, noverlap=noverlap, window="hann")

    y = y.astype(np.float32, copy=False)
    if y.size < n:
        y = np.pad(y, (0, n - y.size))
    return y[:n]


def timbre_transfer_envelope(
    source: np.ndarray,
    excitation: np.ndarray,
    sr: int,
    n_fft: int,
    hop: int,
    lifter_quefrency_bins: int,
    mix_env_amount: float,
    stereo_mode: Literal["per_channel", "shared_envelope"],
) -> np.ndarray:
    """
    Multichannel wrapper.

    stereo_mode:
      - shared_envelope: extract envelope from mid (L+R)/2, apply to all channels (recommended for colored noise)
      - per_channel: extract/apply independently per channel
    """
    if source.ndim == 1:
        exc_m = excitation if excitation.ndim == 1 else excitation[:, 0]
        return timbre_transfer_envelope_mono(
            source, exc_m, sr,
            n_fft=n_fft, hop=hop,
            lifter_quefrency_bins=lifter_quefrency_bins,
            mix_env_amount=mix_env_amount,
            use_excitation_phase=True,
        )

    if excitation.ndim == 1:
        excitation = excitation[:, None]
    if source.shape[1] != excitation.shape[1]:
        raise ValueError(f"source channels={source.shape[1]} vs excitation channels={excitation.shape[1]}")

    n = min(source.shape[0], excitation.shape[0])
    source = source[:n, :]
    excitation = excitation[:n, :]
    C = source.shape[1]

    out = np.zeros_like(excitation, dtype=np.float32)

    if stereo_mode == "shared_envelope":
        mid = np.mean(source, axis=1)
        for c in range(C):
            out[:, c] = timbre_transfer_envelope_mono(
                mid, excitation[:, c], sr,
                n_fft=n_fft, hop=hop,
                lifter_quefrency_bins=lifter_quefrency_bins,
                mix_env_amount=mix_env_amount,
                use_excitation_phase=True,
            )
        return out

    for c in range(C):
        out[:, c] = timbre_transfer_envelope_mono(
            source[:, c], excitation[:, c], sr,
            n_fft=n_fft, hop=hop,
            lifter_quefrency_bins=lifter_quefrency_bins,
            mix_env_amount=mix_env_amount,
            use_excitation_phase=True,
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_audio", type=str, required=True, help="Input audio (.wav/.m4a/etc.)")
    ap.add_argument("--out_wav", type=str, default="out_timbre_white.wav", help="Output wav path")

    # Best practice: keep original sample rate unless user explicitly requests a resample.
    ap.add_argument("--sr", type=int, default=0, help="0=keep original sample rate; otherwise resample to this rate during conversion")

    ap.add_argument("--n_fft", type=int, default=2048)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--lifter_q", type=int, default=30)
    ap.add_argument("--mix", type=float, default=1.0)

    ap.add_argument("--stereo_mode", type=str, default="shared_envelope", choices=["per_channel", "shared_envelope"])
    ap.add_argument("--force_channels", type=int, default=0, help="0=keep; 1=mono; 2=stereo")
    args = ap.parse_args()

    in_path = Path(args.in_audio)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    # Determine whether we need a conversion step:
    # - non-wav always needs conversion for soundfile compatibility in a minimal dep setup
    # - wav needs conversion only if user forces channels or requests resampling
    want_sr: Optional[int] = None if args.sr == 0 else int(args.sr)
    want_ch: Optional[int] = None if args.force_channels == 0 else int(args.force_channels)

    is_wav = in_path.suffix.lower() == ".wav"
    needs_convert = (not is_wav) or (want_sr is not None) or (want_ch is not None)

    if needs_convert:
        tmp_wav = in_path.with_suffix(".converted.wav")

        # If user chose "keep original sr" for non-wav, preserve it by probing
        if not is_wav and want_sr is None:
            src_sr, _ = ffprobe_audio_info(in_path)
            want_sr = src_sr

        convert_to_wav_ffmpeg(in_path, tmp_wav, sr=want_sr, channels=want_ch)
        wav_path = tmp_wav
    else:
        wav_path = in_path

    src, sr = load_audio_wav(wav_path)
    channels = 1 if src.ndim == 1 else int(src.shape[1])

    # White noise excitation with same length & channels
    n_samples = src.shape[0] if src.ndim != 1 else src.size
    exc = make_excitation_white(n_samples, channels)

    y = timbre_transfer_envelope(
        source=src,
        excitation=exc,
        sr=sr,
        n_fft=int(args.n_fft),
        hop=int(args.hop),
        lifter_quefrency_bins=int(args.lifter_q),
        mix_env_amount=float(args.mix),
        stereo_mode=str(args.stereo_mode),
    )

    # Normalize
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1e-8:
        y = y / peak * 0.95

    write_wav(Path(args.out_wav), y, sr)
    print(f"Input channels: {channels}")
    print(f"Sample rate used: {sr} Hz")
    print(f"Wrote: {args.out_wav}")


if __name__ == "__main__":
    main()
