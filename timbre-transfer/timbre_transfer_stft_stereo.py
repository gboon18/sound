from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import stft, istft


def convert_to_wav_ffmpeg(
    in_path: Path,
    out_path: Path,
    sr: int = 48000,
    channels: Optional[int] = None,
) -> None:
    """
    Convert common audio formats (.m4a, .mp3, etc.) to PCM WAV via ffmpeg.
    Requires ffmpeg installed and available on PATH.

    channels:
      None -> keep channel count as-is (stereo stays stereo)
      1    -> force mono
      2    -> force stereo
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", str(in_path)]
    if channels is not None:
        cmd += ["-ac", str(int(channels))]
    cmd += ["-ar", str(int(sr)), "-c:a", "pcm_s24le", str(out_path)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{p.stderr}")


def load_audio(path: Path) -> Tuple[np.ndarray, int]:
    """
    Returns:
      x: shape (n,) for mono, (n, c) for multichannel
      sr
    """
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim == 1:
        return x, sr
    # soundfile returns (n, c) already when always_2d=False, but ensure dtype
    return x.astype(np.float32, copy=False), sr


def write_wav(path: Path, x: np.ndarray, sr: int) -> None:
    x = np.asarray(x, dtype=np.float32)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 0.999:
        x = x / peak * 0.999
    sf.write(path, x, sr, subtype="PCM_24")


def make_excitation(kind: Literal["white", "from_audio"], n: int, channels: int, source_audio: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Returns excitation with shape:
      mono: (n,)
      stereo/multich: (n, channels)
    """
    if kind == "from_audio":
        if source_audio is None:
            raise ValueError("source_audio must be provided for kind='from_audio'")
        x = np.asarray(source_audio, dtype=np.float32)
        if x.ndim == 1:
            x = x[:, None]
        if x.shape[1] != channels:
            raise ValueError(f"source_audio channels={x.shape[1]} does not match requested channels={channels}")
        if x.shape[0] < n:
            reps = int(np.ceil(n / x.shape[0]))
            x = np.tile(x, (reps, 1))
        return x[:n, :].copy()

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
    n_fft: int = 2048,
    hop: int = 256,
    lifter_quefrency_bins: int = 30,
    mix_env_amount: float = 1.0,
    use_excitation_phase: bool = True,
) -> np.ndarray:
    """
    Offline "timbre coloring" (mono).
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
    n_fft: int = 2048,
    hop: int = 256,
    lifter_quefrency_bins: int = 30,
    mix_env_amount: float = 1.0,
    stereo_mode: Literal["per_channel", "shared_envelope"] = "per_channel",
) -> np.ndarray:
    """
    Multichannel wrapper.

    stereo_mode:
      - per_channel: extract envelope independently per channel, apply to same channel.
      - shared_envelope: extract envelope from mid (L+R)/2, apply same envelope to all channels.
        (This tends to preserve a coherent "timbre" across stereo noise.)
    """
    if source.ndim == 1:
        return timbre_transfer_envelope_mono(
            source, excitation if excitation.ndim == 1 else excitation[:, 0], sr,
            n_fft=n_fft, hop=hop, lifter_quefrency_bins=lifter_quefrency_bins, mix_env_amount=mix_env_amount
        )

    if excitation.ndim == 1:
        excitation = excitation[:, None]
    if source.shape[1] != excitation.shape[1]:
        raise ValueError(f"source channels={source.shape[1]} vs excitation channels={excitation.shape[1]}")

    n = min(source.shape[0], excitation.shape[0])
    source = source[:n, :]
    excitation = excitation[:n, :]

    C = source.shape[1]

    if stereo_mode == "shared_envelope":
        mid = np.mean(source, axis=1)
        # apply same envelope to each channel
        out = np.zeros_like(excitation, dtype=np.float32)
        for c in range(C):
            out[:, c] = timbre_transfer_envelope_mono(
                mid, excitation[:, c], sr,
                n_fft=n_fft, hop=hop,
                lifter_quefrency_bins=lifter_quefrency_bins,
                mix_env_amount=mix_env_amount,
                use_excitation_phase=True,
            )
        return out

    # per-channel
    out = np.zeros_like(excitation, dtype=np.float32)
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
    ap.add_argument("--sr", type=int, default=48000, help="Conversion sample rate for non-wav input")
    ap.add_argument("--n_fft", type=int, default=2048)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--lifter_q", type=int, default=30)
    ap.add_argument("--mix", type=float, default=1.0)
    ap.add_argument("--stereo_mode", type=str, default="shared_envelope", choices=["per_channel", "shared_envelope"])
    ap.add_argument("--force_channels", type=int, default=0, help="0=keep, 1=mono, 2=stereo")
    args = ap.parse_args()

    in_path = Path(args.in_audio)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    # If not wav, convert first (keep channels by default)
    if in_path.suffix.lower() != ".wav":
        tmp_wav = in_path.with_suffix(".converted.wav")
        ch = None if args.force_channels == 0 else int(args.force_channels)
        convert_to_wav_ffmpeg(in_path, tmp_wav, sr=args.sr, channels=ch)
        wav_path = tmp_wav
    else:
        wav_path = in_path

    src, sr = load_audio(wav_path)

    channels = 1 if src.ndim == 1 else int(src.shape[1])
    exc = make_excitation("white", n=src.shape[0] if src.ndim != 1 else src.size, channels=channels)

    y = timbre_transfer_envelope(
        source=src,
        excitation=exc,
        sr=sr,
        n_fft=args.n_fft,
        hop=args.hop,
        lifter_quefrency_bins=args.lifter_q,
        mix_env_amount=args.mix,
        stereo_mode=args.stereo_mode,
    )

    # normalize
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1e-8:
        y = y / peak * 0.95

    write_wav(Path(args.out_wav), y, sr)
    print(f"Input channels: {channels}")
    print(f"Wrote: {args.out_wav}")


if __name__ == "__main__":
    main()
