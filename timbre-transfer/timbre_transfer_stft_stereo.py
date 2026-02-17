from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Literal, Optional, Tuple, List, Dict

import numpy as np
import soundfile as sf
from scipy.signal import stft, istft


RectHz = Tuple[float, float, float, float]  # (t0_sec, t1_sec, f0_hz, f1_hz)


def ffprobe_audio_info(in_path: Path) -> Tuple[int, int]:
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
    rng = np.random.default_rng(0)
    w = rng.standard_normal((n, channels)).astype(np.float32)
    if channels == 1:
        return w[:, 0]
    return w


def _safe_log(a: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return np.log(np.maximum(a, eps))


def rects_hz_to_mask(rects: List[RectHz], sr: int, n_fft: int, hop: int, F: int, T: int) -> np.ndarray:
    if not rects:
        return np.ones((F, T), dtype=np.float32)

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


def parse_rects(rect_args: Optional[List[str]]) -> List[RectHz]:
    if not rect_args:
        return []
    rects: List[RectHz] = []
    for s in rect_args:
        parts = [p.strip() for p in s.split(",")]
        if len(parts) != 4:
            raise ValueError(f"Bad --rect '{s}'. Use t0,t1,f0,f1 (seconds, Hz).")
        t0, t1, f0, f1 = (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))
        if t1 < t0:
            t0, t1 = t1, t0
        if f1 < f0:
            f0, f1 = f1, f0
        rects.append((t0, t1, f0, f1))
    return rects


def parse_time_map(s: Optional[str]) -> Dict[float, float]:
    if s is None:
        return {}
    try:
        data = json.loads(s)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}\nGot: {s}") from e
    if not isinstance(data, dict):
        raise ValueError("Time map must be a JSON object/dict.")
    out: Dict[float, float] = {}
    for k, v in data.items():
        out[float(k)] = float(v)
    if len(out) == 0:
        return {}
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def interp_time_map(time_map: Dict[float, float], frame_times: np.ndarray, default: float) -> np.ndarray:
    if not time_map:
        return np.full(frame_times.shape, float(default), dtype=np.float32)
    ts = np.array(list(time_map.keys()), dtype=np.float64)
    vs = np.array(list(time_map.values()), dtype=np.float64)
    out = np.interp(frame_times.astype(np.float64), ts, vs, left=vs[0], right=vs[-1])
    return out.astype(np.float32)


def spectral_envelope_cepstrum_timevarying(mag: np.ndarray, lifter_q_per_frame: np.ndarray) -> np.ndarray:
    log_mag = _safe_log(mag)
    cep = np.fft.irfft(log_mag, axis=0)  # (Q,T)
    Q, T = cep.shape

    q = np.rint(lifter_q_per_frame).astype(np.int32)
    q = np.clip(q, 1, Q)

    k_idx = np.arange(Q, dtype=np.int32)[:, None]
    mask = (k_idx < q[None, :]).astype(np.float32)

    cep_lift = cep * mask
    log_env = np.fft.rfft(cep_lift, axis=0).real
    return np.exp(log_env).astype(np.float32, copy=False)


def timbre_transfer_envelope_mono(
    source: np.ndarray,
    excitation: np.ndarray,
    sr: int,
    n_fft: int,
    hop: int,
    lifter_q_map: Dict[float, float],
    mix_map: Dict[float, float],
    lifter_q_default: float,
    mix_default: float,
    rects: List[RectHz],
    use_excitation_phase: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = min(source.size, excitation.size)
    source = source[:n]
    excitation = excitation[:n]

    noverlap = n_fft - hop
    _, _, Zs = stft(source, fs=sr, nperseg=n_fft, noverlap=noverlap, window="hann")
    _, _, Ze = stft(excitation, fs=sr, nperseg=n_fft, noverlap=noverlap, window="hann")

    frames = min(Zs.shape[1], Ze.shape[1])
    Zs = Zs[:, :frames]
    Ze = Ze[:, :frames]

    F, T = Zs.shape
    frame_times = np.arange(T, dtype=np.float32) * (hop / float(sr))

    M = rects_hz_to_mask(rects, sr, n_fft, hop, F, T)

    mag_s = np.abs(Zs) * M
    mag_s = mag_s + (1e-8 * (1.0 - M))

    lifter_q_per_frame = interp_time_map(lifter_q_map, frame_times, default=lifter_q_default)
    mix_per_frame = interp_time_map(mix_map, frame_times, default=mix_default)
    mix_per_frame = np.clip(mix_per_frame, 0.0, 2.0)

    env_s = spectral_envelope_cepstrum_timevarying(mag_s, lifter_q_per_frame)
    env_s = env_s / (np.mean(env_s, axis=0, keepdims=True) + 1e-8)

    env_apply = np.power(env_s, mix_per_frame[None, :])
    new_mag = np.abs(Ze) * env_apply

    if use_excitation_phase:
        phase = np.angle(Ze)
    else:
        rng = np.random.default_rng(1)
        phase = rng.uniform(-np.pi, np.pi, size=new_mag.shape).astype(np.float32)

    Zout = new_mag * np.exp(1j * phase)
    _, y = istft(Zout, fs=sr, nperseg=n_fft, noverlap=noverlap, window="hann")

    y = y.astype(np.float32, copy=False)
    if y.size < n:
        y = np.pad(y, (0, n - y.size))
    return y[:n], Zs, Ze, Zout, M


def timbre_transfer_envelope(
    source: np.ndarray,
    excitation: np.ndarray,
    sr: int,
    n_fft: int,
    hop: int,
    lifter_q_map: Dict[float, float],
    mix_map: Dict[float, float],
    lifter_q_default: float,
    mix_default: float,
    stereo_mode: Literal["per_channel", "shared_envelope"],
    rects: List[RectHz],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if source.ndim == 1:
        exc_m = excitation if excitation.ndim == 1 else excitation[:, 0]
        return timbre_transfer_envelope_mono(
            source, exc_m, sr, n_fft, hop,
            lifter_q_map, mix_map, lifter_q_default, mix_default,
            rects, True
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
        plot_Zs = plot_Ze = plot_Zout = plot_M = None
        for c in range(C):
            y_c, Zs, Ze, Zout, M = timbre_transfer_envelope_mono(
                mid, excitation[:, c], sr, n_fft, hop,
                lifter_q_map, mix_map, lifter_q_default, mix_default,
                rects, True
            )
            out[:, c] = y_c
            if plot_Zs is None:
                plot_Zs, plot_Ze, plot_Zout, plot_M = Zs, Ze, Zout, M
        return out, plot_Zs, plot_Ze, plot_Zout, plot_M

    plot_Zs = plot_Ze = plot_Zout = plot_M = None
    for c in range(C):
        y_c, Zs, Ze, Zout, M = timbre_transfer_envelope_mono(
            source[:, c], excitation[:, c], sr, n_fft, hop,
            lifter_q_map, mix_map, lifter_q_default, mix_default,
            rects, True
        )
        out[:, c] = y_c
        if c == 0:
            plot_Zs, plot_Ze, plot_Zout, plot_M = Zs, Ze, Zout, M
    return out, plot_Zs, plot_Ze, plot_Zout, plot_M


def stft_to_db(Z: np.ndarray) -> np.ndarray:
    mag = np.abs(Z).astype(np.float32, copy=False)
    return 20.0 * np.log10(np.maximum(mag, 1e-8))


def plot_one_spectrogram_png(
    db: np.ndarray,
    sr: int,
    n_fft: int,
    hop: int,
    title: str,
    out_png: Path,
    rects: List[RectHz],
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    F, T = db.shape
    t = np.arange(T) * (hop / sr)
    f = np.arange(F) * (sr / n_fft)

    fig = plt.figure(figsize=(10, 4), dpi=150)
    ax = fig.add_axes([0.08, 0.15, 0.88, 0.75])

    im = ax.imshow(
        db,
        origin="lower",
        aspect="auto",
        extent=[t[0] if T else 0.0, t[-1] if T else 0.0, f[0], f[-1]],
    )
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(im, ax=ax, label="dB")

    for (t0, t1, f0, f1) in rects:
        rect = patches.Rectangle(
            (t0, f0),
            max(0.0, t1 - t0),
            max(0.0, f1 - f0),
            fill=False,
            linewidth=2,
        )
        ax.add_patch(rect)

    fig.savefig(out_png)
    plt.close(fig)


def stitch_pngs_vertical(png_paths: List[Path], out_png: Path) -> None:
    from PIL import Image

    imgs = [Image.open(p).convert("RGBA") for p in png_paths]
    W = max(im.width for im in imgs)
    H = sum(im.height for im in imgs)

    canvas = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    y = 0
    for im in imgs:
        canvas.paste(im, (0, y))
        y += im.height

    canvas.save(out_png)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_audio", type=str, required=True, help="Input audio (.wav/.m4a/etc.)")
    ap.add_argument("--out_wav", type=str, default="output.wav", help="Output wav filename (written to ./output/wav/)")

    ap.add_argument("--sr", type=int, default=0, help="0=keep original sample rate; otherwise resample to this rate during conversion")
    ap.add_argument("--force_channels", type=int, default=0, help="0=keep; 1=mono; 2=stereo")

    ap.add_argument("--n_fft", type=int, default=2048)
    ap.add_argument("--hop", type=int, default=256)

    ap.add_argument("--lifter_q", type=float, default=30.0)
    ap.add_argument("--mix", type=float, default=1.0)

    ap.add_argument("--lifter_q_map", type=str, default=None, help='JSON dict: {"0.0": 20, "5.0": 40} (seconds->lifter_q)')
    ap.add_argument("--mix_map", type=str, default=None, help='JSON dict: {"0.0": 0.2, "5.0": 1.0} (seconds->mix)')

    ap.add_argument("--stereo_mode", type=str, default="shared_envelope", choices=["per_channel", "shared_envelope"])

    ap.add_argument("--rect", action="append", default=None, help="Rectangle selection: t0,t1,f0,f1 (seconds, Hz). Can be repeated.")
    ap.add_argument("--plot", action="store_true", help="Write a single PNG canvas showing input/selection/output spectrograms.")
    ap.add_argument("--plot_png", type=str, default="spectrogram_canvas.png", help="Output PNG filename (written to ./output/png/)")
    args = ap.parse_args()

    in_path = Path(args.in_audio)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    rects = parse_rects(args.rect)
    lifter_map = parse_time_map(args.lifter_q_map)
    mix_map = parse_time_map(args.mix_map)

    want_sr: Optional[int] = None if int(args.sr) == 0 else int(args.sr)
    want_ch: Optional[int] = None if int(args.force_channels) == 0 else int(args.force_channels)

    is_wav = in_path.suffix.lower() == ".wav"
    needs_convert = (not is_wav) or (want_sr is not None) or (want_ch is not None)

    if needs_convert:
        tmp_wav = in_path.with_suffix(".converted.wav")
        if not is_wav and want_sr is None:
            src_sr, _ = ffprobe_audio_info(in_path)
            want_sr = src_sr
        convert_to_wav_ffmpeg(in_path, tmp_wav, sr=want_sr, channels=want_ch)
        wav_path = tmp_wav
    else:
        wav_path = in_path

    src, sr = load_audio_wav(wav_path)
    channels = 1 if src.ndim == 1 else int(src.shape[1])
    n_samples = src.shape[0] if src.ndim != 1 else src.size

    exc = make_excitation_white(n_samples, channels)

    y, Zs_plot, Ze_plot, Zout_plot, _M = timbre_transfer_envelope(
        source=src,
        excitation=exc,
        sr=sr,
        n_fft=int(args.n_fft),
        hop=int(args.hop),
        lifter_q_map=lifter_map,
        mix_map=mix_map,
        lifter_q_default=float(args.lifter_q),
        mix_default=float(args.mix),
        stereo_mode=str(args.stereo_mode),
        rects=rects,
    )

    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1e-8:
        y = y / peak * 0.95

    wav_dir = Path("./output/wav")
    wav_dir.mkdir(parents=True, exist_ok=True)
    out_wav_path = wav_dir / Path(args.out_wav).name
    write_wav(out_wav_path, y, sr)

    if args.plot:
        src_db = stft_to_db(Zs_plot)
        exc_db = stft_to_db(Ze_plot)
        out_db = stft_to_db(Zout_plot)

        png_dir = Path("./output/png")
        png_dir.mkdir(parents=True, exist_ok=True)
        out_png_path = png_dir / Path(args.plot_png).name
        p1 = png_dir / "_spec_input.png"
        p2 = png_dir / "_spec_excitation.png"
        p3 = png_dir / "_spec_output.png"

        plot_one_spectrogram_png(src_db, sr, int(args.n_fft), int(args.hop), "Input spectrogram (selection overlaid)", p1, rects)
        plot_one_spectrogram_png(exc_db, sr, int(args.n_fft), int(args.hop), "Excitation spectrogram (white noise)", p2, rects=[])
        plot_one_spectrogram_png(out_db, sr, int(args.n_fft), int(args.hop), "Output spectrogram (after timbre transfer)", p3, rects=[])

        stitch_pngs_vertical([p1, p2, p3], out_png_path)

        try:
            p1.unlink()
            p2.unlink()
            p3.unlink()
        except OSError:
            pass

    print(f"Input channels: {channels}")
    print(f"Sample rate used: {sr} Hz")
    print(f"Wrote: {out_wav_path}")
    if args.plot:
        print(f"Wrote: {out_png_path}")


if __name__ == "__main__":
    main()
