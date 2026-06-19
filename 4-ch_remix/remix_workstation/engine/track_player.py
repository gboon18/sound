# =============================================================================
# Author  : Ho San Ko
# Email   : hko@avalanche.energy
# Project : 4-Channel Music Remix Workstation
# =============================================================================

"""Per-channel audio player — loads, stretches, and streams audio chunks.

Threading model (Section 9):
  load_file / stretch  run on a QThreadPool worker thread.
  advance()            called from the real-time sounddevice audio callback.
  get/set_playhead     may be called from the UI thread (read-only for display).

Heavy computation (pyrubberband) runs OUTSIDE _lock; only the final pointer
swap is atomic, ensuring the audio callback is never blocked for more than a
handful of microseconds.
"""

import threading
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from constants import DEFAULT_MASTER_BPM, SAMPLE_RATE, SUPPORTED_FORMATS

# 2 ms crossfade at 44100 Hz (Section 7.3 / Phase 5 Task 5.3)
_XFADE_SAMPLES: int = 88


class TrackPlayer:
    """Holds raw + stretched audio for one channel and maintains the playhead.

    Attributes (Section 6 class diagram):
        _audio_data      Raw PCM float32 stereo.  Shape (N, 2).
        _stretched_data  Time-stretched copy.      Shape (M, 2).
        _playhead        Read position inside _stretched_data.
        _track_bpm       Detected or manually-set BPM (0 = unknown).
        _playing         Playback active flag.
        _lock            Protects _stretched_data, _playhead, _playing.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, buffer_size: int = 512) -> None:
        self._sample_rate = sample_rate
        self._buffer_size = buffer_size

        self._audio_data: Optional[np.ndarray] = None
        self._stretched_data: Optional[np.ndarray] = None
        self._playhead: int = 0
        self._track_bpm: float = 0.0
        self._playing: bool = False
        self._lock: threading.Lock = threading.Lock()
        self._stretch_ratio: float = 1.0

        # Loop wrap detection counter (audio-callback thread only; no lock)
        self._wrap_count: int = 0

        # Optional LoopManager reference — injected after construction
        self._loop_manager = None  # type: Optional[object]  # LoopManager

    # ── LoopManager injection (Phase 5) ──────────────────────────────────────

    def set_loop_manager(self, loop_manager) -> None:
        """Inject the LoopManager for this channel (called once from main.py)."""
        self._loop_manager = loop_manager

    def get_wrap_count(self) -> int:
        """Monotonic counter incremented each time a loop boundary is crossed."""
        return self._wrap_count

    # ── File loading (Phase 2, Section 7.3) ──────────────────────────────────

    def load_file(self, path: str, master_bpm: float = DEFAULT_MASTER_BPM) -> None:
        """Load a file, detect BPM, and perform the initial time-stretch.

        Must be called from a worker thread.  Heavy work happens OUTSIDE _lock;
        only the final pointer swap is inside.

        Raises:
            ValueError:   Unsupported format.
            RuntimeError: soundfile read failure.
        """
        import soundfile as sf
        import librosa
        from engine.bpm_detector import BpmDetector

        path_obj = Path(path)
        if path_obj.suffix.lower() not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {path_obj.suffix!r}")

        # Read & normalise to (N, 2) float32
        data, sr = sf.read(str(path_obj), dtype="float32", always_2d=True)
        if data.shape[1] == 1:
            data = np.column_stack([data[:, 0], data[:, 0]])
        data = data[:, :2].astype(np.float32)

        # Resample if needed
        if sr != self._sample_rate:
            left = librosa.resample(
                data[:, 0], orig_sr=sr, target_sr=self._sample_rate
            )
            right = librosa.resample(
                data[:, 1], orig_sr=sr, target_sr=self._sample_rate
            )
            data = np.column_stack([left, right]).astype(np.float32)

        # Detect BPM
        detected_bpm: Optional[float] = BpmDetector().detect(data, self._sample_rate)

        # Stretch (outside lock — may take seconds)
        if detected_bpm is not None and master_bpm > 0:
            ratio = master_bpm / detected_bpm
            stretched = _pyrubberband_stretch(data, self._sample_rate, ratio)
        else:
            ratio = 1.0
            stretched = data.copy()

        # Atomic swap
        with self._lock:
            self._audio_data = data
            self._stretched_data = stretched
            self._stretch_ratio = ratio
            self._track_bpm = detected_bpm if detected_bpm is not None else 0.0
            self._playhead = 0

    def stretch(self, ratio: float) -> None:
        """Re-stretch raw audio to *ratio*.  Called by SyncManager on BPM change.

        Snapshots _audio_data OUTSIDE lock, runs pyrubberband, atomically swaps
        the result.  The identity guard (``is not raw``) prevents a stale stretch
        from landing after a concurrent load_file().
        """
        with self._lock:
            if self._audio_data is None:
                return
            raw = self._audio_data
            old_ratio = self._stretch_ratio
            old_ph = self._playhead

        stretched = _pyrubberband_stretch(raw, self._sample_rate, ratio)

        # Proportionally rescale playhead
        new_ph = int(old_ph * ratio / old_ratio) if old_ratio > 0 else 0
        new_ph = max(0, min(new_ph, len(stretched) - 1))

        with self._lock:
            if self._audio_data is not raw:
                return  # race guard: new file loaded while we were stretching
            self._stretched_data = stretched
            self._stretch_ratio = ratio
            self._playhead = new_ph

    # ── Playback (Phase 2/5, Section 7.3) ─────────────────────────────────────

    def advance(self, frames: int) -> np.ndarray:
        """Return the next *frames* stereo samples and advance the playhead.

        Phase 5 behaviour:
        - If a LoopManager is attached and its loop is active, wraps at loop_out
          with a 2 ms (88-sample) crossfade to eliminate the click (Task 5.3).
        - If no loop is active, wraps seamlessly at track end (Phase 2 behaviour).

        Returns:
            (frames, 2) float32 array.  Silence when paused or not loaded.
        """
        with self._lock:
            if not self._playing or self._stretched_data is None:
                return np.zeros((frames, 2), dtype=np.float32)

            total = len(self._stretched_data)
            if total == 0:
                return np.zeros((frames, 2), dtype=np.float32)

            # Determine whether we're in loop mode
            lm = self._loop_manager
            loop_active = False
            loop_in = 0
            loop_out = total
            if lm is not None and lm.is_loop_active():
                bounds = lm.get_loop_bounds()
                if bounds is not None:
                    li, lo = bounds
                    if lo > li and lo <= total and (lo - li) >= _XFADE_SAMPLES * 2:
                        loop_active = True
                        loop_in = li
                        loop_out = lo

            if loop_active:
                return self._read_looped(frames, loop_in, loop_out)
            else:
                return self._read_linear(frames, total)

    def _read_linear(self, frames: int, total: int) -> np.ndarray:
        """Read *frames* samples with seamless track-end wrap (Phase 2 behaviour).
        Called under _lock.
        """
        sd = self._stretched_data
        ph = self._playhead
        end = ph + frames

        if end <= total:
            chunk = sd[ph:end].copy()
            self._playhead = end % total
        elif frames <= total:
            tail = sd[ph:]
            remaining = frames - len(tail)
            head = sd[:remaining]
            chunk = np.vstack([tail, head])
            self._playhead = remaining
        else:
            # frames > total: tile the track
            chunk = np.zeros((frames, 2), dtype=np.float32)
            pos = 0
            while pos < frames:
                take = min(total, frames - pos)
                chunk[pos:pos + take] = sd[:take]
                pos += take
            self._playhead = frames % total

        return chunk.astype(np.float32)

    def _read_looped(self, frames: int, loop_in: int, loop_out: int) -> np.ndarray:
        """Read *frames* samples within [loop_in, loop_out) with crossfade wrap.

        Crossfade algorithm (Task 5.3 — 2 ms crossfade):
        - Define xfade_start = loop_out - fade_len.
        - In the normal zone [loop_in+fade_len, xfade_start): straight read.
        - In the xfade zone [xfade_start, loop_out): output is a linear blend of
              tail = stretched_data[ph]          (fading OUT, weight 1→0)
          and head = stretched_data[loop_in + xfade_offset]  (fading IN,  weight 0→1)
        - After the blend, playhead jumps to loop_in + fade_len (skipping the
          fade-in region that was already blended in), preserving loop length.

        Called under _lock.
        """
        sd = self._stretched_data
        loop_len = loop_out - loop_in
        fade_len = min(_XFADE_SAMPLES, loop_len // 4)  # safe for short loops
        xfade_start = loop_out - fade_len

        chunk = np.zeros((frames, 2), dtype=np.float32)
        out_pos = 0  # write cursor in chunk
        ph = self._playhead

        # Clamp playhead into the loop region
        if ph < loop_in or ph >= loop_out:
            ph = loop_in
            self._wrap_count += 1

        while out_pos < frames:
            needed = frames - out_pos

            if ph < xfade_start:
                # ── Normal zone ───────────────────────────────────────────
                available = xfade_start - ph
                take = min(available, needed)
                chunk[out_pos:out_pos + take] = sd[ph:ph + take]
                ph += take
                out_pos += take

            elif ph < loop_out:
                # ── Crossfade zone ────────────────────────────────────────
                xfade_offset = ph - xfade_start   # 0 … fade_len-1
                remaining_xfade = fade_len - xfade_offset
                take = min(remaining_xfade, needed)

                tail_chunk = sd[ph:ph + take]                          # fading out
                head_start = loop_in + xfade_offset
                head_chunk = sd[head_start:head_start + take]          # fading in

                # Linear crossfade weights for this segment
                t0 = xfade_offset / fade_len
                t1 = (xfade_offset + take) / fade_len
                t = np.linspace(t0, t1, take, dtype=np.float32)[:, np.newaxis]
                chunk[out_pos:out_pos + take] = (
                    tail_chunk * (1.0 - t) + head_chunk * t
                )
                ph += take
                out_pos += take

                if ph >= loop_out:
                    # Wrap: skip the fade-in zone already blended above
                    ph = loop_in + fade_len
                    self._wrap_count += 1

            else:
                # Should not reach here — safety reset
                ph = loop_in
                self._wrap_count += 1

        self._playhead = ph
        return chunk

    # ── Playhead ──────────────────────────────────────────────────────────────

    def set_playhead(self, sample: int) -> None:
        with self._lock:
            if self._stretched_data is not None:
                self._playhead = max(0, min(sample, len(self._stretched_data) - 1))

    def get_playhead(self) -> int:
        with self._lock:
            return self._playhead

    def get_stretched_length(self) -> int:
        with self._lock:
            return len(self._stretched_data) if self._stretched_data is not None else 0

    # ── Transport ─────────────────────────────────────────────────────────────

    def play(self) -> None:
        with self._lock:
            if self._stretched_data is not None:
                self._playing = True

    def pause(self) -> None:
        with self._lock:
            self._playing = False

    def stop(self) -> None:
        with self._lock:
            self._playing = False
            self._playhead = 0

    def is_playing(self) -> bool:
        with self._lock:
            return self._playing

    def is_loaded(self) -> bool:
        with self._lock:
            return self._stretched_data is not None

    # ── BPM ───────────────────────────────────────────────────────────────────

    def get_track_bpm(self) -> float:
        return self._track_bpm

    def set_track_bpm(self, bpm: float) -> None:
        self._track_bpm = bpm


# ── Module-level helper ───────────────────────────────────────────────────────

def _pyrubberband_stretch(
    audio: np.ndarray, sample_rate: int, ratio: float
) -> np.ndarray:
    """Time-stretch *audio* (N, 2) by *ratio* using pyrubberband per channel."""
    import pyrubberband

    if abs(ratio - 1.0) < 1e-9:
        return audio.copy()
    left = pyrubberband.time_stretch(audio[:, 0], sample_rate, ratio)
    right = pyrubberband.time_stretch(audio[:, 1], sample_rate, ratio)
    return np.column_stack([left, right]).astype(np.float32)
