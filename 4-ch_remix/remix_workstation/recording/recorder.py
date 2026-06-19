# =============================================================================
# Author  : Ho San Ko
# Email   : hko@avalanche.energy
# Project : 4-Channel Music Remix Workstation
# =============================================================================

"""WAV capture module — records the mixed output to disk (Section 7.14)."""

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from constants import SAMPLE_RATE


class Recorder:
    """Captures the mix bus output to a WAV file.

    Attributes (Section 6 class diagram):
        _file           Open soundfile.SoundFile handle (or None).
        _recording      True while capturing audio.
        _punch_in_armed True when waiting for the next loop start to begin.
        _start_time     time.time() timestamp when recording began.
        _lock           Protects _file and _recording.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self._sample_rate = sample_rate
        self._file: Optional[sf.SoundFile] = None
        self._recording: bool = False
        self._punch_in_armed: bool = False
        self._start_time: float = 0.0
        self._lock: threading.Lock = threading.Lock()
        self._last_path: str = ""

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self, path: Optional[str] = None) -> None:
        """Open a WAV file and begin capturing.

        Steps (Section 7.14):
        1. Open soundfile.SoundFile (float32 stereo WAV).
        2. Set _recording = True.
        3. Record _start_time.
        """
        if path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = str(Path("recordings") / f"remix_{ts}.wav")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._file = sf.SoundFile(
                path, mode="w",
                samplerate=self._sample_rate,
                channels=2,
                format="WAV",
                subtype="FLOAT",
            )
            self._recording = True
            self._start_time = time.time()
            self._last_path = path

    def stop(self) -> str:
        """Stop recording, close the file, and return the output path."""
        with self._lock:
            self._recording = False
            if self._file is not None:
                self._file.close()
                self._file = None
        return self._last_path

    def arm_punch_in(self) -> None:
        """Arm punch-in — recording will start at the next loop-start event."""
        self._punch_in_armed = True

    def on_loop_start(self) -> None:
        """Called by the loop system at each loop boundary.

        If punch-in is armed, begins recording now for a clean edit point.
        """
        if self._punch_in_armed:
            self._punch_in_armed = False
            self.start()

    def write(self, buffer: np.ndarray) -> None:
        """Write a stereo buffer from the audio callback thread.

        Steps (Section 7.14): guard _recording, then _file.write(buffer).
        """
        with self._lock:
            if self._recording and self._file is not None:
                self._file.write(buffer)

    def is_recording(self) -> bool:
        with self._lock:
            return self._recording

    def elapsed(self) -> float:
        """Return seconds elapsed since recording started."""
        if not self._recording:
            return 0.0
        return time.time() - self._start_time
