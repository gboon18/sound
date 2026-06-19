# =============================================================================
# Author  : Ho San Ko
# Email   : hko@avalanche.energy
# Project : 4-Channel Music Remix Workstation
# =============================================================================

"""Scrolling waveform display with beat grid and loop region overlay."""

from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QColor, QPen
from PyQt6.QtWidgets import QWidget


class WaveformWidget(QWidget):
    """Renders a peak-envelope waveform with scrolling playhead.

    Attributes (Section 6 class diagram):
        _audio_data     Downsampled peak envelope array (or None).
        _playhead       Current playhead position in stretched samples.
        _loop_in        Loop-in sample position (or None).
        _loop_out       Loop-out sample position (or None).
        _beat_grid      List of sample positions for beat grid lines.
    """

    _PEAK_COLS: int = 600  # number of columns in the peak envelope

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._audio_data: Optional[np.ndarray] = None  # raw stretched audio
        self._peak_envelope: Optional[np.ndarray] = None
        self._playhead: int = 0
        self._loop_in: Optional[int] = None
        self._loop_out: Optional[int] = None
        self._beat_grid: list[int] = []
        self._total_samples: int = 0

        self.setMinimumHeight(60)

    # ── Public setters ────────────────────────────────────────────────────────

    def set_audio(self, data: np.ndarray) -> None:
        """Store *data* and compute a downsampled peak envelope for display."""
        self._audio_data = data
        self._total_samples = len(data)
        self._build_envelope(data)
        self.update()

    def set_playhead(self, pos: int) -> None:
        self._playhead = pos
        self.update()

    def set_loop_region(self, loop_in: Optional[int], loop_out: Optional[int]) -> None:
        self._loop_in = loop_in
        self._loop_out = loop_out
        self.update()

    def set_beat_grid(self, beats: list[int]) -> None:
        self._beat_grid = beats
        self.update()

    # ── Qt rendering ──────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        w, h = self.width(), self.height()
        mid = h // 2

        painter.fillRect(0, 0, w, h, QColor("#1A1A1A"))

        if self._peak_envelope is None or self._total_samples == 0:
            return

        # Loop region shading
        if self._loop_in is not None and self._loop_out is not None:
            lx = self._sample_to_x(self._loop_in, w)
            rx = self._sample_to_x(self._loop_out, w)
            painter.fillRect(lx, 0, rx - lx, h, QColor(0, 191, 255, 40))

        # Beat grid
        painter.setPen(QPen(QColor("#333333"), 1))
        for beat_sample in self._beat_grid:
            bx = self._sample_to_x(beat_sample, w)
            painter.drawLine(bx, 0, bx, h)

        # Waveform
        painter.setPen(QPen(QColor("#4488CC"), 1))
        num_cols = len(self._peak_envelope)
        for col in range(min(w, num_cols)):
            env_idx = int(col * num_cols / w)
            amp = float(self._peak_envelope[env_idx]) * mid
            painter.drawLine(col, mid - int(amp), col, mid + int(amp))

        # Playhead
        px = self._sample_to_x(self._playhead, w)
        painter.setPen(QPen(QColor("#FFFFFF"), 2))
        painter.drawLine(px, 0, px, h)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_envelope(self, data: np.ndarray) -> None:
        """Compute a downsampled peak envelope for display."""
        if len(data) == 0:
            self._peak_envelope = None
            return
        mono = data.mean(axis=1) if data.ndim == 2 else data
        n = len(mono)
        chunk = max(1, n // self._PEAK_COLS)
        peaks = []
        for i in range(0, n, chunk):
            segment = mono[i:i + chunk]
            peaks.append(float(np.max(np.abs(segment))))
        self._peak_envelope = np.array(peaks, dtype=np.float32)
        if self._peak_envelope.max() > 0:
            self._peak_envelope /= self._peak_envelope.max()

    def _sample_to_x(self, sample: int, width: int) -> int:
        if self._total_samples == 0:
            return 0
        return int(sample / self._total_samples * width)
