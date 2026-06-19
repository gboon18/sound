# =============================================================================
# Author  : Ho San Ko
# Email   : hko@avalanche.energy
# Project : 4-Channel Music Remix Workstation
# =============================================================================

"""Custom rotary knob widget (Section 7.15 / Section 6 class diagram)."""

import math

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen, QFont
from PyQt6.QtWidgets import QWidget


class KnobWidget(QWidget):
    """Circular knob rendered entirely in paintEvent.

    Attributes (Section 6 class diagram):
        _value          Current value in [_min, _max].
        _min, _max      Value range.
        _label          Text label displayed below the arc.
        _midi_highlight Flash flag: True for one UI tick after MIDI input.
    """

    valueChanged = pyqtSignal(float)

    _ARC_START_DEG: float = 225.0   # 7 o'clock position
    _ARC_SPAN_DEG: float = 270.0    # full sweep

    def __init__(
        self,
        label: str = "",
        min_val: float = 0.0,
        max_val: float = 1.0,
        default: float = 0.0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._label = label
        self._min = min_val
        self._max = max_val
        self._value = default
        self._midi_highlight: bool = False
        self._drag_start_y: int | None = None
        self._drag_start_value: float = default

        self.setMinimumSize(60, 70)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.SizeVerCursor)

    # ── Public API ────────────────────────────────────────────────────────────

    def setValue(self, value: float) -> None:
        value = max(self._min, min(self._max, value))
        if value != self._value:
            self._value = value
            self.update()
            self.valueChanged.emit(self._value)

    def value(self) -> float:
        return self._value

    def flash_midi(self) -> None:
        """Briefly highlight the knob outline to indicate MIDI input."""
        self._midi_highlight = True
        self.update()

    def clear_midi_highlight(self) -> None:
        self._midi_highlight = False
        self.update()

    # ── Qt events ─────────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        knob_size = min(w, h - 16)  # leave room for label
        x = (w - knob_size) // 2
        y = 2

        # Background arc
        painter.setPen(QPen(QColor("#333333"), 3))
        painter.drawArc(x, y, knob_size, knob_size, int(self._ARC_START_DEG * 16), -int(self._ARC_SPAN_DEG * 16))

        # Value arc
        norm = (self._value - self._min) / (self._max - self._min) if self._max != self._min else 0.0
        span = int(norm * self._ARC_SPAN_DEG * 16)
        highlight_color = "#FFD700" if self._midi_highlight else "#00BFFF"
        painter.setPen(QPen(QColor(highlight_color), 3))
        painter.drawArc(x, y, knob_size, knob_size, int(self._ARC_START_DEG * 16), -span)

        # Pointer line
        cx = x + knob_size / 2
        cy = y + knob_size / 2
        angle_deg = self._ARC_START_DEG - norm * self._ARC_SPAN_DEG
        angle_rad = math.radians(angle_deg)
        r = knob_size / 2 - 4
        px = cx + r * math.cos(angle_rad)
        py = cy - r * math.sin(angle_rad)
        painter.setPen(QPen(QColor("white"), 2))
        painter.drawLine(int(cx), int(cy), int(px), int(py))

        # Label
        painter.setPen(QColor("#AAAAAA"))
        painter.setFont(QFont("Arial", 7))
        painter.drawText(0, h - 14, w, 14, Qt.AlignmentFlag.AlignHCenter, self._label)

        # Value text
        val_str = f"{self._value:.2f}"
        painter.setPen(QColor("white"))
        painter.setFont(QFont("Arial", 7))
        painter.drawText(0, int(cy) - 6, w, 14, Qt.AlignmentFlag.AlignHCenter, val_str)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start_y = event.pos().y()
            self._drag_start_value = self._value

    def mouseMoveEvent(self, event) -> None:
        if self._drag_start_y is not None:
            delta_y = self._drag_start_y - event.pos().y()
            range_size = self._max - self._min
            new_val = self._drag_start_value + (delta_y / 100.0) * range_size
            self.setValue(new_val)

    def mouseReleaseEvent(self, event) -> None:
        self._drag_start_y = None

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y() / 120.0
        step = (self._max - self._min) / 100.0
        self.setValue(self._value + delta * step)
