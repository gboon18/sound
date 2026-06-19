# =============================================================================
# Author  : Ho San Ko
# Email   : hko@avalanche.energy
# Project : 4-Channel Music Remix Workstation
# =============================================================================

"""Loop Shaper automation lane display and controls (Section 7.15 / Section 6)."""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QLabel,
)

from constants import Param


class LoopShaperWidget(QWidget):
    """Collapsible panel showing 16 automation lanes with a sweep playhead.

    Attributes (Section 6 class diagram):
        _lanes          Copy of automation data from LoopShaper, keyed by Param.
        _active_lanes   Set of Params with non-empty automation.
        _playhead_pos   Current normalized position (0.0–1.0).
        _collapsed      True when the panel is folded.
    """

    recordRequested = pyqtSignal()
    overdubRequested = pyqtSignal()
    clearLaneRequested = pyqtSignal(object)   # Param
    clearAllRequested = pyqtSignal()
    modeToggleRequested = pyqtSignal()

    _LANE_HEIGHT: int = 12
    _LANE_GAP: int = 2

    def __init__(self, channel_color: str = "#3498DB", parent=None) -> None:
        super().__init__(parent)
        self._channel_color = channel_color
        self._lanes: dict[Param, list[tuple[float, float]]] = {p: [] for p in Param}
        self._active_lanes: set[Param] = set()
        self._playhead_pos: float = 0.0
        self._collapsed: bool = False

        self._setup_ui()

    # ── Public setters ────────────────────────────────────────────────────────

    def update_lanes(self, data: dict[Param, list[tuple[float, float]]]) -> None:
        self._lanes = data
        self._active_lanes = {p for p, lane in data.items() if lane}
        self._canvas.update()

    def set_playhead(self, pos: float) -> None:
        self._playhead_pos = max(0.0, min(1.0, pos))
        self._canvas.update()

    def toggle_collapse(self) -> None:
        self._collapsed = not self._collapsed
        self._canvas.setVisible(not self._collapsed)
        self._collapse_btn.setText("▶" if self._collapsed else "▼")

    # ── UI setup ──────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        # Header bar
        header = QHBoxLayout()
        self._collapse_btn = QPushButton("▼")
        self._collapse_btn.setFixedWidth(20)
        self._collapse_btn.clicked.connect(self.toggle_collapse)

        self._record_btn = QPushButton("REC")
        self._record_btn.setCheckable(True)
        self._record_btn.clicked.connect(self.recordRequested.emit)

        self._overdub_btn = QPushButton("OVR")
        self._overdub_btn.setCheckable(True)
        self._overdub_btn.clicked.connect(self.overdubRequested.emit)

        self._clear_lane_combo = QComboBox()
        for p in Param:
            self._clear_lane_combo.addItem(p.name, p)

        clear_lane_btn = QPushButton("CLR Lane")
        clear_lane_btn.clicked.connect(
            lambda: self.clearLaneRequested.emit(
                self._clear_lane_combo.currentData()
            )
        )

        self._clear_all_btn = QPushButton("CLR All")
        self._clear_all_btn.clicked.connect(self.clearAllRequested.emit)

        self._mode_toggle = QPushButton("ABS")
        self._mode_toggle.setCheckable(True)
        self._mode_toggle.clicked.connect(self._on_mode_toggle)

        for w in [
            self._collapse_btn, self._record_btn, self._overdub_btn,
            self._clear_lane_combo, clear_lane_btn, self._clear_all_btn,
            self._mode_toggle,
        ]:
            header.addWidget(w)
        header.addStretch()
        layout.addLayout(header)

        # Canvas
        self._canvas = _LaneCanvas(self)
        layout.addWidget(self._canvas)

    def _on_mode_toggle(self) -> None:
        self._mode_toggle.setText("ADD" if self._mode_toggle.isChecked() else "ABS")
        self.modeToggleRequested.emit()


class _LaneCanvas(QWidget):
    """Draws the 16 automation lanes inside LoopShaperWidget."""

    def __init__(self, parent: LoopShaperWidget) -> None:
        super().__init__(parent)
        self._owner = parent
        lane_count = len(list(Param))
        total_h = lane_count * (LoopShaperWidget._LANE_HEIGHT + LoopShaperWidget._LANE_GAP)
        self.setMinimumHeight(total_h)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        w, h = self.width(), self.height()
        owner = self._owner
        lane_h = LoopShaperWidget._LANE_HEIGHT
        gap = LoopShaperWidget._LANE_GAP

        painter.fillRect(0, 0, w, h, QColor("#111111"))

        for i, param in enumerate(Param):
            y = i * (lane_h + gap)
            lane = owner._lanes.get(param, [])
            active = param in owner._active_lanes
            bg = QColor("#1E2A1E") if active else QColor("#1A1A1A")
            painter.fillRect(0, y, w, lane_h, bg)

            if active and lane:
                painter.setPen(QPen(QColor(owner._channel_color), 1))
                pts = sorted(lane, key=lambda pt: pt[0])
                for j in range(len(pts) - 1):
                    x0 = int(pts[j][0] * w)
                    x1 = int(pts[j + 1][0] * w)
                    y0 = y + lane_h - int(pts[j][1] * lane_h)
                    y1 = y + lane_h - int(pts[j + 1][1] * lane_h)
                    painter.drawLine(x0, y0, x1, y1)
            else:
                painter.setPen(QPen(QColor("#333333"), 1))
                mid_y = y + lane_h // 2
                painter.drawLine(0, mid_y, w, mid_y)

        # Playhead
        px = int(owner._playhead_pos * w)
        painter.setPen(QPen(QColor("#FFFFFF"), 1))
        painter.drawLine(px, 0, px, h)
