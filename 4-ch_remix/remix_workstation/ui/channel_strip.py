# =============================================================================
# Author  : Ho San Ko
# Email   : hko@avalanche.energy
# Project : 4-Channel Music Remix Workstation
# =============================================================================

"""Per-channel UI strip widget (Section 7.15 / Section 6 class diagram).

Signal contract:
  All user interactions emit signals; the strip holds NO engine references.
  MainWindow.connect_signals() maps every signal to the appropriate engine call.

Hot-cue interaction:
  Left-click  → recall (jump to stored position)
  Right-click → set    (store current playhead)
"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog,
)

from constants import Param, PARAM_RANGES, CHANNEL_COLORS
from ui.knob_widget import KnobWidget
from ui.waveform_widget import WaveformWidget
from ui.loop_shaper_widget import LoopShaperWidget


class ChannelStrip(QWidget):
    """Composite widget containing all per-channel controls.

    Attributes (Section 6 class diagram):
        _waveform        WaveformWidget.
        _knobs           {Param: KnobWidget} — all 16 params.
        _loop_shaper_w   LoopShaperWidget.
        _loop_btns       [in, out, loop, esc] QPushButtons.
        _hot_cue_btns    4 QPushButtons (recall L / set R).
        _load_btn        File-open trigger.
        _bpm_label       Detected/manual BPM display.
    """

    # ── Track ──────────────────────────────────────────────────────────────────
    trackLoadRequested = pyqtSignal(int, str)          # channel_idx, path

    # ── DSP ───────────────────────────────────────────────────────────────────
    paramChanged = pyqtSignal(int, object, float)       # channel_idx, Param, value

    # ── Transport ─────────────────────────────────────────────────────────────
    playSignal  = pyqtSignal(int)                       # channel_idx
    pauseSignal = pyqtSignal(int)
    stopSignal  = pyqtSignal(int)

    # ── Loop ──────────────────────────────────────────────────────────────────
    loopInSet   = pyqtSignal(int)                       # channel_idx
    loopOutSet  = pyqtSignal(int)
    loopToggled = pyqtSignal(int)
    loopEscaped = pyqtSignal(int)

    # ── Hot cues ──────────────────────────────────────────────────────────────
    hotCueSet     = pyqtSignal(int, int)               # channel_idx, cue_idx
    hotCueRecalled = pyqtSignal(int, int)

    # ── Loop Shaper ───────────────────────────────────────────────────────────
    shaperRecordRequested  = pyqtSignal(int)
    shaperOverdubRequested = pyqtSignal(int)
    shaperClearLane        = pyqtSignal(int, object)   # channel_idx, Param
    shaperClearAll         = pyqtSignal(int)
    shaperModeToggle       = pyqtSignal(int)

    def __init__(self, channel_idx: int, parent=None) -> None:
        super().__init__(parent)
        self._channel_idx = channel_idx
        self._color = CHANNEL_COLORS[channel_idx]
        self._setup_ui()
        self._apply_border()

    # ── Public update API (called from MainWindow.update_ui) ─────────────────

    def set_playhead(self, sample: int) -> None:
        self._waveform.set_playhead(sample)

    def set_loop_region(self, loop_in, loop_out) -> None:
        self._waveform.set_loop_region(loop_in, loop_out)

    def set_bpm_text(self, text: str) -> None:
        self._bpm_label.setText(text)

    def set_loading(self, loading: bool) -> None:
        self._load_btn.setEnabled(not loading)
        if loading:
            self._bpm_label.setText("Loading…")

    def flash_knob(self, param: Param) -> None:
        """Called from MIDI input to briefly highlight the affected knob."""
        knob = self._knobs.get(param)
        if knob:
            knob.flash_midi()

    # ── Private: load dialog ──────────────────────────────────────────────────

    def _load_track(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, f"Load Track — Channel {self._channel_idx + 1}", "",
            "Audio Files (*.mp3 *.wav *.flac *.aiff *.aif)"
        )
        if path:
            self.trackLoadRequested.emit(self._channel_idx, path)

    # ── UI setup ──────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(3)

        # ── Header: load + BPM ────────────────────────────────────────────────
        hdr = QHBoxLayout()
        self._load_btn = QPushButton("LOAD")
        self._load_btn.clicked.connect(self._load_track)
        self._bpm_label = QLabel("-- BPM")
        self._bpm_label.setStyleSheet("color: #AAAAAA; font-size: 10px;")
        hdr.addWidget(self._load_btn)
        hdr.addWidget(self._bpm_label)
        hdr.addStretch()
        layout.addLayout(hdr)

        # ── Waveform ──────────────────────────────────────────────────────────
        self._waveform = WaveformWidget()
        layout.addWidget(self._waveform)

        # ── Transport ─────────────────────────────────────────────────────────
        transport = QHBoxLayout()
        for label, sig in [
            ("▶",  lambda: self.playSignal.emit(self._channel_idx)),
            ("⏸",  lambda: self.pauseSignal.emit(self._channel_idx)),
            ("⏹",  lambda: self.stopSignal.emit(self._channel_idx)),
        ]:
            btn = QPushButton(label)
            btn.setFixedWidth(32)
            btn.clicked.connect(sig)
            transport.addWidget(btn)
        transport.addStretch()
        layout.addLayout(transport)

        # ── Knobs: grouped by effect section ──────────────────────────────────
        self._knobs: dict[Param, KnobWidget] = {}
        sections = [
            ("EQ",     [Param.EQ_HIGH, Param.EQ_MID, Param.EQ_LOW]),
            ("Filter", [Param.FILTER_CUTOFF, Param.FILTER_RESONANCE, Param.FILTER_TYPE]),
            ("Reverb", [Param.REVERB_SIZE, Param.REVERB_DAMP, Param.REVERB_MIX]),
            ("Echo",   [Param.ECHO_TIME, Param.ECHO_FEEDBACK, Param.ECHO_MIX]),
            ("Pitch",  [Param.PITCH_SEMITONE, Param.PITCH_CENTS]),
            ("Out",    [Param.VOLUME, Param.PAN]),
        ]
        for section_name, params in sections:
            row = QHBoxLayout()
            lbl = QLabel(section_name)
            lbl.setStyleSheet("color: #666666; font-size: 9px;")
            lbl.setFixedWidth(36)
            row.addWidget(lbl)
            for param in params:
                r = PARAM_RANGES[param]
                knob = KnobWidget(
                    label=param.name.replace("_", "\n"),
                    min_val=r.min_val, max_val=r.max_val, default=r.default,
                )
                knob.valueChanged.connect(
                    lambda v, p=param: self.paramChanged.emit(self._channel_idx, p, v)
                )
                self._knobs[param] = knob
                row.addWidget(knob)
            row.addStretch()
            layout.addLayout(row)

        # ── Loop controls ─────────────────────────────────────────────────────
        loop_row = QHBoxLayout()
        loop_defs = [
            ("IN",   lambda: self.loopInSet.emit(self._channel_idx)),
            ("OUT",  lambda: self.loopOutSet.emit(self._channel_idx)),
            ("LOOP", lambda: self.loopToggled.emit(self._channel_idx)),
            ("ESC",  lambda: self.loopEscaped.emit(self._channel_idx)),
        ]
        self._loop_btns: list[QPushButton] = []
        for label, slot in loop_defs:
            btn = QPushButton(label)
            btn.clicked.connect(slot)
            self._loop_btns.append(btn)
            loop_row.addWidget(btn)
        layout.addLayout(loop_row)

        # ── Hot cues: left-click = recall, right-click = set ─────────────────
        cue_row = QHBoxLayout()
        self._hot_cue_btns: list[QPushButton] = []
        for idx in range(4):
            btn = QPushButton(f"CUE {idx + 1}")
            btn.setToolTip("Left-click: recall  |  Right-click: set")
            btn.clicked.connect(
                lambda checked, i=idx: self.hotCueRecalled.emit(self._channel_idx, i)
            )
            btn.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            btn.customContextMenuRequested.connect(
                lambda _pos, i=idx: self.hotCueSet.emit(self._channel_idx, i)
            )
            self._hot_cue_btns.append(btn)
            cue_row.addWidget(btn)
        layout.addLayout(cue_row)

        # ── Loop Shaper widget ────────────────────────────────────────────────
        self._loop_shaper_w = LoopShaperWidget(channel_color=self._color)
        self._loop_shaper_w.recordRequested.connect(
            lambda: self.shaperRecordRequested.emit(self._channel_idx)
        )
        self._loop_shaper_w.overdubRequested.connect(
            lambda: self.shaperOverdubRequested.emit(self._channel_idx)
        )
        self._loop_shaper_w.clearLaneRequested.connect(
            lambda p: self.shaperClearLane.emit(self._channel_idx, p)
        )
        self._loop_shaper_w.clearAllRequested.connect(
            lambda: self.shaperClearAll.emit(self._channel_idx)
        )
        self._loop_shaper_w.modeToggleRequested.connect(
            lambda: self.shaperModeToggle.emit(self._channel_idx)
        )
        layout.addWidget(self._loop_shaper_w)

    def _apply_border(self) -> None:
        self.setStyleSheet(
            f"ChannelStrip {{"
            f"  border: 2px solid {self._color};"
            f"  border-radius: 4px;"
            f"  background: #1A1A1A;"
            f"}}"
        )
