# =============================================================================
# Author  : Ho San Ko
# Email   : hko@avalanche.energy
# Project : 4-Channel Music Remix Workstation
# =============================================================================

"""Per-channel UI strip widget (Section 7.15 / Section 6 class diagram)."""

from PyQt6.QtCore import pyqtSignal
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
        _waveform        WaveformWidget showing the loaded track.
        _knobs           Dict mapping Param → KnobWidget (16 total).
        _loop_shaper_w   LoopShaperWidget for automation lane display.
        _loop_btns       Loop in/out/toggle/escape buttons.
        _hot_cue_btns    4 hot cue buttons.
        _load_btn        QPushButton to open a file dialog.
        _bpm_label       QLabel showing the detected/manual BPM.
    """

    trackLoadRequested = pyqtSignal(int, str)      # channel_idx, path
    paramChanged = pyqtSignal(int, object, float)  # channel_idx, Param, value
    loopInSet = pyqtSignal(int)
    loopOutSet = pyqtSignal(int)
    loopToggled = pyqtSignal(int)
    loopEscaped = pyqtSignal(int)
    hotCueSet = pyqtSignal(int, int)     # channel_idx, cue_idx
    hotCueRecalled = pyqtSignal(int, int)

    def __init__(self, channel_idx: int, parent=None) -> None:
        super().__init__(parent)
        self._channel_idx = channel_idx
        self._color = CHANNEL_COLORS[channel_idx]
        self._setup_ui()
        self._apply_border()

    # ── Public update API ─────────────────────────────────────────────────────

    def load_track(self) -> None:
        """Open a file dialog and emit trackLoadRequested."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Track", "",
            "Audio Files (*.mp3 *.wav *.flac *.aiff *.aif)"
        )
        if path:
            self.trackLoadRequested.emit(self._channel_idx, path)

    def update_waveform(self, audio_data, playhead: int) -> None:
        self._waveform.set_audio(audio_data)
        self._waveform.set_playhead(playhead)

    def on_knob_changed(self, param: Param, value: float) -> None:
        self.paramChanged.emit(self._channel_idx, param, value)

    # ── UI setup ──────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(3)

        # Load button + BPM label
        top_row = QHBoxLayout()
        self._load_btn = QPushButton("LOAD")
        self._load_btn.clicked.connect(self.load_track)
        self._bpm_label = QLabel("-- BPM")
        top_row.addWidget(self._load_btn)
        top_row.addWidget(self._bpm_label)
        top_row.addStretch()
        layout.addLayout(top_row)

        # Waveform
        self._waveform = WaveformWidget()
        layout.addWidget(self._waveform)

        # Transport
        transport = QHBoxLayout()
        for label, slot in [
            ("▶", lambda: None),
            ("⏸", lambda: None),
            ("⏹", lambda: None),
        ]:
            btn = QPushButton(label)
            transport.addWidget(btn)
        layout.addLayout(transport)

        # Knobs — grouped by section
        self._knobs: dict[Param, KnobWidget] = {}
        sections = [
            ("EQ", [Param.EQ_HIGH, Param.EQ_MID, Param.EQ_LOW]),
            ("Filter", [Param.FILTER_CUTOFF, Param.FILTER_RESONANCE, Param.FILTER_TYPE]),
            ("Reverb", [Param.REVERB_SIZE, Param.REVERB_DAMP, Param.REVERB_MIX]),
            ("Echo", [Param.ECHO_TIME, Param.ECHO_FEEDBACK, Param.ECHO_MIX]),
            ("Pitch", [Param.PITCH_SEMITONE, Param.PITCH_CENTS]),
            ("Out", [Param.VOLUME, Param.PAN]),
        ]
        for section_name, params in sections:
            row = QHBoxLayout()
            row.addWidget(QLabel(section_name))
            for param in params:
                r = PARAM_RANGES[param]
                knob = KnobWidget(
                    label=param.name,
                    min_val=r.min_val,
                    max_val=r.max_val,
                    default=r.default,
                )
                knob.valueChanged.connect(
                    lambda v, p=param: self.on_knob_changed(p, v)
                )
                self._knobs[param] = knob
                row.addWidget(knob)
            layout.addLayout(row)

        # Loop controls
        loop_row = QHBoxLayout()
        self._loop_btns = []
        for label, sig in [
            ("IN", lambda: self.loopInSet.emit(self._channel_idx)),
            ("OUT", lambda: self.loopOutSet.emit(self._channel_idx)),
            ("LOOP", lambda: self.loopToggled.emit(self._channel_idx)),
            ("ESC", lambda: self.loopEscaped.emit(self._channel_idx)),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(sig)
            self._loop_btns.append(btn)
            loop_row.addWidget(btn)
        layout.addLayout(loop_row)

        # Hot cues
        cue_row = QHBoxLayout()
        self._hot_cue_btns = []
        for i in range(4):
            btn = QPushButton(f"CUE {i+1}")
            btn.clicked.connect(lambda _, idx=i: self.hotCueRecalled.emit(self._channel_idx, idx))
            self._hot_cue_btns.append(btn)
            cue_row.addWidget(btn)
        layout.addLayout(cue_row)

        # Loop Shaper widget
        self._loop_shaper_w = LoopShaperWidget(channel_color=self._color)
        layout.addWidget(self._loop_shaper_w)

    def _apply_border(self) -> None:
        self.setStyleSheet(
            f"QWidget#channel_{self._channel_idx} {{"
            f"  border: 2px solid {self._color}; border-radius: 4px;"
            f"}}"
        )
        self.setObjectName(f"channel_{self._channel_idx}")
