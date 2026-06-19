# =============================================================================
# Author  : Ho San Ko
# Email   : hko@avalanche.energy
# Project : 4-Channel Music Remix Workstation
# =============================================================================

"""Master section widget — BPM, tap tempo, master volume, record, MIDI learn."""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QDoubleSpinBox,
)

from ui.knob_widget import KnobWidget
from constants import DEFAULT_MASTER_BPM, MASTER_BPM_MIN, MASTER_BPM_MAX


class MasterSection(QWidget):
    """Top-bar widget for global controls.

    Attributes (Section 6 class diagram):
        _bpm_knob         KnobWidget controlling master BPM.
        _tap_tempo_btn    QPushButton for tap tempo.
        _master_vol       KnobWidget for master output volume.
        _record_btn       QPushButton to start/stop WAV recording.
        _midi_learn_btn   QPushButton to toggle MIDI learn mode.
        _rec_timer        QLabel displaying elapsed recording time.
    """

    bpmChanged = pyqtSignal(float)
    tapTempoRequested = pyqtSignal()
    masterVolumeChanged = pyqtSignal(float)
    recordToggled = pyqtSignal(bool)
    midiLearnToggled = pyqtSignal(bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # BPM knob
        self._bpm_knob = KnobWidget(
            label="BPM",
            min_val=MASTER_BPM_MIN,
            max_val=MASTER_BPM_MAX,
            default=DEFAULT_MASTER_BPM,
        )
        self._bpm_knob.valueChanged.connect(self.bpmChanged.emit)
        layout.addWidget(self._bpm_knob)

        # Tap tempo
        self._tap_tempo_btn = QPushButton("TAP")
        self._tap_tempo_btn.clicked.connect(self.tapTempoRequested.emit)
        layout.addWidget(self._tap_tempo_btn)

        # BPM numeric readout
        self._bpm_label = QLabel(f"{DEFAULT_MASTER_BPM:.1f}")
        self._bpm_label.setFixedWidth(50)
        layout.addWidget(self._bpm_label)
        self._bpm_knob.valueChanged.connect(
            lambda v: self._bpm_label.setText(f"{v:.1f}")
        )

        layout.addStretch()

        # Master volume
        self._master_vol = KnobWidget(label="VOL", min_val=0.0, max_val=1.0, default=1.0)
        self._master_vol.valueChanged.connect(self.masterVolumeChanged.emit)
        layout.addWidget(self._master_vol)

        # Record button
        self._record_btn = QPushButton("● REC")
        self._record_btn.setCheckable(True)
        self._record_btn.toggled.connect(self.recordToggled.emit)
        layout.addWidget(self._record_btn)

        # Recording timer label
        self._rec_timer = QLabel("00:00")
        layout.addWidget(self._rec_timer)

        # MIDI learn
        self._midi_learn_btn = QPushButton("MIDI LEARN")
        self._midi_learn_btn.setCheckable(True)
        self._midi_learn_btn.toggled.connect(self.midiLearnToggled.emit)
        layout.addWidget(self._midi_learn_btn)

    def set_bpm(self, bpm: float) -> None:
        self._bpm_knob.setValue(bpm)

    def set_rec_elapsed(self, seconds: float) -> None:
        mins = int(seconds) // 60
        secs = int(seconds) % 60
        self._rec_timer.setText(f"{mins:02d}:{secs:02d}")
