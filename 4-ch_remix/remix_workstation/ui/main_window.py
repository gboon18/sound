# =============================================================================
# Author  : Ho San Ko
# Email   : hko@avalanche.energy
# Project : 4-Channel Music Remix Workstation
# =============================================================================

"""Main application window — 4 channel strips + master section (Section 7.15)."""

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout

from constants import NUM_CHANNELS, UI_UPDATE_FPS
from ui.channel_strip import ChannelStrip
from ui.master_section import MasterSection


class MainWindow(QMainWindow):
    """Top-level window composing all UI regions.

    Attributes (Section 6 class diagram):
        _channel_strips   List of 4 ChannelStrip widgets.
        _master_section   MasterSection top bar.
    """

    def __init__(self, engine_refs: dict | None = None, parent=None) -> None:
        super().__init__(parent)
        self._engine_refs = engine_refs or {}
        self.setWindowTitle("4-Channel Remix Workstation")
        self.setMinimumSize(1280, 720)

        self._channel_strips: list[ChannelStrip] = []
        self._master_section: MasterSection | None = None

        self.setup_layout()
        self.connect_signals()

        # 30fps UI update timer (reads engine state → updates widgets)
        self._update_timer = QTimer(self)
        self._update_timer.setInterval(1000 // UI_UPDATE_FPS)
        self._update_timer.timeout.connect(self.update_ui)
        self._update_timer.start()

    # ── Layout ────────────────────────────────────────────────────────────────

    def setup_layout(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # Master section at top
        self._master_section = MasterSection()
        root.addWidget(self._master_section)

        # Channel strips side by side
        strips_row = QHBoxLayout()
        for i in range(NUM_CHANNELS):
            strip = ChannelStrip(channel_idx=i)
            self._channel_strips.append(strip)
            strips_row.addWidget(strip)
        root.addLayout(strips_row)

    # ── Signal wiring ─────────────────────────────────────────────────────────

    def connect_signals(self) -> None:
        """Wire all Qt signals to engine slots (filled in when engine is wired)."""
        if self._master_section is None:
            return
        # BPM knob → master clock
        master_clock = self._engine_refs.get("master_clock")
        if master_clock is not None:
            self._master_section.bpmChanged.connect(master_clock.set_bpm)
            self._master_section.tapTempoRequested.connect(master_clock.tap_tempo)

        # Mix bus volume
        mix_bus = self._engine_refs.get("mix_bus")
        if mix_bus is not None:
            self._master_section.masterVolumeChanged.connect(mix_bus.set_master_vol)

    # ── 30fps update ──────────────────────────────────────────────────────────

    def update_ui(self) -> None:
        """Called every ~33ms by the QTimer — read engine state, push to widgets."""
        players = self._engine_refs.get("players", [])
        for i, strip in enumerate(self._channel_strips):
            if i < len(players):
                strip._waveform.set_playhead(players[i].get_playhead())

        master_clock = self._engine_refs.get("master_clock")
        recorder = self._engine_refs.get("recorder")
        if recorder is not None and self._master_section is not None:
            if recorder.is_recording():
                self._master_section.set_rec_elapsed(recorder.elapsed())
