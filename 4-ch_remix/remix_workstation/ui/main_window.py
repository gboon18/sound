# =============================================================================
# Author  : Ho San Ko
# Email   : hko@avalanche.energy
# Project : 4-Channel Music Remix Workstation
# =============================================================================

"""Main application window — 4 channel strips + master section (Section 7.15).

Worker thread pattern (Section 9.1 Thread 5+):
  Track loading (file I/O + pyrubberband stretch) runs on QThreadPool workers.
  Completion signals are routed back to the Qt main thread via pyqtSignal.
  The audio callback thread never waits on file I/O.
"""

from PyQt6.QtCore import QObject, QRunnable, QThreadPool, QTimer, pyqtSignal
from PyQt6.QtWidgets import QMainWindow, QMessageBox, QWidget, QVBoxLayout, QHBoxLayout

from constants import NUM_CHANNELS, UI_UPDATE_FPS, Param, AutomationMode
from ui.channel_strip import ChannelStrip
from ui.master_section import MasterSection


# ── Track loader worker ───────────────────────────────────────────────────────

class _LoadSignals(QObject):
    """Carries worker-thread results back to the Qt main thread."""
    finished = pyqtSignal(int, bool, str)   # channel_idx, success, error_msg


class _TrackLoaderWorker(QRunnable):
    """Loads a file on a pool thread; emits finished signal when done."""

    def __init__(
        self,
        channel_idx: int,
        player,          # TrackPlayer
        path: str,
        master_bpm: float,
        signals: _LoadSignals,
    ) -> None:
        super().__init__()
        self.setAutoDelete(True)
        self._ch = channel_idx
        self._player = player
        self._path = path
        self._bpm = master_bpm
        self._signals = signals

    def run(self) -> None:
        try:
            self._player.load_file(self._path, self._bpm)
            self._signals.finished.emit(self._ch, True, "")
        except Exception as exc:
            self._signals.finished.emit(self._ch, False, str(exc))


# ── Main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    """Top-level window composing all UI regions.

    Attributes (Section 6 class diagram):
        _channel_strips   List of 4 ChannelStrip widgets.
        _master_section   MasterSection top bar.
    """

    def __init__(self, engine_refs: dict | None = None, parent=None) -> None:
        super().__init__(parent)
        self._engine_refs: dict = engine_refs or {}
        # Keep _LoadSignals objects alive for the duration of the load
        self._pending_loads: dict[int, _LoadSignals] = {}

        self.setWindowTitle("4-Channel Remix Workstation")
        self.setMinimumSize(1400, 780)

        self._channel_strips: list[ChannelStrip] = []
        self._master_section: MasterSection | None = None

        self.setup_layout()
        self.connect_signals()

        self._update_timer = QTimer(self)
        self._update_timer.setInterval(1000 // UI_UPDATE_FPS)
        self._update_timer.timeout.connect(self.update_ui)
        self._update_timer.start()

    # ── Layout (Section 7.15) ─────────────────────────────────────────────────

    def setup_layout(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        self._master_section = MasterSection()
        root.addWidget(self._master_section)

        strips_row = QHBoxLayout()
        strips_row.setSpacing(4)
        for i in range(NUM_CHANNELS):
            strip = ChannelStrip(channel_idx=i)
            self._channel_strips.append(strip)
            strips_row.addWidget(strip)
        root.addLayout(strips_row)

    # ── Signal wiring ─────────────────────────────────────────────────────────

    def connect_signals(self) -> None:
        """Wire every Qt signal to the correct engine method."""
        ms = self._master_section
        er = self._engine_refs

        master_clock  = er.get("master_clock")
        sync_manager  = er.get("sync_manager")
        mix_bus       = er.get("mix_bus")
        recorder      = er.get("recorder")
        loop_managers = er.get("loop_managers", [])

        if ms is not None:
            if master_clock:
                ms.bpmChanged.connect(master_clock.set_bpm)
                ms.tapTempoRequested.connect(master_clock.tap_tempo)
            if sync_manager:
                ms.bpmChanged.connect(lambda _: sync_manager.recalculate_ratios())
            if mix_bus:
                ms.masterVolumeChanged.connect(mix_bus.set_master_vol)
            if recorder:
                ms.recordToggled.connect(self._on_record_toggled)
            if loop_managers:
                ms.bpmChanged.connect(
                    lambda bpm: [lm.set_master_bpm(bpm) for lm in loop_managers]
                )

        for i, strip in enumerate(self._channel_strips):
            strip.trackLoadRequested.connect(self._on_load_track)
            strip.paramChanged.connect(self._on_param_changed)
            strip.playSignal.connect(self._on_play)
            strip.pauseSignal.connect(self._on_pause)
            strip.stopSignal.connect(self._on_stop)
            strip.loopInSet.connect(self._on_loop_in)
            strip.loopOutSet.connect(self._on_loop_out)
            strip.loopToggled.connect(self._on_loop_toggle)
            strip.loopEscaped.connect(self._on_loop_escape)
            strip.hotCueSet.connect(self._on_hot_cue_set)
            strip.hotCueRecalled.connect(self._on_hot_cue_recall)
            strip.shaperRecordRequested.connect(self._on_shaper_record)
            strip.shaperOverdubRequested.connect(self._on_shaper_overdub)
            strip.shaperClearLane.connect(self._on_shaper_clear_lane)
            strip.shaperClearAll.connect(self._on_shaper_clear_all)
            strip.shaperModeToggle.connect(self._on_shaper_mode_toggle)

    # ── Engine handler methods ────────────────────────────────────────────────

    def _on_load_track(self, channel_idx: int, path: str) -> None:
        players      = self._engine_refs.get("players", [])
        master_clock = self._engine_refs.get("master_clock")
        if channel_idx >= len(players) or master_clock is None:
            return

        signals = _LoadSignals()
        signals.finished.connect(self._on_load_complete)
        self._pending_loads[channel_idx] = signals  # keep alive until done

        self._channel_strips[channel_idx].set_loading(True)
        worker = _TrackLoaderWorker(
            channel_idx, players[channel_idx], path,
            master_clock.get_bpm(), signals,
        )
        QThreadPool.globalInstance().start(worker)

    def _on_load_complete(self, channel_idx: int, success: bool, error: str) -> None:
        self._pending_loads.pop(channel_idx, None)
        strip   = self._channel_strips[channel_idx]
        players = self._engine_refs.get("players", [])

        strip.set_loading(False)
        if success and channel_idx < len(players):
            player = players[channel_idx]
            bpm = player.get_track_bpm()
            strip.set_bpm_text(
                f"{bpm:.1f} BPM" if bpm > 0 else "? BPM — set manually"
            )
            player.play()
            sm = self._engine_refs.get("sync_manager")
            if sm:
                sm.recalculate_ratios()
        else:
            strip.set_bpm_text("Load failed")
            QMessageBox.warning(self, "Load Error", error or "Unknown error")

    def _on_param_changed(self, ch: int, param, val: float) -> None:
        dsps = self._engine_refs.get("channel_dsps", [])
        if ch < len(dsps):
            dsps[ch].set_param(param, val)

    def _on_play(self, ch: int) -> None:
        players = self._engine_refs.get("players", [])
        if ch < len(players):
            players[ch].play()

    def _on_pause(self, ch: int) -> None:
        players = self._engine_refs.get("players", [])
        if ch < len(players):
            players[ch].pause()

    def _on_stop(self, ch: int) -> None:
        players = self._engine_refs.get("players", [])
        if ch < len(players):
            players[ch].stop()

    def _on_loop_in(self, ch: int) -> None:
        players = self._engine_refs.get("players", [])
        lms     = self._engine_refs.get("loop_managers", [])
        if ch < len(players) and ch < len(lms):
            lms[ch].set_loop_in(players[ch].get_playhead())

    def _on_loop_out(self, ch: int) -> None:
        players = self._engine_refs.get("players", [])
        lms     = self._engine_refs.get("loop_managers", [])
        if ch < len(players) and ch < len(lms):
            lms[ch].set_loop_out(players[ch].get_playhead())

    def _on_loop_toggle(self, ch: int) -> None:
        lms = self._engine_refs.get("loop_managers", [])
        if ch < len(lms):
            lms[ch].toggle_loop()

    def _on_loop_escape(self, ch: int) -> None:
        lms = self._engine_refs.get("loop_managers", [])
        if ch < len(lms):
            lms[ch].escape_loop()

    def _on_hot_cue_set(self, ch: int, cue_idx: int) -> None:
        players = self._engine_refs.get("players", [])
        lms     = self._engine_refs.get("loop_managers", [])
        if ch < len(players) and ch < len(lms):
            lms[ch].set_hot_cue(cue_idx, players[ch].get_playhead())

    def _on_hot_cue_recall(self, ch: int, cue_idx: int) -> None:
        players = self._engine_refs.get("players", [])
        lms     = self._engine_refs.get("loop_managers", [])
        if ch < len(players) and ch < len(lms):
            pos = lms[ch].recall_hot_cue(cue_idx)
            if pos is not None:
                players[ch].set_playhead(pos)

    def _on_shaper_record(self, ch: int) -> None:
        shapers = self._engine_refs.get("loop_shapers", [])
        if ch < len(shapers):
            shapers[ch].arm_record()

    def _on_shaper_overdub(self, ch: int) -> None:
        shapers = self._engine_refs.get("loop_shapers", [])
        if ch < len(shapers):
            shapers[ch].arm_overdub()

    def _on_shaper_clear_lane(self, ch: int, param) -> None:
        shapers = self._engine_refs.get("loop_shapers", [])
        if ch < len(shapers):
            shapers[ch].clear_lane(param)

    def _on_shaper_clear_all(self, ch: int) -> None:
        shapers = self._engine_refs.get("loop_shapers", [])
        if ch < len(shapers):
            shapers[ch].clear_all()

    def _on_shaper_mode_toggle(self, ch: int) -> None:
        shapers = self._engine_refs.get("loop_shapers", [])
        if ch < len(shapers):
            shaper = shapers[ch]
            current = shaper.get_mode()
            new_mode = (
                AutomationMode.ADDITIVE
                if current == AutomationMode.ABSOLUTE
                else AutomationMode.ABSOLUTE
            )
            shaper.set_mode(new_mode)

    def _on_record_toggled(self, on: bool) -> None:
        recorder = self._engine_refs.get("recorder")
        if recorder is None:
            return
        if on:
            recorder.start()
        else:
            path = recorder.stop()
            if self._master_section:
                self._master_section.set_rec_elapsed(0.0)

    # ── 30fps update (Section 7.15 Task 7.8) ─────────────────────────────────

    def update_ui(self) -> None:
        """Read current engine state and push to all widgets.

        Called by a 30fps QTimer — never touches the audio callback.
        Reads are thread-safe because all engine state is lock-protected.
        """
        er = self._engine_refs
        players       = er.get("players", [])
        loop_managers = er.get("loop_managers", [])
        loop_shapers  = er.get("loop_shapers", [])
        recorder      = er.get("recorder")

        for i, strip in enumerate(self._channel_strips):
            if i >= len(players):
                break
            player = players[i]

            # Waveform playhead
            strip.set_playhead(player.get_playhead())

            # Loop region overlay
            if i < len(loop_managers):
                lm = loop_managers[i]
                strip.set_loop_region(*lm.get_loop_bounds()) if lm.get_loop_bounds() else strip.set_loop_region(None, None)

            # Loop Shaper: lane data + normalized playhead
            if i < len(loop_shapers):
                ls = loop_shapers[i]
                lane_data = {p: ls.get_lane(p) for p in Param}
                strip._loop_shaper_w.update_lanes(lane_data)

                if i < len(loop_managers):
                    lm = loop_managers[i]
                    if lm.is_loop_active():
                        bounds = lm.get_loop_bounds()
                        if bounds:
                            li, lo = bounds
                            ph = player.get_playhead()
                            ll = lo - li
                            if ll > 0:
                                norm = max(0.0, min(1.0, (ph - li) / ll))
                                strip._loop_shaper_w.set_playhead(norm)

        # Recording elapsed timer
        if recorder and recorder.is_recording() and self._master_section:
            self._master_section.set_rec_elapsed(recorder.elapsed())
