# =============================================================================
# Author  : Ho San Ko
# Email   : hko@avalanche.energy
# Project : 4-Channel Music Remix Workstation
# =============================================================================

"""Application entry point — wires all components and starts the Qt event loop.

Wire-up order follows Section 7.1 of the engineering spec.
Phases implemented here:
  Phase 3 — 4-track sync via SyncManager
  Phase 5 — loop-aware advance, loop-wrap detection
  Phase 6 — Loop Shaper notification on wrap
  Phase 8 — complete MIDI dispatch (set/toggle/trigger/momentary)
  Phase 9 — recording punch-in on loop start
  Phase 10 — pre-allocated output buffer, underrun counter
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd

from constants import (
    NUM_CHANNELS, SAMPLE_RATE, DEFAULT_BUFFER_SIZE, DEFAULT_MASTER_BPM,
    Param, PARAM_RANGES, AutomationMode,
)
from engine.master_clock import MasterClock
from engine.track_player import TrackPlayer
from engine.bpm_detector import BpmDetector
from engine.sync_manager import SyncManager
from engine.mix_bus import MixBus
from dsp.channel_dsp import ChannelDSP
from loop.loop_manager import LoopManager
from loop.loop_shaper import LoopShaper
from midi.midi_map import MidiMap, MidiAddress, ControlTarget
from midi.midi_input import MidiInput
from midi.midi_learn import MidiLearn
from recording.recorder import Recorder


def _load_settings() -> dict:
    path = Path("config/settings.json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def main() -> None:
    # ── CLI args ──────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="4-Channel Remix Workstation")
    parser.add_argument("--buffer-size", type=int, default=None)
    parser.add_argument("--sample-rate", type=int, default=None)
    parser.add_argument("--no-ui", action="store_true", help="Headless / CLI mode")
    args = parser.parse_args()

    settings    = _load_settings()
    buffer_size = args.buffer_size or settings.get("buffer_size", DEFAULT_BUFFER_SIZE)
    sample_rate = args.sample_rate or settings.get("sample_rate", SAMPLE_RATE)
    master_bpm  = settings.get("master_bpm", DEFAULT_MASTER_BPM)

    # ── Engine instantiation (Section 7.1 steps 3-11) ─────────────────────────
    master_clock = MasterClock(bpm=master_bpm)

    players:       list[TrackPlayer]  = []
    channel_dsps:  list[ChannelDSP]   = []
    loop_managers: list[LoopManager]  = []
    loop_shapers:  list[LoopShaper]   = []

    for _ in range(NUM_CHANNELS):
        player = TrackPlayer(sample_rate=sample_rate, buffer_size=buffer_size)
        players.append(player)
        channel_dsps.append(ChannelDSP(sample_rate=sample_rate))
        lm = LoopManager(sample_rate=sample_rate, master_bpm=master_bpm)
        loop_managers.append(lm)
        loop_shapers.append(LoopShaper(loop_manager=lm))

    for i in range(NUM_CHANNELS):
        players[i].set_loop_manager(loop_managers[i])

    sync_manager = SyncManager(master_clock=master_clock, players=players)
    mix_bus      = MixBus(num_channels=NUM_CHANNELS)
    recorder     = Recorder(sample_rate=sample_rate)

    midi_map   = MidiMap()
    midi_map.load("config/midi_map.json")
    midi_input = MidiInput()
    midi_learn = MidiLearn(midi_map=midi_map)

    # ── Phase 8: Complete MIDI dispatch ───────────────────────────────────────
    # Action routing table for non-DSP string targets
    #   key = target.param (str)   value = callable(ch, value)
    # "value" is the raw MIDI value 0-127 (for Notes: >0 = On, 0 = Off)

    def _action_play(ch: int, val: int) -> None:
        if val > 0:
            if players[ch].is_playing():
                players[ch].pause()
            else:
                players[ch].play()

    def _action_stop(ch: int, val: int) -> None:
        if val > 0:
            players[ch].stop()

    def _action_loop_in(ch: int, val: int) -> None:
        if val > 0:
            loop_managers[ch].set_loop_in(players[ch].get_playhead())

    def _action_loop_out(ch: int, val: int) -> None:
        if val > 0:
            loop_managers[ch].set_loop_out(players[ch].get_playhead())

    def _action_loop_toggle(ch: int, val: int) -> None:
        if val > 0:
            loop_managers[ch].toggle_loop()

    def _action_loop_escape(ch: int, val: int) -> None:
        if val > 0:
            loop_managers[ch].escape_loop()

    def _action_tap_tempo(_ch: int, val: int) -> None:
        if val > 0:
            master_clock.tap_tempo()

    def _action_record(_ch: int, val: int) -> None:
        if val > 0:
            if recorder.is_recording():
                recorder.stop()
            else:
                recorder.start()

    def _action_punch_in(_ch: int, val: int) -> None:
        if val > 0:
            recorder.arm_punch_in()

    def _action_shaper_record(ch: int, val: int) -> None:
        if val > 0:
            loop_shapers[ch].arm_record()

    def _action_shaper_overdub(ch: int, val: int) -> None:
        if val > 0:
            loop_shapers[ch].arm_overdub()

    def _make_hot_cue_recall(cue_idx: int):
        def _fn(ch: int, val: int) -> None:
            if val > 0:
                pos = loop_managers[ch].recall_hot_cue(cue_idx)
                if pos is not None:
                    players[ch].set_playhead(pos)
        return _fn

    def _make_hot_cue_set(cue_idx: int):
        def _fn(ch: int, val: int) -> None:
            if val > 0:
                loop_managers[ch].set_hot_cue(cue_idx, players[ch].get_playhead())
        return _fn

    _STRING_ACTIONS = {
        "play":           _action_play,
        "stop":           _action_stop,
        "loop_in":        _action_loop_in,
        "loop_out":       _action_loop_out,
        "loop_toggle":    _action_loop_toggle,
        "loop_escape":    _action_loop_escape,
        "tap_tempo":      _action_tap_tempo,
        "record":         _action_record,
        "punch_in":       _action_punch_in,
        "shaper_record":  _action_shaper_record,
        "shaper_overdub": _action_shaper_overdub,
        **{f"hot_cue_{i}":     _make_hot_cue_recall(i) for i in range(4)},
        **{f"hot_cue_set_{i}": _make_hot_cue_set(i)    for i in range(4)},
    }

    def midi_dispatch(addr: MidiAddress, value: int) -> None:
        """Route an incoming MIDI message to the correct engine action.

        Phase 8 action model (Section 7.11-7.13):
          "set"     + CC    → scale 0-127 → param range → channel_dsp.set_param()
          "set"     + Note  → binary: 0 or max value
          "toggle"  + Note  → fire on Note On; Note Off is ignored
          "trigger" + Note  → same as toggle (one-shot)
          String target     → dispatch via _STRING_ACTIONS table
        """
        if midi_learn.is_active():
            midi_learn.on_midi(addr, value)
            return

        target: ControlTarget | None = midi_map.lookup(addr)
        if target is None:
            return

        ch = target.channel_idx

        # ── DSP parameter (Param enum) ─────────────────────────────────────
        if isinstance(target.param, Param):
            param = target.param
            r = PARAM_RANGES[param]
            if target.action == "set":
                if addr.msg_type == "cc":
                    # CC: scale 0-127 linearly to param range (Section 8.5)
                    param_val = r.min_val + (value / 127.0) * (r.max_val - r.min_val)
                else:
                    # Note: binary — On maps to max, Off maps to default
                    param_val = r.max_val if value > 0 else r.default
                if 0 <= ch < NUM_CHANNELS:
                    channel_dsps[ch].set_param(param, param_val)

        # ── Non-DSP action (string) ────────────────────────────────────────
        elif isinstance(target.param, str):
            fn = _STRING_ACTIONS.get(target.param)
            if fn is not None:
                safe_ch = ch if 0 <= ch < NUM_CHANNELS else 0
                fn(safe_ch, value)

    midi_input.set_callback(midi_dispatch)
    ports = midi_input.list_ports()
    if ports:
        try:
            midi_input.open(settings.get("midi_port_index", 0))
        except Exception:
            pass  # no hardware; soft-fail (edge case: missing MIDI device)

    # ── Phase 10: Pre-allocated audio output buffer ────────────────────────────
    # Avoid allocating a new numpy array every callback (Section 10.3).
    _out_buf  = np.zeros((buffer_size, 2), dtype=np.float32)
    _underrun_count = 0

    # Loop-wrap state for each channel (audio-callback thread, no lock needed)
    _prev_wrap = [0] * NUM_CHANNELS

    # ── Audio callback (Section 7.1 steps 17 / Phase 3+5+8+9+10) ─────────────
    def audio_callback(
        outdata: np.ndarray, frames: int, time_info, status
    ) -> None:
        nonlocal _underrun_count

        # Phase 10: underrun accounting
        if status and status.output_underflow:
            _underrun_count += 1

        # Phase 3: advance all players (drift-corrected)
        channel_bufs: list[np.ndarray] = sync_manager.advance_all(frames)

        for i in range(NUM_CHANNELS):
            # Phase 5+6: detect loop boundary → notify Loop Shaper + Recorder
            wc = players[i].get_wrap_count()
            if wc != _prev_wrap[i]:
                _prev_wrap[i] = wc
                loop_shapers[i].on_loop_wrap()     # Phase 6: ARMED→RECORDING/OVERDUBBING
                recorder.on_loop_start()            # Phase 9: punch-in if armed

            # Loop Shaper evaluation → DSP param overrides (Section 3.1 / 7.10)
            bounds = loop_managers[i].get_loop_bounds()
            if bounds and loop_managers[i].is_loop_active():
                loop_in, loop_out = bounds
                ph       = players[i].get_playhead()
                loop_len = loop_out - loop_in
                if loop_len > 0:
                    norm_pos = max(0.0, min(1.0, (ph - loop_in) / loop_len))
                    overrides = loop_shapers[i].evaluate(norm_pos)
                    for param, val in overrides.items():
                        channel_dsps[i].set_param(param, val)

            channel_bufs[i] = channel_dsps[i].process(channel_bufs[i])

        # Phase 10: mix directly into the pre-allocated scratch buffer,
        # then copy into sounddevice's outdata — avoids one allocation/callback.
        mixed = mix_bus.mix(channel_bufs, out=_out_buf[:frames])

        # Phase 9: capture to WAV
        recorder.write(mixed)

        np.copyto(outdata, mixed)

    # ── Qt UI (Section 7.1 steps 14-16, 18-19) ───────────────────────────────
    if not args.no_ui:
        from PyQt6.QtWidgets import QApplication
        from ui.main_window import MainWindow

        app = QApplication(sys.argv)
        app.setApplicationName("4-Channel Remix Workstation")

        engine_refs = {
            "master_clock":  master_clock,
            "sync_manager":  sync_manager,
            "players":       players,
            "channel_dsps":  channel_dsps,
            "loop_managers": loop_managers,
            "loop_shapers":  loop_shapers,
            "mix_bus":       mix_bus,
            "recorder":      recorder,
            "midi_map":      midi_map,
            "midi_learn":    midi_learn,
        }

        window = MainWindow(engine_refs=engine_refs)
        window.show()

        stream = sd.OutputStream(
            samplerate=sample_rate,
            blocksize=buffer_size,
            channels=2,
            dtype="float32",
            callback=audio_callback,
        )
        with stream:
            master_clock.start()
            exit_code = app.exec()
            master_clock.stop()

        midi_input.close()
        midi_map.save("config/midi_map.json")
        if _underrun_count > 0:
            print(
                f"[Warning] {_underrun_count} audio underrun(s) detected. "
                "Consider increasing buffer size."
            )
        sys.exit(exit_code)

    else:
        # ── Headless / CLI test mode ──────────────────────────────────────────
        stream = sd.OutputStream(
            samplerate=sample_rate,
            blocksize=buffer_size,
            channels=2,
            dtype="float32",
            callback=audio_callback,
        )
        with stream:
            master_clock.start()
            print("Headless mode — press Ctrl-C to stop.")
            try:
                import time
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                pass
            master_clock.stop()
        midi_input.close()
        if _underrun_count > 0:
            print(f"[Warning] {_underrun_count} underrun(s) during session.")


if __name__ == "__main__":
    main()
