# =============================================================================
# Author  : Ho San Ko
# Email   : hko@avalanche.energy
# Project : 4-Channel Music Remix Workstation
# =============================================================================

"""Application entry point — wires all components and starts the Qt event loop.

Wire-up order follows Section 7.1 of the engineering spec.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd

from constants import (
    NUM_CHANNELS, SAMPLE_RATE, DEFAULT_BUFFER_SIZE, DEFAULT_MASTER_BPM,
    PARAM_RANGES,
)
from engine.master_clock import MasterClock
from engine.track_player import TrackPlayer
from engine.bpm_detector import BpmDetector
from engine.sync_manager import SyncManager
from engine.mix_bus import MixBus
from dsp.channel_dsp import ChannelDSP
from loop.loop_manager import LoopManager
from loop.loop_shaper import LoopShaper
from midi.midi_map import MidiMap, MidiAddress
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
    parser.add_argument("--no-ui", action="store_true", help="Headless mode (CLI)")
    args = parser.parse_args()

    settings = _load_settings()
    buffer_size: int = args.buffer_size or settings.get("buffer_size", DEFAULT_BUFFER_SIZE)
    sample_rate: int = args.sample_rate or settings.get("sample_rate", SAMPLE_RATE)
    master_bpm: float = settings.get("master_bpm", DEFAULT_MASTER_BPM)

    # ── Engine instantiation (Step 3–11 of Section 7.1) ──────────────────────
    master_clock = MasterClock(bpm=master_bpm)
    bpm_detector = BpmDetector()

    players: list[TrackPlayer] = []
    channel_dsps: list[ChannelDSP] = []
    loop_managers: list[LoopManager] = []
    loop_shapers: list[LoopShaper] = []

    for _ in range(NUM_CHANNELS):
        player = TrackPlayer(sample_rate=sample_rate, buffer_size=buffer_size)
        players.append(player)
        channel_dsps.append(ChannelDSP(sample_rate=sample_rate))
        lm = LoopManager(sample_rate=sample_rate)
        loop_managers.append(lm)
        loop_shapers.append(LoopShaper(loop_manager=lm))

    sync_manager = SyncManager(master_clock=master_clock, players=players)
    mix_bus = MixBus(num_channels=NUM_CHANNELS)
    recorder = Recorder(sample_rate=sample_rate)

    midi_map = MidiMap()
    midi_map.load("config/midi_map.json")
    midi_input = MidiInput()
    midi_learn = MidiLearn(midi_map=midi_map)

    # ── MIDI dispatch (Step 13 of Section 7.1) ────────────────────────────────
    def midi_dispatch(addr: MidiAddress, value: int) -> None:
        if midi_learn.is_active():
            midi_learn.on_midi(addr, value)
            return
        target = midi_map.lookup(addr)
        if target is None:
            return
        if target.action == "set":
            from constants import Param, PARAM_RANGES
            if isinstance(target.param, Param):
                r = PARAM_RANGES[target.param]
                normalized = value / 127.0
                param_val = r.min_val + normalized * (r.max_val - r.min_val)
                ch = target.channel_idx
                if 0 <= ch < NUM_CHANNELS:
                    channel_dsps[ch].set_param(target.param, param_val)

    midi_input.set_callback(midi_dispatch)
    ports = midi_input.list_ports()
    if ports:
        try:
            midi_input.open(settings.get("midi_port_index", 0))
        except Exception:
            pass  # No MIDI hardware; continue without it

    # ── Audio callback (Step 17 of Section 7.1) ───────────────────────────────
    def audio_callback(outdata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            pass  # log underrun etc.

        channel_bufs: list[np.ndarray] = []
        for i in range(NUM_CHANNELS):
            buf = players[i].advance(frames)

            bounds = loop_managers[i].get_loop_bounds()
            if bounds:
                loop_in, loop_out = bounds
                ph = players[i].get_playhead()
                loop_len = loop_out - loop_in
                if loop_len > 0:
                    norm_pos = (ph - loop_in) / loop_len
                    norm_pos = max(0.0, min(1.0, norm_pos))
                    overrides = loop_shapers[i].evaluate(norm_pos)
                    for param, val in overrides.items():
                        channel_dsps[i].set_param(param, val)

            buf = channel_dsps[i].process(buf)
            channel_bufs.append(buf)

        mixed = mix_bus.mix(channel_bufs)
        recorder.write(mixed)
        outdata[:] = mixed

    # ── Qt UI (Steps 14–16, 18–19) ────────────────────────────────────────────
    if not args.no_ui:
        from PyQt6.QtWidgets import QApplication
        from ui.main_window import MainWindow

        app = QApplication(sys.argv)
        app.setApplicationName("4-Channel Remix Workstation")

        engine_refs = {
            "master_clock": master_clock,
            "players": players,
            "channel_dsps": channel_dsps,
            "loop_managers": loop_managers,
            "loop_shapers": loop_shapers,
            "mix_bus": mix_bus,
            "recorder": recorder,
            "midi_map": midi_map,
            "midi_learn": midi_learn,
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
        sys.exit(exit_code)
    else:
        # Headless: just run the clock for testing
        stream = sd.OutputStream(
            samplerate=sample_rate,
            blocksize=buffer_size,
            channels=2,
            dtype="float32",
            callback=audio_callback,
        )
        with stream:
            master_clock.start()
            print("Running headless. Press Ctrl+C to stop.")
            try:
                import time
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                pass
            master_clock.stop()
        midi_input.close()


if __name__ == "__main__":
    main()
