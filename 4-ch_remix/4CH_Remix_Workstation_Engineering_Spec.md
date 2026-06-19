# 4-Channel Music Remix Workstation — Complete Engineering Specification

```
Author  : Ho San Ko
Email   : hko@avalanche.energy
Project : 4-Channel Music Remix Workstation
Version : 1.0.0
Date    : 2026-06-19
```

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview & Rationale](#2-architecture-overview--rationale)
3. [Data Flow Diagrams](#3-data-flow-diagrams)
4. [File Structure](#4-file-structure)
5. [Constants & Parameter Ranges](#5-constants--parameter-ranges)
6. [Complete Class Diagram (ASCII)](#6-complete-class-diagram-ascii)
7. [Module-by-Module Specification](#7-module-by-module-specification)
8. [Edge Cases & Mitigation](#8-edge-cases--mitigation)
9. [Threading Model & Concurrency](#9-threading-model--concurrency)
10. [Phased Implementation Plan](#10-phased-implementation-plan)
11. [Unit Test Plan](#11-unit-test-plan)
12. [Performance Budget](#12-performance-budget)

---

## 1. Executive Summary

This document specifies a Python-based, real-time 4-channel DJ/remix workstation that fuses Rekordbox-style loop DJing with Ableton-style live automation. The centrepiece creative feature is the **Loop Shaper** — a per-channel gesture automation recorder that captures knob movements across all 16 DSP parameters and replays them in perfect sync with every subsequent loop iteration.

The system is built on three pillars:

1. **Single Master Clock** — one timing source drives all four channel playheads; every track is time-stretched to the master BPM via `pyrubberband`, preserving pitch.
2. **Per-Channel DSP Chain** — powered by Spotify's `pedalboard` library, each channel has 16 independently controllable effect parameters.
3. **Loop Shaper** — records real-time parameter gestures mapped to the loop timeline with overdub, per-lane clear, additive/absolute modes, and proportional rescale on loop-length change.

All controls are MIDI-mappable via a learn-mode system supporting commercial and custom DIY controllers.

---

## 2. Architecture Overview & Rationale

### 2.1 High-Level Architecture

The system follows a **layered architecture** with strict unidirectional data flow:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        UI LAYER (PyQt6)                             │
│  main_window ─► channel_strip ─► knob_widget / waveform_widget     │
│                                  loop_shaper_widget                 │
│                 master_section                                      │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ signals / slots (Qt)
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     CONTROL LAYER                                   │
│  midi_input ─► midi_learn ─► midi_map                              │
│  (python-rtmidi)                                                    │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ parameter change events
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     ENGINE LAYER                                    │
│  master_clock ─► sync_manager ─► track_player (×4)                 │
│                                  bpm_detector                       │
│  loop_manager ─► loop_shaper                                       │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ audio buffers (numpy arrays)
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       DSP LAYER                                     │
│  channel_dsp ─► eq ─► filter ─► reverb ─► echo ─► pitch            │
│  (pedalboard)                                                       │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ processed audio buffers
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     OUTPUT LAYER                                    │
│  sounddevice callback ◄── mix_bus (sum 4 channels)                  │
│  recorder (WAV capture)                                             │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Architectural Rationale

| Decision | Rationale |
|---|---|
| **Single master clock thread** | Eliminates inter-channel drift. One `threading.Event`-based tick engine is simpler and more deterministic than four independent clocks. |
| **pyrubberband for time-stretch** | Phase-vocoder stretch preserves pitch at any ratio (including non-integer). Rubberband is industry-standard quality. |
| **pedalboard for DSP** | Spotify's `pedalboard` runs native C++ under the hood, giving near-zero latency per effect. Avoids reinventing EQ/reverb/delay. |
| **Layered separation** | Audio processing never runs on the Qt event loop. The UI thread only reads shared state via locks or Qt signals, never blocks on audio. |
| **Automation as list[tuple[float, float]]** | Normalized 0.0–1.0 position makes rescaling trivial on loop-length change. Linear interpolation between points is cheap. |
| **MIDI mappings in JSON** | Human-readable, version-controllable, portable between machines. |
| **No global mutable state** | All dependencies injected via constructors. Makes unit testing straightforward and eliminates hidden coupling. |

---

## 3. Data Flow Diagrams

### 3.1 Audio Playback Data Flow (Per Channel)

```
[Audio File on Disk]
        │
        ▼
  ┌──────────────┐
  │  soundfile    │  Load raw PCM (float32, mono/stereo)
  │  / pydub      │  Normalize to float32 numpy array
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │ bpm_detector  │  librosa.beat.beat_track()
  │               │  Output: detected_bpm (float)
  └──────┬───────┘
         │
         ▼
  ┌──────────────────────┐
  │ pyrubberband stretch │  ratio = master_bpm / track_bpm
  │                      │  Output: stretched_audio (numpy)
  └──────┬───────────────┘
         │
         ▼
  ┌──────────────────────┐
  │ track_player          │  Maintains playhead (sample index)
  │                       │  Reads buffer of N samples per tick
  │                       │  Applies loop logic (wrap playhead)
  └──────┬───────────────┘
         │  raw audio chunk (numpy float32)
         ▼
  ┌──────────────────────┐
  │ loop_shaper           │  Reads current normalized position
  │                       │  Interpolates automation values
  │                       │  Overrides / offsets DSP params
  └──────┬───────────────┘
         │  param overrides dict
         ▼
  ┌──────────────────────┐
  │ channel_dsp           │  Applies pedalboard chain:
  │  eq → filter →        │  EQ → Filter → Reverb → Echo → Pitch
  │  reverb → echo →      │  Then Volume + Pan
  │  pitch → vol/pan      │
  └──────┬───────────────┘
         │  processed audio chunk
         ▼
  ┌──────────────────────┐
  │ mix_bus                │  Sum all 4 channel outputs
  │                        │  Apply master volume
  └──────┬───────────────┘
         │
    ┌────┴─────┐
    ▼          ▼
 [sounddevice] [recorder]
  (output)     (WAV file)
```

### 3.2 Master Clock Tick Flow

```
  ┌──────────────────┐
  │  master_clock     │
  │  (dedicated       │
  │   thread)         │
  └──────┬───────────┘
         │  tick event (every buffer_size / sample_rate seconds)
         │
         ├───► sync_manager.advance_all()
         │         │
         │         ├── track_player[0].advance(frames)
         │         ├── track_player[1].advance(frames)
         │         ├── track_player[2].advance(frames)
         │         └── track_player[3].advance(frames)
         │
         ├───► loop_manager.check_boundaries()
         │         │
         │         └── For each channel: if playhead >= loop_out → wrap to loop_in
         │
         ├───► loop_shaper.evaluate_all()
         │         │
         │         └── For each channel with active automation:
         │             compute normalized_position, interpolate params
         │
         └───► ui_update_signal.emit()  (throttled to ~30 fps)
```

### 3.3 Loop Shaper Record / Playback Flow

```
  ┌──────────────────────────────────────────────────────────┐
  │  RECORD MODE                                              │
  │                                                           │
  │  User presses Record → loop_shaper.state = RECORDING     │
  │  Wait for next loop_start (clean alignment)               │
  │                                                           │
  │  Each knob change event:                                  │
  │    normalized_pos = (playhead - loop_in)                  │
  │                     / (loop_out - loop_in)                │
  │    automation_lane[param].append(                         │
  │        (normalized_pos, param_value)                      │
  │    )                                                      │
  │                                                           │
  │  When playhead wraps (one full pass):                     │
  │    loop_shaper.state = PLAYING                            │
  │    Sort each lane by normalized_pos                       │
  └──────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────┐
  │  PLAYBACK MODE                                            │
  │                                                           │
  │  Each audio tick:                                         │
  │    norm_pos = (playhead - loop_in) / (loop_out - loop_in)│
  │                                                           │
  │    For each param with non-empty automation:              │
  │      value = linear_interpolate(lane, norm_pos)           │
  │                                                           │
  │      if mode == ABSOLUTE:                                 │
  │        dsp_param = value                                  │
  │      elif mode == ADDITIVE:                               │
  │        dsp_param = clamp(manual_value + value, min, max)  │
  └──────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────┐
  │  OVERDUB MODE                                             │
  │                                                           │
  │  Same as RECORD but:                                      │
  │    - Only touched parameters are overwritten              │
  │    - Untouched lanes keep existing automation             │
  │    - New points replace existing points in same region    │
  │      (within ±tolerance of normalized_pos)                │
  │    - Non-overlapping regions merge intact                 │
  └──────────────────────────────────────────────────────────┘
```

### 3.4 MIDI Signal Flow

```
  [Physical MIDI Controller]
        │
        │  USB / DIN MIDI
        ▼
  ┌──────────────┐
  │ python-rtmidi │  Raw MIDI bytes
  └──────┬───────┘
         │
         ▼
  ┌──────────────────────┐
  │ midi_input            │  Parse message type:
  │                       │    CC  → (channel, cc_number, value 0–127)
  │                       │    NoteOn  → (channel, note, velocity)
  │                       │    NoteOff → (channel, note, 0)
  └──────┬───────────────┘
         │
         ▼
  ┌──────────────────────┐         ┌───────────────────────┐
  │ midi_learn            │◄────────│ midi_map (JSON store) │
  │                       │         │                       │
  │  If learn_mode == ON: │         │  lookup:              │
  │    assign incoming CC │         │  (ch, cc) → target    │
  │    to selected control│         │  (ch, note) → target  │
  │                       │         │                       │
  │  If learn_mode == OFF:│────────►│  resolve target:      │
  │    lookup mapping     │         │  channel_idx, param,  │
  │    dispatch to target │         │  action               │
  └──────────────────────┘         └───────────┬───────────┘
                                               │
                                               ▼
                                   ┌──────────────────────┐
                                   │ target control        │
                                   │  (knob / button /     │
                                   │   loop_shaper_record) │
                                   └──────────────────────┘
```

---

## 4. File Structure

```
remix_workstation/
│
├── main.py                        # Application entry point
├── constants.py                   # All parameter ranges, named constants
│
├── engine/
│   ├── __init__.py
│   ├── master_clock.py            # Master BPM clock, tick engine, threading
│   ├── track_player.py            # Per-channel audio player, playhead, stretch
│   ├── bpm_detector.py            # librosa BPM detection + manual override
│   ├── sync_manager.py            # Drift correction, ratio calculation
│   └── mix_bus.py                 # Sum 4 channels, apply master volume
│
├── dsp/
│   ├── __init__.py
│   ├── channel_dsp.py             # pedalboard FX chain per channel (orchestrator)
│   ├── eq.py                      # 3-band EQ wrapper
│   ├── filter.py                  # LP/HP filter wrapper
│   ├── reverb.py                  # Reverb wrapper
│   ├── echo.py                    # Delay/echo wrapper
│   └── pitch.py                   # Pitch shift wrapper
│
├── loop/
│   ├── __init__.py
│   ├── loop_manager.py            # Loop in/out, escape, hot cues
│   └── loop_shaper.py             # Automation recorder, overdub, rescale, replay
│
├── midi/
│   ├── __init__.py
│   ├── midi_input.py              # rtmidi listener, CC + Note parsing
│   ├── midi_learn.py              # MIDI learn mode, mapping assignment
│   └── midi_map.py                # Mapping store, JSON load/save, conflict detect
│
├── ui/
│   ├── __init__.py
│   ├── main_window.py             # PyQt6 main window, top-level layout
│   ├── channel_strip.py           # Per-channel UI widget (composite)
│   ├── master_section.py          # Master BPM, record, MIDI learn
│   ├── knob_widget.py             # Custom rotary knob widget (QWidget)
│   ├── waveform_widget.py         # Scrolling waveform + beat grid overlay
│   └── loop_shaper_widget.py      # Automation lane display + controls
│
├── recording/
│   ├── __init__.py
│   └── recorder.py                # WAV capture, punch-in logic
│
├── config/
│   ├── midi_map.json              # Saved MIDI mappings
│   └── settings.json              # User preferences, buffer size, etc.
│
└── tests/
    ├── __init__.py
    ├── test_bpm_detector.py
    ├── test_sync_manager.py
    ├── test_loop_shaper.py
    ├── test_midi_map.py
    └── test_mix_bus.py
```

### 4.1 File Structure Rationale

| Directory | Purpose |
|---|---|
| `engine/` | All timing-critical, audio-thread code. Zero UI imports. |
| `dsp/` | Thin wrappers around `pedalboard` plugins. Each file owns one effect type. `channel_dsp.py` chains them in order. |
| `loop/` | Loop boundaries and the Loop Shaper are separated from the playback engine so they can be unit-tested independently. |
| `midi/` | Fully decoupled from UI — MIDI events are dispatched to abstract targets, not Qt widgets directly. |
| `ui/` | All PyQt6 code lives here. No audio processing. Communicates via Qt signals/slots. |
| `recording/` | Isolated recorder module — subscribes to the mix bus output, writes WAV. |
| `config/` | JSON files only. No code. |
| `tests/` | Mirror of source structure. Pytest-compatible. |

---

## 5. Constants & Parameter Ranges

File: `constants.py`

```python
# =============================================================================
# Author  : Ho San Ko
# Email   : hko@avalanche.energy
# Project : 4-Channel Music Remix Workstation
# =============================================================================

from enum import Enum, auto
from typing import NamedTuple


# ── General ──────────────────────────────────────────────────────────────────

NUM_CHANNELS: int = 4
SAMPLE_RATE: int = 44100
DEFAULT_BUFFER_SIZE: int = 512
DEFAULT_MASTER_BPM: float = 120.0

MASTER_BPM_MIN: float = 60.0
MASTER_BPM_MAX: float = 300.0

HOT_CUES_PER_CHANNEL: int = 4

SUPPORTED_FORMATS: tuple[str, ...] = (".mp3", ".wav", ".flac", ".aiff", ".aif")

# ── UI ───────────────────────────────────────────────────────────────────────

UI_UPDATE_FPS: int = 30
CHANNEL_COLORS: tuple[str, ...] = ("#E74C3C", "#3498DB", "#2ECC71", "#F39C12")

# ── Loop Lengths (in bars) ───────────────────────────────────────────────────

LOOP_LENGTHS_BARS: tuple[int, ...] = (1, 2, 4, 8, 16, 32)


# ── DSP Parameter Enum ──────────────────────────────────────────────────────

class Param(Enum):
    EQ_HIGH = auto()
    EQ_MID = auto()
    EQ_LOW = auto()
    FILTER_CUTOFF = auto()
    FILTER_RESONANCE = auto()
    FILTER_TYPE = auto()
    REVERB_SIZE = auto()
    REVERB_DAMP = auto()
    REVERB_MIX = auto()
    ECHO_TIME = auto()
    ECHO_FEEDBACK = auto()
    ECHO_MIX = auto()
    PITCH_SEMITONE = auto()
    PITCH_CENTS = auto()
    VOLUME = auto()
    PAN = auto()


# ── Parameter Range Definition ──────────────────────────────────────────────

class ParamRange(NamedTuple):
    min_val: float
    max_val: float
    default: float
    unit: str


PARAM_RANGES: dict[Param, ParamRange] = {
    Param.EQ_HIGH:          ParamRange(-12.0,   12.0,    0.0,  "dB"),
    Param.EQ_MID:           ParamRange(-12.0,   12.0,    0.0,  "dB"),
    Param.EQ_LOW:           ParamRange(-12.0,   12.0,    0.0,  "dB"),
    Param.FILTER_CUTOFF:    ParamRange( 20.0, 20000.0, 20000.0, "Hz"),
    Param.FILTER_RESONANCE: ParamRange(  0.1,   10.0,    0.707, "Q"),
    Param.FILTER_TYPE:      ParamRange(  0.0,    1.0,    0.0,  "enum"),  # 0=LP, 1=HP
    Param.REVERB_SIZE:      ParamRange(  0.0,    1.0,    0.3,  ""),
    Param.REVERB_DAMP:      ParamRange(  0.0,    1.0,    0.5,  ""),
    Param.REVERB_MIX:       ParamRange(  0.0,    1.0,    0.0,  ""),
    Param.ECHO_TIME:        ParamRange( 10.0, 2000.0,  500.0,  "ms"),
    Param.ECHO_FEEDBACK:    ParamRange(  0.0,    0.95,   0.3,  ""),
    Param.ECHO_MIX:         ParamRange(  0.0,    1.0,    0.0,  ""),
    Param.PITCH_SEMITONE:   ParamRange(-12.0,   12.0,    0.0,  "st"),
    Param.PITCH_CENTS:      ParamRange(-100.0, 100.0,    0.0,  "cents"),
    Param.VOLUME:           ParamRange(  0.0,    1.0,    0.8,  ""),
    Param.PAN:              ParamRange( -1.0,    1.0,    0.0,  ""),     # -1=L, +1=R
}


# ── Automation Modes ────────────────────────────────────────────────────────

class AutomationMode(Enum):
    ABSOLUTE = auto()   # Automation value replaces manual knob
    ADDITIVE = auto()   # Automation value adds to manual knob (clamped)


# ── Loop Shaper States ──────────────────────────────────────────────────────

class LoopShaperState(Enum):
    IDLE = auto()
    ARMED = auto()       # Waiting for next loop start to begin recording
    RECORDING = auto()
    OVERDUBBING = auto()
    PLAYING = auto()


# ── MIDI ─────────────────────────────────────────────────────────────────────

MIDI_CC_MIN: int = 0
MIDI_CC_MAX: int = 127
MIDI_CHANNELS: int = 16

# ── Filter Types ─────────────────────────────────────────────────────────────

class FilterType(Enum):
    LOWPASS = 0
    HIGHPASS = 1
```

---

## 6. Complete Class Diagram (ASCII)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              APPLICATION                                     │
│                                                                              │
│  main.py                                                                     │
│    └── creates: MainWindow, MasterClock, SyncManager,                        │
│                 TrackPlayer[4], ChannelDSP[4], LoopManager[4],               │
│                 LoopShaper[4], MidiInput, MidiLearn, MidiMap,                 │
│                 MixBus, Recorder                                             │
└──────────────────────────────────────────────────────────────────────────────┘

 ENGINE LAYER
┌─────────────────────────┐     ┌─────────────────────────────────┐
│ MasterClock              │     │ SyncManager                      │
│─────────────────────────│     │─────────────────────────────────│
│ - _bpm: float            │     │ - _master_clock: MasterClock     │
│ - _running: bool         │     │ - _players: list[TrackPlayer]    │
│ - _tick_event: Event     │     │ - _ratios: list[float]           │
│ - _lock: Lock            │     │ - _drift_accum: list[float]      │
│ - _thread: Thread        │     │─────────────────────────────────│
│─────────────────────────│     │ + recalculate_ratios() -> None   │
│ + start() -> None        │     │ + advance_all(frames) -> None    │
│ + stop() -> None         │     │ + correct_drift(ch) -> int       │
│ + set_bpm(bpm) -> None   │     │ + get_ratio(ch) -> float         │
│ + get_bpm() -> float     │     └─────────────┬───────────────────┘
│ + tap_tempo() -> None    │                    │ drives
│ + register_callback(fn)  │                    ▼
└─────────┬───────────────┘     ┌─────────────────────────────────┐
          │ ticks                │ TrackPlayer                      │
          └────────────────────►│─────────────────────────────────│
                                │ - _audio_data: np.ndarray        │
                                │ - _stretched_data: np.ndarray    │
                                │ - _playhead: int (sample index)  │
                                │ - _track_bpm: float              │
                                │ - _playing: bool                 │
                                │ - _lock: Lock                    │
                                │─────────────────────────────────│
                                │ + load_file(path) -> None        │
                                │ + stretch(ratio) -> None         │
                                │ + advance(frames) -> np.ndarray  │
                                │ + set_playhead(sample) -> None   │
                                │ + get_playhead() -> int          │
                                │ + play() / pause() / stop()      │
                                │ + get_track_bpm() -> float       │
                                │ + set_track_bpm(bpm) -> None     │
                                └─────────────────────────────────┘

┌─────────────────────────┐
│ BpmDetector              │
│─────────────────────────│
│ (stateless utility)      │
│─────────────────────────│
│ + detect(audio, sr)      │
│   -> float               │
└─────────────────────────┘

┌─────────────────────────┐
│ MixBus                   │
│─────────────────────────│
│ - _master_volume: float  │
│ - _num_channels: int     │
│─────────────────────────│
│ + mix(buffers: list[     │
│     np.ndarray])         │
│   -> np.ndarray          │
│ + set_master_vol(v)      │
└─────────────────────────┘

 DSP LAYER
┌─────────────────────────────────────────────┐
│ ChannelDSP                                   │
│─────────────────────────────────────────────│
│ - _board: pedalboard.Pedalboard              │
│ - _eq: EQProcessor                           │
│ - _filter: FilterProcessor                   │
│ - _reverb: ReverbProcessor                   │
│ - _echo: EchoProcessor                       │
│ - _pitch: PitchProcessor                     │
│ - _volume: float                             │
│ - _pan: float                                │
│ - _params: dict[Param, float]                │
│ - _lock: Lock                                │
│─────────────────────────────────────────────│
│ + process(buffer: np.ndarray) -> np.ndarray  │
│ + set_param(param: Param, value: float)      │
│ + get_param(param: Param) -> float           │
│ + get_all_params() -> dict[Param, float]     │
│ + reset_to_defaults() -> None                │
└──────────────┬──────────────────────────────┘
               │ owns
    ┌──────────┼──────────┬──────────┬──────────┐
    ▼          ▼          ▼          ▼          ▼
┌────────┐┌────────┐┌─────────┐┌────────┐┌────────┐
│EQProc  ││FilterP ││ReverbP  ││EchoProc││PitchP  │
│--------││--------││---------││--------││--------│
│hi,mi,lo││cutoff  ││size     ││time    ││semi    │
│        ││reso    ││damp     ││feedbk  ││cents   │
│        ││type    ││mix      ││mix     ││        │
│--------││--------││---------││--------││--------│
│+apply()││+apply()││+apply() ││+apply()││+apply()│
│+update ││+update ││+update  ││+update ││+update │
│ _gains ││ _param ││ _params ││ _param ││ _param │
└────────┘└────────┘└─────────┘└────────┘└────────┘

 LOOP LAYER
┌───────────────────────────────────────┐   ┌────────────────────────────────────────────┐
│ LoopManager                            │   │ LoopShaper                                  │
│───────────────────────────────────────│   │────────────────────────────────────────────│
│ - _loop_in: int | None (sample)        │   │ - _state: LoopShaperState                   │
│ - _loop_out: int | None (sample)       │   │ - _mode: AutomationMode                     │
│ - _loop_active: bool                   │   │ - _lanes: dict[Param, list[tuple[float,     │
│ - _hot_cues: list[int | None] (4)      │   │                                   float]]]  │
│ - _lock: Lock                          │   │ - _touched_params: set[Param]                │
│───────────────────────────────────────│   │ - _loop_manager: LoopManager                 │
│ + set_loop_in(playhead) -> None        │   │ - _lock: Lock                                │
│ + set_loop_out(playhead) -> None       │   │────────────────────────────────────────────│
│ + toggle_loop() -> None                │   │ + arm_record() -> None                       │
│ + escape_loop() -> None                │   │ + start_record() -> None                     │
│ + is_loop_active() -> bool             │   │ + stop_record() -> None                      │
│ + get_loop_bounds()                    │   │ + arm_overdub() -> None                      │
│   -> tuple[int, int] | None            │   │ + start_overdub() -> None                    │
│ + wrap_playhead(ph) -> int             │   │ + record_point(param, norm_pos, val) -> None │
│ + set_hot_cue(idx, pos) -> None        │   │ + evaluate(norm_pos)                         │
│ + recall_hot_cue(idx) -> int | None    │   │   -> dict[Param, float]                      │
│ + get_loop_length_bars(bpm, sr)        │   │ + clear_lane(param) -> None                  │
│   -> float                             │   │ + clear_all() -> None                        │
└───────────────────────────────────────┘   │ + rescale(old_len, new_len) -> None           │
                                            │ + set_mode(mode) -> None                      │
                                            │ + get_lane(param)                              │
                                            │   -> list[tuple[float, float]]                 │
                                            │ + has_automation(param) -> bool                 │
                                            └────────────────────────────────────────────────┘

 MIDI LAYER
┌─────────────────────────┐  ┌──────────────────────────┐  ┌──────────────────────────────┐
│ MidiInput                │  │ MidiLearn                 │  │ MidiMap                       │
│─────────────────────────│  │──────────────────────────│  │──────────────────────────────│
│ - _port: rtmidi.MidiIn   │  │ - _active: bool           │  │ - _mappings: dict[            │
│ - _callback: Callable    │  │ - _pending_target:        │  │     MidiAddress,              │
│ - _channel_filter: int|  │  │     ControlTarget | None  │  │     ControlTarget]            │
│   None                   │  │ - _midi_map: MidiMap      │  │ - _reverse: dict[             │
│─────────────────────────│  │──────────────────────────│  │     ControlTarget,             │
│ + open(port_idx) -> None │  │ + enable() -> None        │  │     MidiAddress]              │
│ + close() -> None        │  │ + disable() -> None       │  │──────────────────────────────│
│ + list_ports()           │  │ + set_target(t) -> None   │  │ + add(addr, target) -> None   │
│   -> list[str]           │  │ + on_midi(msg) -> None    │  │ + remove(addr) -> None        │
│ + set_channel_filter(ch) │  │ + is_active() -> bool     │  │ + lookup(addr)                │
│ + _on_message(msg, data) │  └──────────────────────────┘  │   -> ControlTarget | None      │
└─────────────────────────┘                                 │ + check_conflicts()             │
                                                            │   -> list[str]                   │
                                                            │ + save(path) -> None             │
                                                            │ + load(path) -> None             │
                                                            └──────────────────────────────────┘

 SUPPORTING TYPES
┌─────────────────────────────┐  ┌──────────────────────────────────┐
│ MidiAddress (NamedTuple)     │  │ ControlTarget (NamedTuple)        │
│─────────────────────────────│  │──────────────────────────────────│
│ + channel: int               │  │ + channel_idx: int  (0-3 or -1   │
│ + msg_type: str  (cc/note)   │  │                      for global)  │
│ + number: int    (0-127)     │  │ + param: Param | str              │
└─────────────────────────────┘  │ + action: str  (set/toggle/       │
                                 │                 trigger)            │
                                 └──────────────────────────────────┘

 RECORDING LAYER
┌─────────────────────────────┐
│ Recorder                     │
│─────────────────────────────│
│ - _file: sf.SoundFile | None│
│ - _recording: bool           │
│ - _punch_in_armed: bool     │
│ - _start_time: float         │
│ - _lock: Lock                │
│─────────────────────────────│
│ + start(path) -> None        │
│ + stop() -> str (filepath)   │
│ + arm_punch_in() -> None     │
│ + write(buffer) -> None      │
│ + is_recording() -> bool     │
│ + elapsed() -> float         │
└─────────────────────────────┘

 UI LAYER (PyQt6)
┌──────────────────────────────────────────────────────────────────────────┐
│ MainWindow(QMainWindow)                                                   │
│──────────────────────────────────────────────────────────────────────────│
│ - _channel_strips: list[ChannelStrip] (4)                                 │
│ - _master_section: MasterSection                                          │
│──────────────────────────────────────────────────────────────────────────│
│ + setup_layout() -> None                                                  │
│ + connect_signals() -> None                                               │
│ + update_ui() -> None  (30fps timer)                                      │
└──────────────────┬───────────────────────────────────────────────────────┘
                   │ contains
      ┌────────────┼───────────────────────┐
      ▼                                    ▼
┌──────────────────────────┐  ┌─────────────────────────────┐
│ ChannelStrip(QWidget)     │  │ MasterSection(QWidget)       │
│──────────────────────────│  │─────────────────────────────│
│ - _waveform: WaveformW    │  │ - _bpm_knob: KnobWidget      │
│ - _knobs: dict[Param,     │  │ - _tap_tempo_btn: QPushBtn    │
│            KnobWidget]    │  │ - _master_vol: KnobWidget     │
│ - _loop_shaper_w:         │  │ - _record_btn: QPushButton    │
│     LoopShaperWidget      │  │ - _midi_learn_btn: QPushBtn   │
│ - _loop_btns              │  │ - _rec_timer: QLabel           │
│ - _hot_cue_btns (4)       │  └─────────────────────────────┘
│ - _load_btn: QPushButton  │
│ - _bpm_label: QLabel      │
│──────────────────────────│
│ + load_track() -> None    │
│ + update_waveform() -> None│
│ + on_knob_changed(p, v)   │
└──────────────────────────┘

┌──────────────────────────┐  ┌──────────────────────────────────┐
│ KnobWidget(QWidget)       │  │ WaveformWidget(QWidget)           │
│──────────────────────────│  │──────────────────────────────────│
│ - _value: float            │  │ - _audio_data: np.ndarray | None  │
│ - _min, _max: float        │  │ - _playhead: int                  │
│ - _label: str              │  │ - _loop_in, _loop_out: int | None │
│ - _midi_highlight: bool    │  │ - _beat_grid: list[int]           │
│──────────────────────────│  │──────────────────────────────────│
│ + setValue(v) -> None      │  │ + set_audio(data) -> None         │
│ + value() -> float         │  │ + set_playhead(pos) -> None       │
│ + paintEvent(e) -> None    │  │ + set_loop_region(in, out)        │
│ + mouseMoveEvent(e)        │  │ + set_beat_grid(beats)            │
│ + wheelEvent(e)            │  │ + paintEvent(e) -> None           │
└──────────────────────────┘  └──────────────────────────────────┘

┌────────────────────────────────────────────┐
│ LoopShaperWidget(QWidget)                   │
│────────────────────────────────────────────│
│ - _lanes: dict[Param, list[tuple]]          │
│ - _active_lanes: set[Param]                 │
│ - _playhead_pos: float (0.0-1.0)           │
│ - _record_btn, _overdub_btn: QPushButton    │
│ - _clear_lane_combo: QComboBox              │
│ - _clear_all_btn: QPushButton               │
│ - _mode_toggle: QPushButton                 │
│ - _collapsed: bool                          │
│────────────────────────────────────────────│
│ + update_lanes(data) -> None                │
│ + set_playhead(pos) -> None                 │
│ + paintEvent(e) -> None                     │
│ + toggle_collapse() -> None                 │
└────────────────────────────────────────────┘
```

---

## 7. Module-by-Module Specification

### 7.1 `main.py` — Application Entry Point

**Responsibility:** Wire all components together, start the Qt event loop.

**Step-by-step logic:**

1. Parse command-line arguments (optional: `--buffer-size`, `--sample-rate`).
2. Load `config/settings.json` if it exists; otherwise use defaults from `constants.py`.
3. Instantiate `MasterClock(bpm=DEFAULT_MASTER_BPM)`.
4. Instantiate `BpmDetector()` (stateless, shared).
5. For `i` in `range(4)`:
   - Instantiate `TrackPlayer(sample_rate, buffer_size)`.
   - Instantiate `ChannelDSP(sample_rate)`.
   - Instantiate `LoopManager()`.
   - Instantiate `LoopShaper(loop_manager)`.
6. Instantiate `SyncManager(master_clock, players=[...])`.
7. Instantiate `MixBus(num_channels=4)`.
8. Instantiate `Recorder(sample_rate)`.
9. Instantiate `MidiMap()`; call `midi_map.load("config/midi_map.json")`.
10. Instantiate `MidiInput()`; open first available MIDI port.
11. Instantiate `MidiLearn(midi_map)`.
12. Wire MIDI input callback: `midi_input.set_callback(midi_dispatch)`.
13. Define `midi_dispatch(msg)`:
    - If `midi_learn.is_active()`: forward to `midi_learn.on_midi(msg)`.
    - Else: lookup target via `midi_map.lookup(address)`; dispatch to corresponding parameter setter.
14. Instantiate `QApplication`.
15. Instantiate `MainWindow(...)` passing all engine/dsp/loop/midi references.
16. Open `sounddevice.OutputStream(callback=audio_callback)`.
17. Define `audio_callback(outdata, frames, time, status)`:
    - For each channel: `buf = track_player[i].advance(frames)`.
    - For each channel: apply loop shaper overrides via `loop_shaper[i].evaluate(norm_pos)`.
    - For each channel: `buf = channel_dsp[i].process(buf)`.
    - Mix: `mixed = mix_bus.mix(buffers)`.
    - If recording: `recorder.write(mixed)`.
    - Copy `mixed` into `outdata`.
18. Start master clock: `master_clock.start()`.
19. Run Qt event loop: `app.exec()`.
20. On exit: stop clock, close MIDI, close audio stream, save settings.

---

### 7.2 `engine/master_clock.py` — MasterClock

**Responsibility:** Generate timing ticks at the master BPM. All channel playheads advance from this single source.

**Key design:**

- Runs in a dedicated `threading.Thread` with daemon mode.
- Does **not** drive audio directly — the `sounddevice` callback runs on its own real-time thread. The master clock maintains the BPM value and a beat counter. The audio callback reads `master_clock.bpm` to compute how many samples to advance.
- Tick interval: `60.0 / bpm` seconds per beat.
- Uses `time.perf_counter()` for sub-millisecond precision.
- Maintains a beat counter (`_beat_count: int`), incrementing on each tick.
- Registered callbacks are invoked on each tick (for UI beat flash, quantize triggers, etc.).
- Tap tempo: stores last 4 tap timestamps, computes mean interval, sets BPM.

**Thread safety:**

- `_bpm` protected by `threading.Lock`.
- Callbacks list protected by `threading.Lock`.
- Beat counter read via atomic-safe property.

---

### 7.3 `engine/track_player.py` — TrackPlayer

**Responsibility:** Load an audio file, hold the raw and stretched audio buffers, maintain the playhead, and return audio chunks on request.

**Step-by-step logic for `load_file(path)`:**

1. Read audio via `soundfile.read(path)` → `(data: np.ndarray, sr: int)`.
2. If stereo, keep as-is (shape: `[N, 2]`). If mono, duplicate to stereo.
3. If `sr != SAMPLE_RATE`, resample using `librosa.resample`.
4. Store `_audio_data = data`.
5. Detect BPM: `_track_bpm = BpmDetector.detect(data, sr)`.
6. Trigger initial stretch: `self.stretch(master_bpm / _track_bpm)`.

**Step-by-step logic for `stretch(ratio)`:**

1. Call `pyrubberband.time_stretch(audio_data, sr, ratio)`.
2. Store result as `_stretched_data`.
3. Recalculate beat grid positions in stretched coordinates.
4. If loop is active, recalculate loop boundaries: `new_loop_in = int(old_loop_in * ratio / old_ratio)`.

**Step-by-step logic for `advance(frames)`:**

1. Acquire lock.
2. If not playing, return silence buffer `np.zeros((frames, 2))`.
3. Read `frames` samples from `_stretched_data` starting at `_playhead`.
4. If loop active and playhead would cross `loop_out`:
   - Split read: samples up to `loop_out`, then wrap to `loop_in`.
   - Crossfade at boundary (2ms, ~88 samples) to prevent click.
5. Advance `_playhead += frames` (with wrap).
6. Release lock.
7. Return the buffer.

---

### 7.4 `engine/bpm_detector.py` — BpmDetector

**Responsibility:** Detect BPM of a loaded audio file. Provide fallback on failure.

**Step-by-step logic:**

1. Convert stereo to mono via mean.
2. Call `librosa.beat.beat_track(y=mono, sr=sr)`.
3. If result is `0.0` or outside `[40, 300]` range → return `None` (signals manual entry needed).
4. Otherwise return detected BPM as `float`.

---

### 7.5 `engine/sync_manager.py` — SyncManager

**Responsibility:** Calculate stretch ratios, apply drift correction for non-integer ratios.

**Step-by-step logic for `recalculate_ratios()`:**

1. For each channel `i`:
   - `_ratios[i] = master_clock.get_bpm() / players[i].get_track_bpm()`.
2. Trigger re-stretch on each player: `players[i].stretch(_ratios[i])`.

**Step-by-step logic for drift correction:**

1. Maintain `_drift_accumulator[i]: float` per channel.
2. On each audio callback:
   - Ideal position (fractional samples): `ideal = elapsed_beats * samples_per_beat`.
   - Actual position: `actual = player[i].get_playhead()`.
   - Drift: `delta = ideal - actual`.
   - If `abs(delta) > threshold` (e.g., 0.5 samples):
     - Nudge playhead: `player[i].set_playhead(actual + round(delta))`.
     - Reset accumulator.
3. For integer ratios (1:1, 2:1), drift is zero; skip correction.
4. For non-integer ratios (4:3, 3:5), run correction every N beats (configurable, default: every 4 beats).

---

### 7.6 `engine/mix_bus.py` — MixBus

**Responsibility:** Sum all 4 channel outputs, apply master volume.

**Step-by-step logic for `mix(buffers)`:**

1. Stack all buffers: `stacked = np.stack(buffers)` → shape `(4, frames, 2)`.
2. Sum along axis 0: `mixed = stacked.sum(axis=0)`.
3. Apply master volume: `mixed *= _master_volume`.
4. Clip to `[-1.0, 1.0]` to prevent clipping distortion.
5. Return `mixed`.

---

### 7.7 `dsp/channel_dsp.py` — ChannelDSP

**Responsibility:** Orchestrate the pedalboard FX chain for one channel. Apply all effects in order.

**Processing chain order:**

1. EQ (3-band parametric)
2. Filter (LP or HP)
3. Reverb
4. Echo / Delay
5. Pitch Shift
6. Volume + Pan

**Step-by-step logic for `process(buffer)`:**

1. Acquire lock.
2. Update each sub-processor with current param values (from `_params` dict, possibly overridden by Loop Shaper).
3. Run buffer through `pedalboard.Pedalboard([eq, filter, reverb, echo, pitch])`.
4. Apply volume: `buffer *= _params[Param.VOLUME]`.
5. Apply pan (constant-power pan law):
   - `left_gain = cos(pan * pi/4 + pi/4)`
   - `right_gain = sin(pan * pi/4 + pi/4)`
   - `buffer[:, 0] *= left_gain`
   - `buffer[:, 1] *= right_gain`
6. Release lock.
7. Return processed buffer.

**Rationale for chain order:** EQ first removes unwanted frequencies before they hit time-domain effects (reverb, echo). Pitch last to avoid artifacts from shifting already-effected audio. Volume/pan at the very end for consistent gain staging.

---

### 7.8 `dsp/eq.py`, `filter.py`, `reverb.py`, `echo.py`, `pitch.py`

Each is a thin wrapper around the corresponding `pedalboard` plugin:

| File | pedalboard Plugin | Parameters |
|---|---|---|
| `eq.py` | `pedalboard.LowShelfFilter` + `pedalboard.PeakFilter` + `pedalboard.HighShelfFilter` | `EQ_LOW`, `EQ_MID`, `EQ_HIGH` |
| `filter.py` | `pedalboard.LowpassFilter` / `pedalboard.HighpassFilter` | `FILTER_CUTOFF`, `FILTER_RESONANCE`, `FILTER_TYPE` |
| `reverb.py` | `pedalboard.Reverb` | `REVERB_SIZE`, `REVERB_DAMP`, `REVERB_MIX` |
| `echo.py` | `pedalboard.Delay` | `ECHO_TIME`, `ECHO_FEEDBACK`, `ECHO_MIX` |
| `pitch.py` | `pedalboard.PitchShift` | `PITCH_SEMITONE`, `PITCH_CENTS` |

Each wrapper class exposes:

- `update(params: dict[Param, float]) -> None`: Update plugin parameters.
- `get_plugin() -> pedalboard.Plugin`: Return the configured plugin instance for chain assembly.

---

### 7.9 `loop/loop_manager.py` — LoopManager

**Responsibility:** Manage loop in/out points, loop activation, hot cues. All positions stored in sample indices of the **stretched** audio.

**Step-by-step logic for `set_loop_in(playhead)`:**

1. Quantize `playhead` to nearest beat on the stretched beat grid.
2. Store `_loop_in = quantized_pos`.
3. If `_loop_out is not None` and `_loop_out <= _loop_in`:
   - Swap: `_loop_in, _loop_out = _loop_out, _loop_in`.

**Step-by-step logic for `set_loop_out(playhead)`:**

1. If `_loop_in is None`: no-op (cannot set out without in).
2. Quantize `playhead` to nearest beat.
3. If `_loop_out is None` (first press):
   - Store `_loop_out = quantized_pos`.
   - If `_loop_out <= _loop_in`: swap.
   - Activate loop: `_loop_active = True`.
4. If `_loop_out is not None` and `_loop_active` (second press):
   - Escape loop: `_loop_active = False`.

**Step-by-step logic for `wrap_playhead(playhead)`:**

1. If not `_loop_active`: return `playhead` unchanged.
2. If `playhead >= _loop_out`: return `_loop_in + (playhead - _loop_out)`.
3. Otherwise: return `playhead`.

**Hot cues:**

- `set_hot_cue(idx, position)`: Store absolute stretched-time sample position.
- `recall_hot_cue(idx)`: Return stored position (player jumps to it).
- Hot cues are **not** loop-relative — they are always in absolute track time.

---

### 7.10 `loop/loop_shaper.py` — LoopShaper

**Responsibility:** The centrepiece. Record, overdub, and replay parameter automation gestures locked to the loop timeline.

**Data structure:**

```python
_lanes: dict[Param, list[tuple[float, float]]]
# Key: parameter enum
# Value: sorted list of (normalized_loop_position, parameter_value)
# normalized_loop_position: 0.0 = loop start, 1.0 = loop end
```

**Step-by-step logic for `arm_record()`:**

1. Set `_state = LoopShaperState.ARMED`.
2. On next loop wrap (detected by `loop_manager`), transition to `RECORDING`.
3. Clear all lanes (fresh recording).

**Step-by-step logic for `start_record()`:**

1. Set `_state = LoopShaperState.RECORDING`.
2. Clear `_touched_params`.
3. Initialize all lanes as empty lists.

**Step-by-step logic for `record_point(param, norm_pos, value)`:**

1. Guard: if `_state` not in `{RECORDING, OVERDUBBING}`, return.
2. Append `(norm_pos, value)` to `_lanes[param]`.
3. Add `param` to `_touched_params`.

**Step-by-step logic for `stop_record()`:**

1. For each lane in `_lanes`: sort by `norm_pos`.
2. Set `_state = LoopShaperState.PLAYING`.

**Step-by-step logic for `arm_overdub()`:**

1. Set `_state = LoopShaperState.ARMED` (overdub variant).
2. On next loop wrap, transition to `OVERDUBBING`.
3. Do **not** clear existing lanes.

**Step-by-step logic for `start_overdub()`:**

1. Set `_state = LoopShaperState.OVERDUBBING`.
2. Clear `_touched_params` (tracks which params are being overdubbed this pass).

During overdub recording:

1. Only lanes for touched params are modified.
2. New points in the touched region replace existing points within a tolerance window (`±0.005` normalized).
3. Existing points outside the touched region are preserved.
4. On stop: merge and re-sort each touched lane.

**Step-by-step logic for `evaluate(norm_pos)`:**

1. Guard: if `_state` not in `{PLAYING, RECORDING, OVERDUBBING}`, return empty dict.
2. For each `param` in `_lanes`:
   - If lane is empty, skip.
   - Binary search for the two bracketing points around `norm_pos`.
   - Linear interpolate between them to get `interp_value`.
   - If `_mode == ABSOLUTE`: `result[param] = interp_value`.
   - If `_mode == ADDITIVE`: `result[param] = interp_value` (caller adds to manual).
3. Return `result: dict[Param, float]`.

**Step-by-step logic for `rescale(old_length, new_length)`:**

1. `scale_factor = old_length / new_length`.
2. For each lane:
   - Multiply each `norm_pos` by `scale_factor`.
   - Clamp all positions to `[0.0, 1.0]`.
   - Discard points outside `[0.0, 1.0]` (they fall outside the new shorter loop).
   - If loop is expanding (`new_length > old_length`), existing points compress toward 0.0; the tail region (`old_length/new_length` to `1.0`) has no automation (flat at last value).
3. Re-sort all lanes.

**Step-by-step logic for `clear_lane(param)`:**

1. Set `_lanes[param] = []`.

**Step-by-step logic for `clear_all()`:**

1. For every `param` in `Param`: `_lanes[param] = []`.
2. Set `_state = LoopShaperState.IDLE`.

---

### 7.11 `midi/midi_input.py` — MidiInput

**Responsibility:** Listen for MIDI messages from any connected controller via `python-rtmidi`.

**Step-by-step logic:**

1. On `open(port_idx)`: open the specified MIDI input port.
2. Register internal callback `_on_message(midi_msg, timestamp)`.
3. Parse message bytes:
   - `status = msg[0] & 0xF0`
   - `channel = msg[0] & 0x0F`
   - If `status == 0xB0` (CC): extract `(channel, cc_number=msg[1], value=msg[2])`.
   - If `status == 0x90` (Note On): extract `(channel, note=msg[1], velocity=msg[2])`.
   - If `status == 0x80` (Note Off): extract `(channel, note=msg[1], velocity=0)`.
4. If `_channel_filter is not None` and `channel != _channel_filter`: discard.
5. Construct `MidiAddress(channel, msg_type, number)`.
6. Invoke registered callback with `(MidiAddress, value)`.

---

### 7.12 `midi/midi_learn.py` — MidiLearn

**Responsibility:** Capture the next incoming MIDI message and assign it to a selected UI control.

**Step-by-step logic:**

1. User clicks a knob in the UI → `set_target(ControlTarget(...))`.
2. `_active = True`.
3. Next MIDI message arrives via `on_midi(address, value)`:
   - Call `_midi_map.add(address, _pending_target)`.
   - Check conflicts: `conflicts = _midi_map.check_conflicts()`.
   - If conflicts: emit warning signal to UI.
   - `_active = False`.
   - `_pending_target = None`.

---

### 7.13 `midi/midi_map.py` — MidiMap

**Responsibility:** Store, load, save, and query MIDI-to-control mappings.

**Data structure:**

```python
_mappings: dict[MidiAddress, ControlTarget]
# MidiAddress = (channel: int, msg_type: str, number: int)
# ControlTarget = (channel_idx: int, param: Param|str, action: str)
```

**Step-by-step logic for `check_conflicts()`:**

1. Build reverse map: `ControlTarget → list[MidiAddress]`.
2. Also check forward: `MidiAddress → list[ControlTarget]`.
3. A conflict occurs when one `MidiAddress` maps to multiple `ControlTarget`s.
4. Return list of human-readable conflict descriptions.

**Step-by-step logic for `save(path)`:**

1. Serialize `_mappings` to JSON:
   - Keys: `f"{addr.channel}:{addr.msg_type}:{addr.number}"`.
   - Values: `{"channel_idx": t.channel_idx, "param": t.param.name, "action": t.action}`.
2. Write to `path`.

**Step-by-step logic for `load(path)`:**

1. Read JSON from `path`.
2. Deserialize into `_mappings` dict.
3. Run `check_conflicts()` and log warnings.

---

### 7.14 `recording/recorder.py` — Recorder

**Responsibility:** Capture the mixed output to a WAV file.

**Step-by-step logic for `start(path)`:**

1. Open `soundfile.SoundFile(path, mode='w', samplerate=SAMPLE_RATE, channels=2, format='WAV', subtype='FLOAT')`.
2. Set `_recording = True`.
3. Record `_start_time = time.time()`.

**Step-by-step logic for `arm_punch_in()`:**

1. Set `_punch_in_armed = True`.
2. On next loop start event (from any channel's loop manager), call `start(...)`.

**Step-by-step logic for `write(buffer)`:**

1. If not `_recording`: return.
2. `_file.write(buffer)`.

**Step-by-step logic for `stop()`:**

1. Set `_recording = False`.
2. Close `_file`.
3. Return filepath string.

---

### 7.15 UI Modules

#### `ui/main_window.py`

- `QMainWindow` subclass.
- Horizontal layout: 4 × `ChannelStrip` side by side.
- Top bar: `MasterSection`.
- 30fps `QTimer` calls `update_ui()` which reads engine state and pushes to widgets.
- File dialogs for track loading.
- All signal/slot connections wired in `connect_signals()`.

#### `ui/channel_strip.py`

- Composite `QWidget` containing all per-channel controls.
- Sections stacked vertically: waveform, transport, EQ, filter, reverb, echo, pitch, loop, loop shaper.
- Each knob emits `valueChanged(Param, float)` signal.
- Color-coded border using `CHANNEL_COLORS[i]`.

#### `ui/knob_widget.py`

- Custom `QWidget` rendering a circular knob.
- Mouse drag (vertical) and mouse wheel change value.
- `paintEvent` draws arc, pointer, label, and value.
- `_midi_highlight` flag: when MIDI input received, briefly glow the knob outline.
- Emits `valueChanged(float)` on interaction.

#### `ui/waveform_widget.py`

- Custom `QWidget` rendering a scrolling waveform.
- Downsamples audio for display (peak envelope).
- Overlays: beat grid (vertical lines), loop region (shaded), playhead (bright line).
- Scrolls to keep playhead centered.

#### `ui/loop_shaper_widget.py`

- Collapsible panel per channel.
- 16 mini lanes, each a thin horizontal strip showing the automation curve.
- Active (non-empty) lanes highlighted with channel color.
- Empty lanes shown as flat gray line.
- Vertical playhead sweeps across all lanes in sync.
- Buttons: Record, Overdub, Clear Lane (dropdown), Clear All, Additive/Absolute toggle.

---

## 8. Edge Cases & Mitigation

| # | Edge Case | Mitigation Strategy |
|---|---|---|
| 1 | **Non-integer BPM ratio drift** (e.g., 120/90 = 1.333…) | `SyncManager` accumulates fractional-sample error and applies integer correction every N beats (default 4). Threshold: 0.5 samples. |
| 2 | **Loop Shaper automation during loop length change** | `LoopShaper.rescale()` proportionally scales all `norm_pos` values. Points falling outside `[0.0, 1.0]` after a shrink are discarded. |
| 3 | **Loop Shaper overdub merge** | Only touched params are modified. Within a tolerance window (`±0.005` norm), new points replace old. Untouched regions preserved. Re-sort after merge. |
| 4 | **Additive mode overflow** | After adding automation offset to manual value, clamp result to `[ParamRange.min_val, ParamRange.max_val]`. |
| 5 | **MIDI CC conflict** | `MidiMap.check_conflicts()` detects duplicate mappings. UI shows warning dialog. User can override or reassign. |
| 6 | **Audio buffer underrun** | `sounddevice` status callback detects `status.output_underflow`. Log warning, emit UI signal (yellow flash). User can increase buffer size in settings. |
| 7 | **Track not loaded** | `TrackPlayer.advance()` returns silence. All DSP/loop/shaper calls are guarded with `if _audio_data is None: return`. |
| 8 | **BPM detection failure** | `BpmDetector.detect()` returns `None`. UI prompts manual BPM entry dialog. Channel is not stretched until BPM is set. |
| 9 | **Loop Out before Loop In** | `LoopManager.set_loop_out()` checks order; if `out <= in`, auto-swaps. |
| 10 | **Recording punch-in timing** | `Recorder.arm_punch_in()` sets a flag. Actual recording starts on the next loop-start event from any active loop, ensuring the capture begins on a clean boundary. |
| 11 | **Hot cue inside active loop** | Hot cues are stored in absolute stretched-time, not loop-relative. Recalling a hot cue that is outside the current loop region does not break the loop — the loop wraps normally on the next boundary. |
| 12 | **MIDI Learn collision** | If the incoming CC/Note is already mapped to another control, show a confirmation dialog: "This MIDI message is already mapped to [X]. Replace?" |
| 13 | **Loop Shaper Record starts mid-loop** | `arm_record()` sets state to `ARMED`. Actual recording begins at the next loop wrap, ensuring a full clean pass. |
| 14 | **Pitch shift + time-stretch interaction** | `pyrubberband` handles global time-stretch (preserving pitch). `pedalboard.PitchShift` handles creative pitch offset. These are independent operations on separate stages of the pipeline — no doubling. |
| 15 | **Loop Shaper Additive mode overflow** | Clamping applied after additive calculation: `value = max(param_range.min_val, min(param_range.max_val, manual + automation))`. |
| 16 | **Master BPM change while playing** | `SyncManager.recalculate_ratios()` is called. All channels re-stretch. Loop boundaries are recalculated in new stretched coordinates. Loop Shaper automation is unaffected (normalized positions remain valid). |
| 17 | **File format unsupported** | `soundfile.read()` raises `RuntimeError`. Catch, show error dialog, channel remains empty. |
| 18 | **Zero-length loop** | Guard: if `loop_out - loop_in < minimum_loop_samples` (e.g., 1 beat), reject and show warning. |
| 19 | **Rapid knob movement during Loop Shaper recording** | Rate-limit recorded points to 1 per millisecond per param to prevent memory bloat. Downsample on stop-record using Douglas-Peucker or similar. |
| 20 | **Sounddevice callback latency** | Audio callback must complete within `buffer_size / sample_rate` seconds. All heavy computation (stretching) is pre-computed. Callback only reads from pre-stretched buffers. |

---

## 9. Threading Model & Concurrency

### 9.1 Thread Map

```
┌────────────────────────────────────────────────────────────┐
│  THREAD 1: Qt Main / UI Thread                              │
│  - All QWidget rendering                                    │
│  - User input (mouse, keyboard)                             │
│  - 30fps QTimer for UI updates (reads shared state)         │
│  - NEVER performs audio processing                          │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│  THREAD 2: sounddevice Audio Callback (real-time)           │
│  - Invoked by OS audio subsystem                            │
│  - Calls track_player[i].advance(frames)                    │
│  - Calls loop_shaper[i].evaluate(norm_pos)                  │
│  - Calls channel_dsp[i].process(buffer)                     │
│  - Calls mix_bus.mix(buffers)                               │
│  - Calls recorder.write(mixed)                              │
│  - MUST complete within buffer_size/sample_rate seconds      │
│  - NO memory allocation, NO I/O, NO Qt calls                │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│  THREAD 3: Master Clock Thread (daemon)                     │
│  - Sleeps between beats                                     │
│  - Increments beat counter                                  │
│  - Fires registered callbacks (queued to UI thread)         │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│  THREAD 4: MIDI Input Thread (managed by python-rtmidi)     │
│  - Callback invoked on incoming MIDI messages               │
│  - Parses and dispatches to parameter setters               │
│  - Must be fast — no blocking                               │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│  THREAD 5+: File Loading Workers (QThreadPool)              │
│  - Spawned on track load                                    │
│  - Reads file, detects BPM, performs initial stretch         │
│  - Emits signal to UI on completion                         │
│  - One-shot, terminates after load complete                  │
└────────────────────────────────────────────────────────────┘
```

### 9.2 Lock Strategy

| Resource | Lock | Held By | Accessed By |
|---|---|---|---|
| `TrackPlayer._stretched_data` | `TrackPlayer._lock` | File loader (write), Audio callback (read) | Audio callback reads chunks |
| `TrackPlayer._playhead` | `TrackPlayer._lock` | Audio callback (write), UI (read) | UI reads for waveform position |
| `ChannelDSP._params` | `ChannelDSP._lock` | MIDI thread (write), UI thread (write), Audio callback (read) | Audio callback reads params |
| `LoopShaper._lanes` | `LoopShaper._lock` | MIDI/UI thread (write during record), Audio callback (read) | Audio callback evaluates automation |
| `LoopManager._loop_in/out` | `LoopManager._lock` | UI/MIDI thread (write), Audio callback (read) | Audio callback checks boundaries |
| `MasterClock._bpm` | `MasterClock._lock` | UI/MIDI thread (write), Audio callback (read), Clock thread (read) | Multiple readers |
| `Recorder._file` | `Recorder._lock` | Audio callback (write), UI thread (start/stop) | Concurrent access on start/stop |
| `MidiMap._mappings` | `MidiMap._lock` | MIDI learn (write), MIDI dispatch (read) | Concurrent read/write |

### 9.3 Lock Ordering Convention

To prevent deadlocks, locks are always acquired in this order when multiple are needed:

```
MasterClock._lock  →  SyncManager._lock  →  TrackPlayer._lock  →
LoopManager._lock  →  LoopShaper._lock  →  ChannelDSP._lock  →
Recorder._lock
```

No code path ever acquires a lock that appears earlier in this sequence while holding one that appears later.

---

## 10. Phased Implementation Plan

### Phase 1: Master Clock Engine

**Goal:** A standalone tick system with sub-millisecond precision.

| Task | Description | Estimated Effort |
|---|---|---|
| 1.1 | Create `constants.py` with all named constants | 1 hr |
| 1.2 | Implement `MasterClock` class with dedicated thread | 4 hrs |
| 1.3 | Implement `time.perf_counter()` based high-resolution timing loop | 2 hrs |
| 1.4 | Implement BPM setter with lock, range validation (`60–300`) | 1 hr |
| 1.5 | Implement tap tempo (rolling window of 4 taps, mean interval → BPM) | 2 hrs |
| 1.6 | Implement callback registration and invocation on each tick | 1 hr |
| 1.7 | Write unit tests: timing accuracy (assert tick intervals within ±0.5ms), BPM range, tap tempo | 3 hrs |
| 1.8 | Verify thread-safe start/stop/restart without deadlocks | 2 hrs |

**Deliverable:** `engine/master_clock.py` + `constants.py` + tests. Clock runs standalone and prints beat timestamps.

---

### Phase 2: Single Track Playback

**Goal:** Load one audio file, detect BPM, time-stretch, play back via `sounddevice`.

| Task | Description | Estimated Effort |
|---|---|---|
| 2.1 | Implement `BpmDetector.detect()` using `librosa.beat.beat_track()` | 2 hrs |
| 2.2 | Handle detection failure (return `None`, test with silence and noise files) | 1 hr |
| 2.3 | Implement `TrackPlayer.load_file()`: read via `soundfile`, normalize to float32 stereo | 3 hrs |
| 2.4 | Implement `TrackPlayer.stretch(ratio)` via `pyrubberband.time_stretch()` | 3 hrs |
| 2.5 | Implement `TrackPlayer.advance(frames)`: return audio chunk, advance playhead | 2 hrs |
| 2.6 | Set up `sounddevice.OutputStream` with callback that calls `advance()` | 2 hrs |
| 2.7 | Implement play/pause/stop state machine | 1 hr |
| 2.8 | Test: load WAV, detect BPM, stretch to master BPM, play back — verify pitch preserved | 3 hrs |
| 2.9 | Test with MP3, FLAC, AIFF files | 1 hr |

**Deliverable:** Single-track playback with BPM detection and time-stretching. CLI-based test harness.

---

### Phase 3: 4-Track Sync

**Goal:** All 4 channels driven by the master clock with drift correction.

| Task | Description | Estimated Effort |
|---|---|---|
| 3.1 | Implement `SyncManager` with ratio calculation for all 4 channels | 3 hrs |
| 3.2 | Implement drift accumulator and correction logic | 4 hrs |
| 3.3 | Implement `MixBus.mix()`: sum 4 channels, apply master volume, clip | 2 hrs |
| 3.4 | Modify `sounddevice` callback to process all 4 channels per tick | 2 hrs |
| 3.5 | Test: 4 tracks at different BPMs (120, 90, 140, 60), verify sync after 5 minutes | 4 hrs |
| 3.6 | Test non-integer ratios: verify drift stays under 1ms after extended playback | 3 hrs |
| 3.7 | Test master BPM change while playing: verify all channels re-sync within 1 beat | 2 hrs |

**Deliverable:** 4-channel playback locked to master clock. CLI test harness plays 4 simultaneous files.

---

### Phase 4: DSP / Effects Layer

**Goal:** Full pedalboard FX chain per channel with all 16 parameters.

| Task | Description | Estimated Effort |
|---|---|---|
| 4.1 | Implement `EQProcessor` wrapping pedalboard shelf/peak filters | 3 hrs |
| 4.2 | Implement `FilterProcessor` wrapping LP/HP with cutoff + resonance | 2 hrs |
| 4.3 | Implement `ReverbProcessor` wrapping pedalboard.Reverb | 2 hrs |
| 4.4 | Implement `EchoProcessor` wrapping pedalboard.Delay | 2 hrs |
| 4.5 | Implement `PitchProcessor` wrapping pedalboard.PitchShift | 2 hrs |
| 4.6 | Implement `ChannelDSP` orchestrator: chain assembly, param dispatch, volume/pan | 4 hrs |
| 4.7 | Implement constant-power pan law | 1 hr |
| 4.8 | Integrate DSP into audio callback (after `advance()`, before `mix()`) | 2 hrs |
| 4.9 | Test: sweep each parameter end-to-end, verify no clicks/pops | 4 hrs |
| 4.10 | Test: all effects simultaneously on all 4 channels — verify latency budget | 3 hrs |

**Deliverable:** Full DSP chain per channel. Parameters controllable via code. No UI yet.

---

### Phase 5: Loop System

**Goal:** Rekordbox-style loop in/out, seamless looping, hot cues.

| Task | Description | Estimated Effort |
|---|---|---|
| 5.1 | Implement `LoopManager`: `set_loop_in()`, `set_loop_out()`, `toggle_loop()`, `escape_loop()` | 4 hrs |
| 5.2 | Implement beat-grid quantization for loop points | 2 hrs |
| 5.3 | Implement `wrap_playhead()` with seamless crossfade (2ms at boundary) | 3 hrs |
| 5.4 | Implement loop-length display calculation (bars based on BPM and sample rate) | 1 hr |
| 5.5 | Implement hot cue set/recall (4 per channel, absolute time) | 2 hrs |
| 5.6 | Handle edge: Loop Out before Loop In (auto-swap) | 1 hr |
| 5.7 | Handle edge: zero-length loop (minimum 1 beat) | 1 hr |
| 5.8 | Integrate loop logic into `TrackPlayer.advance()` | 2 hrs |
| 5.9 | Test: loop engage/escape, verify zero-gap seamless playback | 3 hrs |
| 5.10 | Test: hot cue recall inside and outside active loop | 2 hrs |

**Deliverable:** Per-channel loop system. CLI test: engage loop, escape, recall hot cues.

---

### Phase 6: Loop Shaper

**Goal:** Gesture recording, all 16 parameter lanes, overdub, rescale, additive/absolute.

| Task | Description | Estimated Effort |
|---|---|---|
| 6.1 | Implement `LoopShaper` state machine (IDLE → ARMED → RECORDING → PLAYING) | 3 hrs |
| 6.2 | Implement `record_point()`: append `(norm_pos, value)` to lane | 2 hrs |
| 6.3 | Implement `evaluate()`: binary search + linear interpolation across all 16 lanes | 4 hrs |
| 6.4 | Implement `stop_record()`: sort lanes, transition to PLAYING | 1 hr |
| 6.5 | Implement overdub: ARMED → OVERDUBBING, merge logic with tolerance window | 4 hrs |
| 6.6 | Implement `clear_lane()` and `clear_all()` | 1 hr |
| 6.7 | Implement `rescale()` for loop-length changes | 3 hrs |
| 6.8 | Implement additive vs absolute mode with clamping | 2 hrs |
| 6.9 | Integrate LoopShaper into audio callback: evaluate → override DSP params | 3 hrs |
| 6.10 | Implement record-start-at-next-loop-wrap (ARMED state handling) | 2 hrs |
| 6.11 | Implement rate-limiting of recorded points (1 per ms per param) | 1 hr |
| 6.12 | Test: record EQ sweep, verify playback matches original gesture | 3 hrs |
| 6.13 | Test: overdub new param without erasing existing | 3 hrs |
| 6.14 | Test: rescale from 8-bar to 4-bar loop, verify proportional compression | 2 hrs |
| 6.15 | Test: additive mode overflow clamping | 1 hr |

**Deliverable:** Full Loop Shaper engine. Testable via programmatic knob moves + verification.

---

### Phase 7: UI (PyQt6)

**Goal:** Complete visual interface with all controls.

| Task | Description | Estimated Effort |
|---|---|---|
| 7.1 | Implement `KnobWidget`: circular knob, mouse drag, wheel, paint, value label | 6 hrs |
| 7.2 | Implement `WaveformWidget`: peak envelope, scrolling, beat grid, loop region, playhead | 8 hrs |
| 7.3 | Implement `LoopShaperWidget`: 16 mini lanes, curve rendering, playhead, collapse/expand | 8 hrs |
| 7.4 | Implement `ChannelStrip`: composite widget, all sections laid out vertically | 6 hrs |
| 7.5 | Implement `MasterSection`: BPM knob, tap tempo, master volume, record, MIDI learn | 4 hrs |
| 7.6 | Implement `MainWindow`: 4 channel strips + master section, file dialogs | 4 hrs |
| 7.7 | Wire all signals: knob → param setter, button → loop/shaper action | 4 hrs |
| 7.8 | Implement 30fps UI update timer: read engine state → update widgets | 3 hrs |
| 7.9 | Implement channel color coding (borders, highlights) | 2 hrs |
| 7.10 | Implement MIDI highlight on knobs (brief glow on MIDI input) | 2 hrs |
| 7.11 | Visual polish: consistent sizing, font choices, dark theme | 4 hrs |

**Deliverable:** Full UI. All controls functional. Waveforms scroll, Loop Shaper lanes display.

---

### Phase 8: MIDI

**Goal:** Full MIDI input, learn mode, JSON mapping, DIY controller support.

| Task | Description | Estimated Effort |
|---|---|---|
| 8.1 | Implement `MidiInput`: open port, parse CC/NoteOn/NoteOff, channel filter | 3 hrs |
| 8.2 | Implement `MidiMap`: add/remove/lookup mappings, JSON save/load | 4 hrs |
| 8.3 | Implement `MidiMap.check_conflicts()` and conflict resolution UI | 3 hrs |
| 8.4 | Implement `MidiLearn`: click control → move knob → assign mapping | 4 hrs |
| 8.5 | Implement CC value scaling: `0–127` → `param_range.min_val..max_val` | 1 hr |
| 8.6 | Implement momentary button support (Note On = trigger, Note Off = release) | 2 hrs |
| 8.7 | Wire MIDI dispatch: incoming message → lookup → set_param / trigger action | 3 hrs |
| 8.8 | Implement MIDI learn toggle in UI (MasterSection) | 1 hr |
| 8.9 | Test with commercial controller (any USB MIDI) | 3 hrs |
| 8.10 | Test with DIY Arduino MIDI controller (raw CC messages) | 3 hrs |
| 8.11 | Test: save mapping, restart app, load mapping, verify all assignments | 2 hrs |

**Deliverable:** Full MIDI support. Learn mode, JSON persistence, DIY controller tested.

---

### Phase 9: Recording

**Goal:** Capture mixed output, punch-in, WAV export.

| Task | Description | Estimated Effort |
|---|---|---|
| 9.1 | Implement `Recorder.start()`: open WAV via `soundfile`, begin capture | 2 hrs |
| 9.2 | Implement `Recorder.write()`: called from audio callback, writes buffer | 1 hr |
| 9.3 | Implement `Recorder.stop()`: close file, return path | 1 hr |
| 9.4 | Implement `arm_punch_in()`: wait for next loop start, then start capture | 3 hrs |
| 9.5 | Implement recording timer display in MasterSection | 1 hr |
| 9.6 | Implement timestamp-based filename: `remix_YYYYMMDD_HHMMSS.wav` | 1 hr |
| 9.7 | Wire Record Start/Stop button in UI | 1 hr |
| 9.8 | Test: record 4-channel mix, open WAV in external DAW, verify content | 2 hrs |
| 9.9 | Test: punch-in at loop boundary, verify clean edit point | 2 hrs |

**Deliverable:** Recording module. WAV files capture full mix with punch-in support.

---

### Phase 10: Integration & Performance

**Goal:** Full system stress test, latency profiling, buffer tuning.

| Task | Description | Estimated Effort |
|---|---|---|
| 10.1 | End-to-end integration test: load 4 tracks, engage loops, record Loop Shaper gestures on all channels, play back, record output | 4 hrs |
| 10.2 | Latency profiling: measure audio callback duration with `time.perf_counter()` | 3 hrs |
| 10.3 | Optimize hot paths: pre-allocate numpy buffers, minimize allocations in callback | 4 hrs |
| 10.4 | Buffer size tuning: test at 128, 256, 512, 1024 samples; document min stable size | 3 hrs |
| 10.5 | Stress test: all 4 channels playing, all effects active, Loop Shaper on all channels, MIDI input, recording — verify no underruns for 30 minutes | 4 hrs |
| 10.6 | Memory profiling: verify no leaks over extended sessions (1 hour) | 3 hrs |
| 10.7 | UI responsiveness: verify no frame drops while audio is processing | 2 hrs |
| 10.8 | Cross-platform smoke test: macOS, Windows, Linux | 4 hrs |
| 10.9 | Final documentation: README, setup instructions, known issues | 3 hrs |

**Deliverable:** Production-ready application. Performance benchmarks documented.

---

## 11. Unit Test Plan

| Test Module | Test Cases |
|---|---|
| `test_bpm_detector.py` | Detect known-BPM file (120 BPM click track) → assert within ±2 BPM. Silence input → returns `None`. Noise input → returns `None` or within valid range. |
| `test_sync_manager.py` | Integer ratio (120/120=1.0) → zero drift after 1000 beats. Non-integer ratio (120/90=1.333) → drift < 1ms after 1000 beats with correction. BPM change triggers re-stretch. |
| `test_loop_shaper.py` | Record 3 points → evaluate at intermediate position → correct interpolation. Overdub adds to existing lane without clearing others. Rescale 8→4 bars: positions compressed by 0.5×. Clear lane: lane is empty, others untouched. Additive mode: value clamped at parameter bounds. |
| `test_midi_map.py` | Add mapping → lookup returns correct target. Conflict detection: same CC mapped twice → returns conflict string. Save/load round-trip: mappings identical after save + load. Remove mapping → lookup returns `None`. |
| `test_mix_bus.py` | Mix 4 silence buffers → silence output. Mix 4 identical buffers → amplitude = 4×. Master volume 0.5 → output halved. Clipping: mix exceeds 1.0 → clipped to 1.0. |

---

## 12. Performance Budget

| Metric | Target | Rationale |
|---|---|---|
| Audio callback duration | < 50% of buffer period | At 512 samples / 44100 Hz = 11.6ms period → callback must complete in < 5.8ms |
| UI frame rate | ≥ 30 fps | Smooth waveform scrolling and Loop Shaper playhead |
| MIDI-to-audio latency | < 10ms | Imperceptible for live performance |
| BPM detection time | < 5 seconds per track | Acceptable for track loading UX |
| Time-stretch (initial) | < 10 seconds per track | One-time cost on load; display progress bar |
| Memory per channel | < 200 MB (10-min track, float32, stereo, stretched) | 10 min × 44100 Hz × 2 ch × 4 bytes × 2 (raw + stretched) ≈ 212 MB |
| Loop Shaper evaluation | < 0.1ms per channel per tick | Binary search on sorted lane is O(log n); 16 lanes × O(log n) is negligible |

---

*End of Engineering Specification.*

```
# =============================================================================
# Author  : Ho San Ko
# Email   : hko@avalanche.energy
# Project : 4-Channel Music Remix Workstation
# =============================================================================
```
