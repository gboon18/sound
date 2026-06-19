# 4-Channel Music Remix Workstation

```
Author  : Ho San Ko
Email   : hko@avalanche.energy
Version : 1.0.0
```

A Python-based real-time 4-channel DJ/remix workstation that fuses
Rekordbox-style loop DJing with Ableton-style live automation.

The centrepiece feature is the **Loop Shaper** — a per-channel gesture
automation recorder that captures knob movements across all 16 DSP parameters
and replays them in perfect sync with every subsequent loop iteration.

---

## Features

- **4 independent channels** — each loads its own audio file
- **Master clock** — single BPM source drives all four playheads; tap-tempo included
- **Time-stretch** — `pyrubberband` stretches every track to the master BPM while preserving pitch
- **BPM detection** — `librosa` auto-detects each track's BPM on load
- **Per-channel DSP chain** — EQ → Filter → Reverb → Echo → Pitch → Volume/Pan (powered by Spotify `pedalboard`)
- **Loop system** — set loop in/out with beat-grid snap, seamless 2 ms crossfade, 4 hot cues per channel
- **Loop Shaper** — records/overdubs/replays knob gestures locked to the loop timeline (16 parameter lanes, additive/absolute modes, proportional rescale on loop-length change)
- **MIDI learn** — map any CC or Note from any USB/DIN controller; mappings persist as JSON
- **WAV recording** — capture the full mix with punch-in on loop boundary

---

## Requirements

| Requirement | Notes |
|---|---|
| **Python 3.12** | Older versions are not supported (union-type syntax) |
| **rubberband CLI** | External binary — see [Installation](#installation) |
| **ASIO or WDM driver** (Windows) | Low-latency audio; use ASIO4ALL if no ASIO card |
| A **USB MIDI controller** | Optional; app works fully with mouse/keyboard |

---

## Installation

### Option A — Poetry (recommended for this repo)

Follow the one-time setup in [README.poetry.md](README.poetry.md) to configure
pyenv + Poetry for this project, then:

```powershell
# From the 4-ch_remix/ directory
poetry install
```

That installs all Python packages.  You still need the **rubberband CLI** binary:

```powershell
# Download from https://breakfastquay.com/rubberband/
# Extract and add the folder containing rubberband.exe to PATH.
# Verify:
rubberband --version
```

### Option B — pip

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Then install rubberband CLI as above.
```

---

## Running

### GUI (default)

```powershell
# Poetry
poetry run remix

# pip venv
python remix_workstation/main.py
```

Optional flags:

| Flag | Default | Description |
|---|---|---|
| `--buffer-size N` | `512` | Audio buffer size in samples.  Lower = less latency, higher CPU risk. |
| `--sample-rate N` | `44100` | Output sample rate in Hz. |
| `--no-ui` | off | Headless CLI mode (audio only, no window). |

```powershell
# Example: lower latency
poetry run remix --buffer-size 256

# Headless test run
poetry run remix --no-ui
```

### First-run checklist

1. Open the app — you will see 4 channel strips and a master section at the top.
2. Click **LOAD** on any channel strip and pick an audio file (`.mp3`, `.wav`, `.flac`, `.aiff`).
   - BPM is detected automatically.  If detection fails, the label shows `? BPM — set manually` (the master BPM knob still controls playback speed).
3. Press **▶** on the channel strip.  The track plays at the master BPM.
4. Adjust the **BPM** knob in the master section to sync multiple tracks.

---

## Configuration

### `remix_workstation/config/settings.json`

Persisted automatically on exit; edit manually if needed.

```json
{
  "sample_rate": 44100,
  "buffer_size": 512,
  "master_bpm": 120.0,
  "master_volume": 1.0,
  "midi_port_index": 0
}
```

| Key | Type | Description |
|---|---|---|
| `sample_rate` | int | Sample rate (Hz).  Must match your audio interface. |
| `buffer_size` | int | Frames per audio callback.  512 = ~11.6 ms latency. |
| `master_bpm` | float | BPM restored on next launch. |
| `master_volume` | float | 0.0 – 1.0 linear gain. |
| `midi_port_index` | int | Index of the MIDI port to open (0 = first available). |

### `remix_workstation/config/midi_map.json`

MIDI mappings saved automatically on exit.  You can also edit it by hand:

```json
{
  "0:cc:1": {"channel_idx": 0, "param": "EQ_HIGH", "action": "set"},
  "0:note:36": {"channel_idx": 0, "param": "loop_in", "action": "trigger"}
}
```

Key format: `<midi_channel>:<type>:<number>` where type is `cc` or `note`.

`param` is either a **DSP parameter name** (from the table below) or a
**string action** (from the MIDI action table further below).

---

## Controls Reference

### Master Section (top bar)

| Control | Action |
|---|---|
| **BPM knob** | Set master BPM (60 – 300).  All channels re-stretch immediately. |
| **TAP** | Tap tempo — 2 to 4 taps, mean interval → BPM. |
| **VOL knob** | Master output volume (0 – 1). |
| **● REC** | Toggle WAV recording.  File saved to `recordings/remix_YYYYMMDD_HHMMSS.wav`. |
| **MIDI LEARN** | Activate MIDI learn mode — see [MIDI Learn](#midi-learn). |

### Channel Strip

#### Transport

| Button | Action |
|---|---|
| **▶** | Play (or resume from pause) |
| **⏸** | Pause (hold position) |
| **⏹** | Stop and rewind to position 0 |

#### DSP Parameters

All 16 parameters are controlled by the rotary knobs on each channel strip.
Mouse drag (vertical) or scroll wheel adjusts the value.

| Section | Parameter | Range | Default | Unit |
|---|---|---|---|---|
| EQ | EQ HIGH | −12 → +12 | 0 | dB |
| EQ | EQ MID | −12 → +12 | 0 | dB |
| EQ | EQ LOW | −12 → +12 | 0 | dB |
| Filter | FILTER CUTOFF | 20 → 20 000 | 20 000 | Hz |
| Filter | FILTER RESONANCE | 0.1 → 10 | 0.707 | Q |
| Filter | FILTER TYPE | 0 / 1 | 0 | 0 = LP, 1 = HP |
| Reverb | REVERB SIZE | 0 → 1 | 0.3 | — |
| Reverb | REVERB DAMP | 0 → 1 | 0.5 | — |
| Reverb | REVERB MIX | 0 → 1 | 0 | — |
| Echo | ECHO TIME | 10 → 2 000 | 500 | ms |
| Echo | ECHO FEEDBACK | 0 → 0.95 | 0.3 | — |
| Echo | ECHO MIX | 0 → 1 | 0 | — |
| Pitch | PITCH SEMITONE | −12 → +12 | 0 | semitones |
| Pitch | PITCH CENTS | −100 → +100 | 0 | cents |
| Output | VOLUME | 0 → 1 | 0.8 | — |
| Output | PAN | −1 → +1 | 0 | L = −1, R = +1 |

#### Loop Controls

| Button | Action |
|---|---|
| **IN** | Set loop-in point (snaps to nearest beat) |
| **OUT** (first press) | Set loop-out point, activate loop |
| **OUT** (second press) | Escape loop (keep in/out points) |
| **LOOP** | Toggle loop on/off without moving in/out |
| **ESC** | Escape loop without clearing in/out |

#### Hot Cues

Each of the 4 **CUE** buttons stores one absolute position in the track.

| Interaction | Action |
|---|---|
| **Left-click** | Recall — jump playhead to stored position |
| **Right-click** | Set — store current playhead as this cue |

Hot cues are in absolute track time; recalling a cue that falls outside the
current loop region does not break loop playback.

#### Loop Shaper

See the [Loop Shaper Guide](#loop-shaper-guide) below for full detail.

| Button | Action |
|---|---|
| **REC** | Arm record — recording starts at the next loop boundary |
| **OVR** | Arm overdub — merges on top of existing automation at the next loop boundary |
| **CLR Lane** | Clear one parameter lane (choose from the dropdown) |
| **CLR All** | Erase all lanes and return to IDLE |
| **ABS / ADD** | Toggle Absolute / Additive mode |

---

## MIDI Learn

1. Click **MIDI LEARN** in the master section.
2. Click the knob or button you want to map — it is now the *pending target*.
3. Move a knob or press a button on your hardware controller.
   The mapping is created instantly and highlighted in the UI.
4. Click **MIDI LEARN** again to exit learn mode.

MIDI mappings are saved to `config/midi_map.json` when the app exits and
reloaded on next launch.

If a CC or Note is already mapped to another control, a warning dialog appears:
confirm to overwrite or cancel to keep the existing mapping.

### MIDI Action Strings (for `midi_map.json`)

These can be assigned to Note On/Off messages for engine-level controls that
are not DSP parameters:

| `param` string | Triggered on | Effect |
|---|---|---|
| `play` | Note On | Toggle play/pause for `channel_idx` |
| `stop` | Note On | Stop + rewind `channel_idx` |
| `loop_in` | Note On | Set loop-in at current playhead |
| `loop_out` | Note On | Set loop-out / escape (same as the OUT button) |
| `loop_toggle` | Note On | Toggle loop on/off |
| `loop_escape` | Note On | Escape loop |
| `hot_cue_0` … `hot_cue_3` | Note On | Recall hot cue 0 – 3 |
| `hot_cue_set_0` … `hot_cue_set_3` | Note On | Set hot cue 0 – 3 |
| `tap_tempo` | Note On | Tap tempo (global) |
| `record` | Note On | Toggle WAV recording on/off |
| `punch_in` | Note On | Arm punch-in (recording starts at next loop boundary) |
| `shaper_record` | Note On | Arm Loop Shaper record for `channel_idx` |
| `shaper_overdub` | Note On | Arm Loop Shaper overdub for `channel_idx` |

Note Off messages are ignored for all actions (one-shot trigger model).

---

## Loop Shaper Guide

The Loop Shaper records real-time knob movements as automation gestures locked
to the loop timeline.  Each loop iteration replays the gesture exactly in sync.

### Workflow

```
1. Set a loop (IN → OUT buttons or MIDI).
2. Press REC on the Loop Shaper panel.
   → State: ARMED (waiting for the next loop start for clean alignment).
3. At the loop boundary the shaper automatically enters RECORDING.
4. Move any knobs freely — every gesture is captured.
5. Press REC again (or wait for one full loop pass) to stop recording.
   → State: PLAYING (automation replays every loop).
```

### Overdub

```
1. While in PLAYING state, press OVR.
   → State: ARMED (overdub variant — existing lanes are NOT cleared).
2. At the next loop boundary → OVERDUBBING.
3. Touch only the knobs you want to change.
   - Untouched lanes replay as before.
   - Points within ±0.5 % of the loop where you move are replaced.
   - Regions you do not touch are preserved intact.
4. Press OVR again to stop overdubbing → back to PLAYING.
```

### Modes

| Mode | Behaviour |
|---|---|
| **Absolute** (ABS) | Automation value directly replaces the knob position. |
| **Additive** (ADD) | Automation value is added to the manual knob position and clamped to the parameter range. Useful for layering a gesture on top of a static setting. |

### Rescale on loop-length change

If you shorten the loop after recording automation:

- Points that fall inside the new shorter loop expand proportionally toward 1.0.
- Points beyond the new end are **discarded** (they fall outside the loop region).

If you lengthen the loop:

- All points compress toward 0.0 (they still represent the same musical positions).
- The tail region has no automation (last value is held).

---

## Recording

The app records the post-mix stereo output (all 4 channels mixed, effects applied).

### Immediate record

Click **● REC** in the master section.  Recording begins immediately.
Click again to stop.  The WAV file is saved automatically with a timestamp filename.

### Punch-in (clean loop boundary)

1. In your MIDI map, assign a Note to the `punch_in` action.
2. Arm punch-in (press the Note or click the button if you add one to the UI).
3. Recording begins on the next loop start, ensuring a clean edit point.

Output files are written to `recordings/` (created automatically) in the
project working directory.

---

## Development & Testing

```powershell
# Run all tests
poetry run pytest

# Or with pip venv
python -m pytest remix_workstation/tests

# Run only the Loop Shaper tests
poetry run pytest remix_workstation/tests/test_loop_shaper.py -v
```

### Test suite overview

| File | Covers |
|---|---|
| `test_bpm_detector.py` | librosa beat detection, silence/noise edge cases |
| `test_loop_shaper.py` | 22 tests — interpolation, overdub merge, rescale, state machine |
| `test_midi_map.py` | Add/remove/lookup, conflict detection, JSON round-trip |
| `test_mix_bus.py` | Silence, sum, clipping, master volume, dtype |
| `test_sync_manager.py` | Ratio calculation (integration tests skipped until TrackPlayer complete) |

---

## Project Structure

```
4-ch_remix/
├── README.md                       ← this file
├── pyproject.toml                  ← Poetry manifest
├── requirements.txt                ← pip alternative
├── 4CH_Remix_Workstation_Engineering_Spec.md
│
└── remix_workstation/
    ├── main.py                     ← entry point
    ├── constants.py                ← all ranges, enums, named constants
    │
    ├── engine/
    │   ├── master_clock.py         ← BPM clock, tap tempo, beat callbacks
    │   ├── track_player.py         ← file load, pyrubberband stretch, advance()
    │   ├── bpm_detector.py         ← librosa beat detection
    │   ├── sync_manager.py         ← stretch ratios, drift correction
    │   └── mix_bus.py              ← sum 4 channels, master volume, clip
    │
    ├── dsp/
    │   ├── channel_dsp.py          ← pedalboard chain orchestrator
    │   ├── eq.py                   ← 3-band shelf/peak EQ
    │   ├── filter.py               ← LP / HP filter (dual-plugin passthrough)
    │   ├── reverb.py               ← Reverb
    │   ├── echo.py                 ← Delay / echo
    │   └── pitch.py                ← Pitch shift
    │
    ├── loop/
    │   ├── loop_manager.py         ← loop in/out, beat snap, hot cues
    │   └── loop_shaper.py          ← automation recorder, overdub, rescale
    │
    ├── midi/
    │   ├── midi_input.py           ← rtmidi listener, CC + Note parsing
    │   ├── midi_learn.py           ← capture next message, assign mapping
    │   └── midi_map.py             ← JSON store, lookup, conflict detection
    │
    ├── ui/
    │   ├── main_window.py          ← top-level window, worker thread loader
    │   ├── channel_strip.py        ← per-channel composite widget
    │   ├── master_section.py       ← BPM, tap, master vol, record, MIDI learn
    │   ├── knob_widget.py          ← custom rotary knob (drag + wheel)
    │   ├── waveform_widget.py      ← scrolling waveform + beat grid + loop region
    │   └── loop_shaper_widget.py   ← 16 automation lanes + controls
    │
    ├── recording/
    │   └── recorder.py             ← WAV capture, punch-in
    │
    ├── config/
    │   ├── settings.json           ← persisted user preferences
    │   └── midi_map.json           ← persisted MIDI mappings
    │
    └── tests/
        ├── conftest.py
        ├── test_bpm_detector.py
        ├── test_loop_shaper.py
        ├── test_midi_map.py
        ├── test_mix_bus.py
        └── test_sync_manager.py
```

---

## Troubleshooting

### "No module named 'rubberband'" or stretch hangs forever

`pyrubberband` requires the **rubberband CLI binary** on the system PATH.

1. Download from [breakfastquay.com/rubberband](https://breakfastquay.com/rubberband/)
2. Extract and add the directory containing `rubberband.exe` to your PATH.
3. Verify: `rubberband --version` prints a version string.

### Audio underruns / crackling

The app prints a warning on exit if underruns were detected.  To fix:

- Increase `buffer_size` in `config/settings.json` (try 1024 or 2048).
- Use an ASIO driver (Windows) — install [ASIO4ALL](https://www.asio4all.org/) if you do not have a dedicated audio interface.
- Close other audio applications.

### MIDI port not found

The app opens port index `0` (the first MIDI input device) by default.
If your controller is not on port 0, list available ports:

```python
from midi.midi_input import MidiInput
for i, name in enumerate(MidiInput().list_ports()):
    print(i, name)
```

Set the correct index in `config/settings.json`:

```json
{ "midi_port_index": 1 }
```

### BPM detection returns "? BPM"

librosa's beat tracker needs at least a few seconds of audio with a clear
rhythmic pulse.  For tracks with unusual meters or very sparse percussion:

1. The BPM display shows `? BPM — set manually`.
2. Use the master BPM knob to dial in the correct tempo.
3. All time-stretch ratios update automatically.

### High CPU usage during track load

Time-stretching via pyrubberband is CPU-intensive and single-threaded inside
the C++ library.  Loading runs on a background QThreadPool worker, so the UI
and audio remain responsive — but load time for a 10-minute track at a
large ratio can take several seconds.  The **LOAD** button is disabled during
load and re-enables on completion.

---

## Performance Budget (reference)

| Metric | Target |
|---|---|
| Audio callback duration | < 5.8 ms (50 % of 512-sample buffer period at 44 100 Hz) |
| UI frame rate | ≥ 30 fps |
| MIDI-to-audio latency | < 10 ms |
| BPM detection per track | < 5 s |
| Time-stretch per track | < 10 s (10-min track at 1.5× ratio) |
| RAM per channel | < 200 MB (10-min stereo float32 raw + stretched) |
