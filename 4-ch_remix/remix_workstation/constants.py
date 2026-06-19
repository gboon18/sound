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
