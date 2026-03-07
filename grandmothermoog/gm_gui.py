#!/usr/bin/env python3
"""
GrandMotherMoog real-time GUI controller.
Sends OSC to ChucK on port 9000.

Requirements:
    pip install python-osc

Usage:
    1. chuck gm.ck
    2. python gm_gui.py
"""

import tkinter as tk
from tkinter import ttk
import math
import os
import queue
import re
import threading
import time

import numpy as np
from PIL import Image, ImageTk

try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    from pythonosc import udp_client
except ImportError:
    raise SystemExit("Install python-osc first:  pip install python-osc")

OSC_IP   = "127.0.0.1"
OSC_PORT = 9000
_client  = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)


def send(addr, val):
    _client.send_message(addr, float(val))


def send_int(addr, val):
    _client.send_message(addr, int(val))


def send_str(addr, val):
    _client.send_message(addr, str(val))


# ─── Knob widget ──────────────────────────────────────────────────────────────

class Knob(tk.Canvas):
    """Rotary knob. Drag up to increase, down to decrease."""

    BG      = "#1a1a2e"
    RING    = "#5555aa"
    BODY    = "#2d2d44"
    INNER   = "#444466"
    NEEDLE  = "#ff6600"
    TIP     = "#ffaa44"
    VAL_CLR = "#8888aa"
    LBL_CLR = "#ccccee"

    def __init__(self, parent, label, min_val, max_val, default,
                 command=None, width=66, height=82, fmt=".1f", **kwargs):
        super().__init__(parent, width=width, height=height,
                         bg=self.BG, highlightthickness=0, **kwargs)
        self.min_val    = float(min_val)
        self.max_val    = float(max_val)
        self._val       = float(default)
        self.command    = command
        self.fmt        = fmt
        self.label_text = label
        self.w, self.h  = width, height
        self.cx         = width  // 2
        self.cy         = (height - 22) // 2 + 2
        self.radius     = min(self.cx, self.cy) - 3
        self._last_y    = None
        self._draw()
        self.bind("<ButtonPress-1>",   self._on_press)
        self.bind("<B1-Motion>",        self._on_drag)
        self.bind("<Double-Button-1>",  self._on_reset)
        self._default = float(default)

    # angle in math convention (y-up, CCW+)
    # min → 7-o'clock (−120°), max → 5-o'clock (−420° = −60°)
    def _angle(self, val):
        rng  = self.max_val - self.min_val
        norm = (val - self.min_val) / rng if rng else 0.0
        return math.radians(-120.0 - norm * 300.0)

    def _draw(self):
        self.delete("all")
        r, cx, cy = self.radius, self.cx, self.cy

        # outer ring
        self.create_oval(cx-r, cy-r, cx+r, cy+r,
                         fill=self.BODY, outline=self.RING, width=2)
        # inner body
        ir = r - 6
        self.create_oval(cx-ir, cy-ir, cx+ir, cy+ir,
                         fill="#252540", outline=self.INNER, width=1)

        # needle
        a  = self._angle(self._val)
        lx = cx + (ir - 3) * math.cos(a)
        ly = cy - (ir - 3) * math.sin(a)   # y-axis flip for canvas
        self.create_line(cx, cy, lx, ly,
                         fill=self.NEEDLE, width=2, capstyle="round")
        self.create_oval(lx-2, ly-2, lx+2, ly+2,
                         fill=self.TIP, outline="")

        # value
        self.create_text(cx, cy + r + 5,
                         text=f"{self._val:{self.fmt}}",
                         fill=self.VAL_CLR, font=("Courier", 7))
        # label
        self.create_text(cx, self.h - 5,
                         text=self.label_text,
                         fill=self.LBL_CLR, font=("Courier", 8, "bold"))

    def _on_press(self, event):
        self._last_y = event.y

    def _on_drag(self, event):
        if self._last_y is None:
            return
        dy     = self._last_y - event.y
        delta  = dy * (self.max_val - self.min_val) / 150.0
        self._val = max(self.min_val, min(self.max_val, self._val + delta))
        self._last_y = event.y
        self._draw()
        if self.command:
            self.command(self._val)

    def _on_reset(self, _event):
        self._val = self._default
        self._draw()
        if self.command:
            self.command(self._val)

    def get(self):
        return self._val


# ─── Signal-flow diagram ──────────────────────────────────────────────────────

class PatchDiagram(tk.Canvas):
    """Static signal-flow diagram showing the GrandMotherVoice patch."""

    def __init__(self, parent, width=660, height=155, **kwargs):
        super().__init__(parent, width=width, height=height,
                         bg="#0a0a18", highlightthickness=0, **kwargs)
        self._draw()

    def _box(self, x, y, w, h, fc, oc, t1, t2=None):
        self.create_rectangle(x, y, x+w, y+h, fill=fc, outline=oc, width=1)
        ty = y + h//2 - (5 if t2 else 0)
        self.create_text(x+w//2, ty, text=t1, fill=oc, font=("Courier", 7, "bold"))
        if t2:
            self.create_text(x+w//2, y+h//2+5, text=t2,
                             fill="#556677", font=("Courier", 6))

    def _arr(self, x1, y1, x2, y2, c="#334455", dash=None):
        kw = dict(fill=c, arrow="last", arrowshape=(5, 7, 3), width=1)
        if dash:
            kw["dash"] = dash
        self.create_line(x1, y1, x2, y2, **kw)

    def _draw(self):
        OSC = ("#0c1a0c", "#44aa55")   # oscillators
        MXR = ("#0c0c1a", "#4455bb")   # mix / routing
        FLT = ("#1a1006", "#cc7722")   # filter / drive
        ENV = ("#1a0808", "#cc4444")   # envelopes
        FX  = ("#0a1220", "#4488cc")   # limiter / reverb
        OUT = ("#061616", "#33aaaa")   # output
        MOD = ("#181806", "#aaaa33")   # modulation
        AC  = "#334455"                # audio-path arrow
        MC  = "#887733"                # modulation arrow

        # ─ Row 1 : sources (y=10) ───────────────────────────────────────────
        self._box(6,   10, 62, 28, *OSC, "VCO 1", "TRI/SAW/SQR")
        self._box(74,  10, 62, 28, *OSC, "VCO 2", "+DETUNE")
        self._box(142, 10, 62, 28, *MXR, "NOISE", "& EXT IN")

        # source bus: vertical drops → horizontal line → arrow to MIX
        for cx in (37, 105, 173):
            self.create_line(cx, 38, cx, 48, fill=AC, width=1)
        self.create_line(37, 48, 173, 48, fill=AC, width=1)
        self._arr(105, 48, 105, 57, AC)

        # ─ Row 2 : main audio chain (y=57..85, center y=71) ─────────────────
        CY = 71
        chain = [
            # x,   w,   fc,  oc,  label,    sub
            (83,   44,  *MXR, "MIX",     None),
            (133,  44,  *FLT, "DRIVE",   None),
            (183,  54,  *FLT, "FILTER",  "4×LPF"),
            (243,  54,  *ENV, "AMP ENV", "ADSR"),
            (303,  38,  *ENV, "VEL",     None),
            (347,  54,  *FX,  "LIMITER", "∞:1"),
            (407,  48,  *FX,  "REVERB",  None),
            (461,  40,  *OUT, "DAC",     "OUT"),
        ]
        prev_rx = None
        for x, w, fc, oc, lbl, sub in chain:
            self._box(x, 57, w, 28, fc, oc, lbl, sub)
            if prev_rx is not None:
                self._arr(prev_rx, CY, x, CY, AC)
            prev_rx = x + w

        # ─ Row 3 : modulation sources (y=108) ───────────────────────────────
        FILT_CX = 183 + 27   # center-x of FILTER box = 210
        self._box(6,   108, 80, 28, *MOD, "LFO", "→PITCH  →CUTOFF")
        self._box(183, 108, 54, 28, *MOD, "FILT ENV", "ADSR →CUTOFF")

        # FILT ENV → FILTER (dashed upward)
        self._arr(FILT_CX, 108, FILT_CX, 85, MC, dash=(4, 3))

        # LFO → VCO pitch (dashed vertical to VCO row)
        self._arr(37, 108, 37, 38, "#666633", dash=(3, 3))
        # LFO → FILTER cutoff (dashed horizontal + up)
        self.create_line(86, 120, FILT_CX - 8, 120,
                         fill="#666633", dash=(3, 3), width=1)
        self._arr(FILT_CX - 8, 120, FILT_CX - 8, 85, "#666633", dash=(3, 3))

        # ─ Legend ────────────────────────────────────────────────────────────
        legend = [
            (530, "#44aa55", "OSC"),
            (560, "#cc7722", "FILT"),
            (595, "#cc4444", "ENV"),
            (625, "#4488cc", "FX"),
            (650, "#aaaa33", "MOD"),
        ]
        for lx, lc, lt in legend:
            self.create_rectangle(lx, 142, lx+8, 150, fill=lc, outline=lc)
            self.create_text(lx+10, 146, text=lt, fill="#556677",
                             font=("Courier", 6), anchor="w")


# ─── Step Sequencer widget ────────────────────────────────────────────────────

class StepSequencer(tk.Frame):
    """16-step note sequencer. Timing runs in a daemon thread, OSC to ChucK."""

    STEPS    = 16
    _NOTE_RE = re.compile(r'^([A-Ga-g])([#b]?)(-?\d+)$')
    _SEMIS   = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}

    # Default pattern: C minor pentatonic, 2 octaves
    _DEFAULT = ["C4","Eb4","F4","G4","Bb4","C5","-",".",
                "G4","F4","Eb4",".",  "C4", "-", ".","."]

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg="#1a1a2e", **kwargs)
        self._running    = False
        self._thread     = None
        self._last_midi  = None
        self._bpm_var    = tk.StringVar(value="120")
        self._step_vars  = [tk.StringVar(value=self._DEFAULT[i])
                            for i in range(self.STEPS)]
        self._cells      = []
        self._build_ui()

    def _build_ui(self):
        ctrl = tk.Frame(self, bg="#1a1a2e")
        ctrl.pack(fill="x", padx=4, pady=(4, 2))

        tk.Label(ctrl, text="BPM", fg="#ccccee", bg="#1a1a2e",
                 font=("Courier", 8, "bold")).pack(side="left", padx=(0, 3))
        tk.Entry(ctrl, textvariable=self._bpm_var, width=5,
                 bg="#2d2d44", fg="#ddddff", insertbackground="#ff6600",
                 font=("Courier", 9), relief="flat", justify="center"
                 ).pack(side="left")

        self._play_btn = tk.Button(
            ctrl, text="▶  PLAY", command=self._toggle,
            bg="#0d2a0d", fg="#44cc44", activebackground="#1a3a1a",
            font=("Courier", 8, "bold"), relief="flat", padx=8, pady=2)
        self._play_btn.pack(side="left", padx=(10, 0))

        tk.Label(ctrl,
                 text="   note: C4  D#3  Gb5  |  .=rest  -=tie",
                 fg="#445566", bg="#1a1a2e",
                 font=("Courier", 7)).pack(side="left", padx=8)

        cells_frame = tk.Frame(self, bg="#1a1a2e")
        cells_frame.pack(padx=4, pady=(0, 4))

        for i in range(self.STEPS):
            col = tk.Frame(cells_frame, bg="#1a1a2e")
            col.pack(side="left", padx=2)
            num_clr = "#556688" if i % 4 == 0 else "#334455"
            tk.Label(col, text=str(i + 1), fg=num_clr, bg="#1a1a2e",
                     font=("Courier", 6)).pack()
            e = tk.Entry(col, textvariable=self._step_vars[i], width=4,
                         bg="#2d2d44", fg="#ddddff",
                         insertbackground="#ff6600",
                         font=("Courier", 8), relief="flat", justify="center")
            e.pack()
            self._cells.append(e)

    # ── playback ──────────────────────────────────────────────────────────────

    def _toggle(self):
        if self._running:
            self._stop()
        else:
            self._start()

    def _start(self):
        self._running = True
        self._play_btn.config(text="■  STOP", bg="#2a0d0d", fg="#ff4444",
                              activebackground="#3a1a1a")
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _stop(self):
        self._running = False
        self._play_btn.config(text="▶  PLAY", bg="#0d2a0d", fg="#44cc44",
                              activebackground="#1a3a1a")
        if self._last_midi is not None:
            send_int("/gm/seq/noteoff", self._last_midi)
            self._last_midi = None
        self.after(0, self._highlight, -1)

    def _loop(self):
        while self._running:
            try:
                bpm = max(10.0, float(self._bpm_var.get()))
            except ValueError:
                bpm = 120.0
            step_dur = 60.0 / bpm / 4   # 16th-note steps

            for i in range(self.STEPS):
                if not self._running:
                    break
                t0   = time.perf_counter()
                midi = self._parse(self._step_vars[i].get().strip())
                self.after(0, self._highlight, i)

                if midi == -1:
                    pass                            # tie: keep ringing
                elif midi is None:                  # rest
                    if self._last_midi is not None:
                        send_int("/gm/seq/noteoff", self._last_midi)
                        self._last_midi = None
                else:                               # new note
                    if self._last_midi is not None:
                        send_int("/gm/seq/noteoff", self._last_midi)
                    send_int("/gm/seq/noteon", midi)
                    self._last_midi = midi

                remaining = step_dur - (time.perf_counter() - t0)
                if remaining > 0:
                    time.sleep(remaining)

        self.after(0, self._highlight, -1)

    def _highlight(self, step):
        for i, e in enumerate(self._cells):
            e.config(bg="#ff6600" if i == step else "#2d2d44")

    def _parse(self, s):
        """Return MIDI int, None (rest), or -1 (tie)."""
        if not s or s == '.':
            return None
        if s == '-':
            return -1
        m = self._NOTE_RE.match(s)
        if not m:
            return None
        name, acc, octave = m.groups()
        semi = self._SEMIS[name.upper()]
        if acc == '#':
            semi += 1
        elif acc == 'b':
            semi -= 1
        return max(0, min(127, (int(octave) + 1) * 12 + semi))


# ─── Spectrogram widget ───────────────────────────────────────────────────────

class Spectrogram(tk.Canvas):
    """Scrolling waterfall spectrogram. Captures from default audio input."""

    SAMPLE_RATE = 44100
    CHUNK       = 1024
    N_FFT       = 2048
    F_MIN       = 30.0
    F_MAX       = 10000.0
    UPDATE_MS   = 33        # ~30 fps
    DB_RANGE    = 60.0      # dB window to display below adaptive peak

    # Colormap: black → deep blue → purple → orange → white
    _CMAP = np.array([
        [0.00, 0x00, 0x00, 0x00],
        [0.20, 0x10, 0x10, 0x80],
        [0.50, 0x99, 0x00, 0x99],
        [0.80, 0xff, 0x66, 0x00],
        [1.00, 0xff, 0xff, 0xcc],
    ], dtype=np.float32)

    def __init__(self, parent, width=620, height=150, **kwargs):
        super().__init__(parent, width=width, height=height,
                         bg="#1a1a2e", highlightthickness=0, **kwargs)
        self.w, self.h = width, height
        self._buf    = np.zeros((height, width), dtype=np.float32)
        self._photo  = None
        self._q      = queue.Queue(maxsize=64)
        self._peak   = -20.0    # smoothed peak dB, adaptive

        freqs = np.fft.rfftfreq(self.N_FFT, 1.0 / self.SAMPLE_RATE)
        mask  = (freqs >= self.F_MIN) & (freqs <= self.F_MAX)
        self._fmask  = mask
        self._fslice = freqs[mask]
        self._log_y  = np.logspace(
            np.log10(self.F_MIN), np.log10(self.F_MAX), height
        )

        if sd is not None:
            self._stream = sd.InputStream(
                samplerate=self.SAMPLE_RATE, channels=1,
                blocksize=self.CHUNK, callback=self._cb
            )
            self._stream.start()
            self.bind("<Destroy>", lambda _: self._stream.stop())
        else:
            self.create_text(width // 2, height // 2,
                             text="sounddevice not available",
                             fill="#667788", font=("Courier", 9))
        self._tick()

    def _cb(self, indata, _frames, _time, _status):
        try:
            self._q.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass

    def _tick(self):
        new_cols = []
        while True:
            try:
                chunk = self._q.get_nowait()
            except queue.Empty:
                break
            mag    = np.abs(np.fft.rfft(chunk, n=self.N_FFT))[self._fmask]
            col    = np.interp(self._log_y, self._fslice, mag)
            col_db = 20.0 * np.log10(col + 1e-8)
            # adaptive peak: fast attack, slow decay
            p = float(np.percentile(col_db, 98))
            self._peak = p if p > self._peak else self._peak * 0.997 + p * 0.003
            col = np.clip((col_db - (self._peak - self.DB_RANGE)) / self.DB_RANGE,
                          0.0, 1.0)
            new_cols.append(col[::-1])   # low freq at bottom

        if new_cols:
            n = len(new_cols)
            self._buf = np.roll(self._buf, -n, axis=1)
            for i, col in enumerate(new_cols):
                self._buf[:, self.w - n + i] = col
            self._render()

        self.after(self.UPDATE_MS, self._tick)

    def _render(self):
        c = self._CMAP
        v = self._buf
        rgb = np.stack([
            np.interp(v, c[:, 0], c[:, 1]),
            np.interp(v, c[:, 0], c[:, 2]),
            np.interp(v, c[:, 0], c[:, 3]),
        ], axis=2).astype(np.uint8)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb, "RGB"))
        self.delete("all")
        self.create_image(0, 0, anchor="nw", image=self._photo)

        log_min = np.log10(self.F_MIN)
        log_rng = np.log10(self.F_MAX) - log_min
        for freq, lbl in [(50,"50"),(100,"100"),(200,"200"),(500,"500"),
                          (1000,"1k"),(2000,"2k"),(5000,"5k"),(10000,"10k")]:
            if self.F_MIN <= freq <= self.F_MAX:
                y = int((1.0 - (np.log10(freq) - log_min) / log_rng) * (self.h - 1))
                self.create_line(0, y, 8, y, fill="#445566")
                self.create_text(10, y, text=lbl, anchor="w",
                                 fill="#8899aa", font=("Courier", 6))


# ─── Helpers ──────────────────────────────────────────────────────────────────

def section(parent, title):
    return tk.LabelFrame(parent, text=title,
                         fg="#8888cc", bg="#1a1a2e",
                         font=("Courier", 8, "bold"),
                         bd=1, relief="groove", labelanchor="n")


def wave_sel(parent, osc_addr, default="SAW"):
    """Small wave-type dropdown; returns the frame."""
    options = ["OFF", "TRI", "SAW", "SQR"]
    vals    = {"OFF": 0, "TRI": 1, "SAW": 2, "SQR": 3}
    f   = tk.Frame(parent, bg="#1a1a2e")
    var = tk.StringVar(value=default)
    cb  = ttk.Combobox(f, textvariable=var, values=options,
                       width=4, state="readonly", font=("Courier", 8))
    cb.pack(padx=4, pady=(8, 0))
    tk.Label(f, text="WAVE", fg="#ccccee", bg="#1a1a2e",
             font=("Courier", 8, "bold")).pack(pady=(2, 6))
    var.trace_add("write", lambda *_: send_int(osc_addr, vals[var.get()]))
    return f


def knob_row(parent, specs):
    """
    specs = [(label, min, max, default, osc_addr, fmt), ...]
    Grids knobs in row=0, column=0,1,2...
    """
    for col, (lbl, mn, mx, dflt, addr, fmt) in enumerate(specs):
        Knob(parent, lbl, mn, mx, dflt,
             command=lambda v, a=addr: send(a, v),
             fmt=fmt).grid(row=0, column=col, padx=4, pady=4)


# ─── Build GUI ────────────────────────────────────────────────────────────────

def build():
    root = tk.Tk()
    root.title("GrandMotherMoog")
    root.configure(bg="#1a1a2e")
    root.resizable(False, False)

    # ttk combobox dark style
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TCombobox",
                    fieldbackground="#2d2d44", background="#2d2d44",
                    foreground="#ddddff",
                    selectbackground="#2d2d44", selectforeground="#ff6600")
    style.map("TCombobox", fieldbackground=[("readonly", "#2d2d44")],
              foreground=[("readonly", "#ddddff")])

    # ── Title ──────────────────────────────────────────────────────────────
    tk.Label(root, text="  GRANDMOTHER MOOG  ",
             bg="#0d0d1a", fg="#ff6600",
             font=("Courier", 13, "bold"), pady=6).pack(fill="x")

    # ── Row 1 : VCO 1 / VCO 2 / MIXER / GLIDE ─────────────────────────────
    r1 = tk.Frame(root, bg="#1a1a2e")
    r1.pack(fill="x", padx=6, pady=3)

    # VCO 1
    v1 = section(r1, "VCO 1")
    v1.pack(side="left", padx=3, fill="y")
    wave_sel(v1, "/gm/vco1wave", "SAW").grid(row=0, column=0, padx=4)
    Knob(v1, "LEVEL", 0, 1, 0.3,
         command=lambda v: send("/gm/vco1level", v),
         fmt=".2f").grid(row=0, column=1, padx=4, pady=4)

    # VCO 2
    v2 = section(r1, "VCO 2")
    v2.pack(side="left", padx=3, fill="y")
    wave_sel(v2, "/gm/vco2wave", "SQR").grid(row=0, column=0, padx=4)
    Knob(v2, "LEVEL", 0, 1, 0.3,
         command=lambda v: send("/gm/vco2level", v),
         fmt=".2f").grid(row=0, column=1, padx=4, pady=4)
    Knob(v2, "DETUNE c", -50, 50, 7.0,
         command=lambda v: send("/gm/detune", v),
         fmt=".1f").grid(row=0, column=2, padx=4, pady=4)

    # MIXER
    mx = section(r1, "MIXER")
    mx.pack(side="left", padx=3, fill="y")
    knob_row(mx, [
        ("NOISE",  0, 1, 0.02, "/gm/noise",  ".3f"),
        ("EXT IN", 0, 1, 0.0,  "/gm/extin",  ".2f"),
    ])

    # GLIDE
    gl = section(r1, "GLIDE")
    gl.pack(side="left", padx=3, fill="y")
    Knob(gl, "ms", 0, 500, 35,
         command=lambda v: send("/gm/glidems", v),
         fmt=".0f").grid(row=0, column=0, padx=4, pady=4)

    # ── Row 2 : FILTER / LFO ───────────────────────────────────────────────
    r2 = tk.Frame(root, bg="#1a1a2e")
    r2.pack(fill="x", padx=6, pady=3)

    ft = section(r2, "FILTER")
    ft.pack(side="left", padx=3, fill="y")
    knob_row(ft, [
        ("CUTOFF", 0, 100, 35, "/gm/cutoff",  ".0f"),
        ("RES",    0, 100, 40, "/gm/res",     ".0f"),
        ("ENV AMT",0, 100, 55, "/gm/envamt",  ".0f"),
    ])

    lf = section(r2, "LFO")
    lf.pack(side="left", padx=3, fill="y")
    knob_row(lf, [
        ("RATE Hz", 0.05, 20,  5.5, "/gm/lforate",   ".2f"),
        ("->PITCH", 0,    100, 10,  "/gm/lfopitch",  ".0f"),
        ("->CUTOFF",0,    100, 18,  "/gm/lfocutoff", ".0f"),
    ])

    # ── Row 3 : AMP ENV / FILT ENV / FX ───────────────────────────────────
    r3 = tk.Frame(root, bg="#1a1a2e")
    r3.pack(fill="x", padx=6, pady=3)

    ae = section(r3, "AMP ENV")
    ae.pack(side="left", padx=3, fill="y")
    knob_row(ae, [
        ("A ms", 1, 2000, 5,    "/gm/ampA", ".0f"),
        ("D ms", 1, 2000, 120,  "/gm/ampD", ".0f"),
        ("S",    0, 1,    0.65, "/gm/ampS", ".2f"),
        ("R ms", 1, 2000, 120,  "/gm/ampR", ".0f"),
    ])

    fe = section(r3, "FILT ENV")
    fe.pack(side="left", padx=3, fill="y")
    knob_row(fe, [
        ("A ms", 1, 2000, 3,   "/gm/filtA", ".0f"),
        ("D ms", 1, 2000, 140, "/gm/filtD", ".0f"),
        ("S",    0, 1,    0.0, "/gm/filtS", ".2f"),
        ("R ms", 1, 2000, 180, "/gm/filtR", ".0f"),
    ])

    fx = section(r3, "FX")
    fx.pack(side="left", padx=3, fill="y")
    knob_row(fx, [
        ("DRIVE",  0, 100, 35, "/gm/drive",  ".0f"),
        ("REVERB", 0, 100, 10, "/gm/reverb", ".0f"),
    ])

    # ── Row 4 : Signal Flow + Record ────────────────────────────────────────
    sf = section(root, "SIGNAL FLOW")
    sf.pack(fill="x", padx=6, pady=3)
    PatchDiagram(sf).pack(padx=4, pady=(4, 2))

    # record bar
    rb = tk.Frame(sf, bg="#1a1a2e")
    rb.pack(fill="x", padx=4, pady=(0, 4))

    tk.Label(rb, text="REC →  output/", fg="#556677", bg="#1a1a2e",
             font=("Courier", 8)).pack(side="left", padx=(2, 0))
    _fname = tk.StringVar(value="recording")
    tk.Entry(rb, textvariable=_fname, width=14,
             bg="#2d2d44", fg="#ddddff", insertbackground="#ff6600",
             font=("Courier", 9), relief="flat", justify="left"
             ).pack(side="left")
    tk.Label(rb, text=".wav", fg="#556677", bg="#1a1a2e",
             font=("Courier", 8)).pack(side="left", padx=(0, 10))

    _recording = [False]

    def _toggle_rec():
        if _recording[0]:
            send_int("/gm/record/stop", 0)
            _recording[0] = False
            _rec_btn.config(text="● REC", bg="#1a0d0d", fg="#cc4444",
                            activebackground="#2a1010")
        else:
            fname = _fname.get().strip() or "recording"
            os.makedirs("output", exist_ok=True)
            send_str("/gm/record/start", "output/" + fname + ".wav")
            _recording[0] = True
            _rec_btn.config(text="■ STOP REC", bg="#4a0000", fg="#ff3333",
                            activebackground="#5a1010")

    _rec_btn = tk.Button(rb, text="● REC", command=_toggle_rec,
                         bg="#1a0d0d", fg="#cc4444", activebackground="#2a1010",
                         font=("Courier", 8, "bold"), relief="flat", padx=8, pady=2)
    _rec_btn.pack(side="left")

    # ── Row 5 : Step Sequencer ──────────────────────────────────────────────
    sq = section(root, "STEP SEQUENCER  ( 16 × 1/16 note )")
    sq.pack(fill="x", padx=6, pady=3)
    StepSequencer(sq).pack(fill="x", padx=4, pady=2)

    # ── Row 6 : Spectrogram ─────────────────────────────────────────────────
    sp = section(root, "SPECTROGRAM  ( time →  |  30 Hz – 10 kHz log )")
    sp.pack(fill="x", padx=6, pady=3)
    Spectrogram(sp).pack(padx=4, pady=4)

    # ── Status bar ─────────────────────────────────────────────────────────
    tk.Label(root,
             text=f"OSC -> {OSC_IP}:{OSC_PORT}   |   drag up/down   |   double-click resets",
             bg="#0d0d1a", fg="#555577", font=("Courier", 7), pady=3
             ).pack(fill="x")

    root.mainloop()


if __name__ == "__main__":
    build()
