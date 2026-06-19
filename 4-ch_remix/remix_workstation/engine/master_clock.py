# =============================================================================
# Author  : Ho San Ko
# Email   : hko@avalanche.energy
# Project : 4-Channel Music Remix Workstation
# =============================================================================

"""Master BPM clock — single timing source for all four channel playheads.

Design notes (Section 7.2):
- Runs on its own daemon thread; does NOT drive audio directly.
- The sounddevice callback reads master_clock.get_bpm() to decide how many
  samples to advance each tick.
- Uses time.perf_counter() for sub-millisecond precision.
- All mutable state protected by threading.Lock.
"""

import threading
import time
from typing import Callable

from constants import DEFAULT_MASTER_BPM, MASTER_BPM_MIN, MASTER_BPM_MAX


class MasterClock:
    """Generates beat ticks at the current BPM from a dedicated daemon thread.

    Attributes match the class diagram in Section 6:
        _bpm            Current BPM (float, lock-protected).
        _running        True while the clock thread is alive.
        _tick_event     Event used to interrupt the sleep on stop().
        _lock           Protects _bpm.
        _thread         The daemon Thread.
    """

    def __init__(self, bpm: float = DEFAULT_MASTER_BPM) -> None:
        self._bpm: float = float(bpm)
        self._running: bool = False
        self._tick_event: threading.Event = threading.Event()
        self._lock: threading.Lock = threading.Lock()
        self._thread: threading.Thread | None = None

        # Internal state not in the class diagram but required by the spec
        self._beat_count: int = 0
        self._callbacks: list[Callable[[], None]] = []
        self._callbacks_lock: threading.Lock = threading.Lock()
        self._tap_times: list[float] = []  # stores perf_counter timestamps of taps

    # ── Public API (Section 6 class diagram) ─────────────────────────────────

    def start(self) -> None:
        """Start the clock thread. No-op if already running."""
        if self._running:
            return
        self._running = True
        self._beat_count = 0
        self._tick_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="MasterClock"
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the clock thread to stop and wait for it to join."""
        self._running = False
        self._tick_event.set()  # unblock a sleeping Event.wait()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._tick_event.clear()

    def set_bpm(self, bpm: float) -> None:
        """Set BPM, clamped to [MASTER_BPM_MIN, MASTER_BPM_MAX]."""
        bpm = max(MASTER_BPM_MIN, min(MASTER_BPM_MAX, float(bpm)))
        with self._lock:
            self._bpm = bpm

    def get_bpm(self) -> float:
        """Return current BPM (thread-safe)."""
        with self._lock:
            return self._bpm

    def tap_tempo(self) -> None:
        """Record one tap; derive BPM from the last 2–4 tap intervals.

        Stores the last four perf_counter timestamps.  When at least two taps
        are available, computes the mean inter-tap interval and sets the BPM.
        A tap that arrives more than 3 seconds after the previous one resets
        the buffer (treated as a fresh sequence).
        """
        now = time.perf_counter()

        # Reset buffer if the gap is too large (user paused between taps)
        if self._tap_times and (now - self._tap_times[-1]) > 3.0:
            self._tap_times.clear()

        self._tap_times.append(now)
        if len(self._tap_times) > 4:
            self._tap_times = self._tap_times[-4:]

        if len(self._tap_times) >= 2:
            intervals = [
                self._tap_times[i] - self._tap_times[i - 1]
                for i in range(1, len(self._tap_times))
            ]
            mean_interval = sum(intervals) / len(intervals)
            if mean_interval > 0:
                self.set_bpm(60.0 / mean_interval)

    def register_callback(self, fn: Callable[[], None]) -> None:
        """Register a callable invoked on every beat tick.

        The callback is called from the clock thread — keep it short and
        non-blocking.  For Qt, post a queued signal rather than touching
        widgets directly.
        """
        with self._callbacks_lock:
            if fn not in self._callbacks:
                self._callbacks.append(fn)

    def unregister_callback(self, fn: Callable[[], None]) -> None:
        """Remove a previously registered callback."""
        with self._callbacks_lock:
            try:
                self._callbacks.remove(fn)
            except ValueError:
                pass

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def beat_count(self) -> int:
        """Monotonically increasing beat counter (no lock needed — int is atomic on CPython)."""
        return self._beat_count

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run(self) -> None:
        """Clock thread body.

        Maintains a running 'next_tick' timestamp computed from
        time.perf_counter() to avoid cumulative drift.  On each iteration:

        1. Compute next_tick += 60.0 / bpm  (beat period in seconds).
        2. Sleep until next_tick using Event.wait(timeout) so that stop()
           can interrupt the sleep immediately.
        3. Increment beat counter and fire callbacks.

        Reading _bpm inside the loop (rather than caching it) means BPM
        changes take effect on the very next beat with no special signalling.
        """
        next_tick = time.perf_counter()

        while self._running:
            beat_period = 60.0 / self.get_bpm()
            next_tick += beat_period

            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0.0:
                # Event.wait returns True if the event was set (i.e. stop()
                # was called), False on timeout (normal beat expiry).
                interrupted = self._tick_event.wait(timeout=sleep_for)
                self._tick_event.clear()
                if interrupted and not self._running:
                    break  # stop() was called; exit cleanly

            self._beat_count += 1
            self._fire_callbacks()

    def _fire_callbacks(self) -> None:
        """Invoke all registered callbacks, swallowing exceptions individually."""
        with self._callbacks_lock:
            snapshot = list(self._callbacks)
        for cb in snapshot:
            try:
                cb()
            except Exception:
                # Callbacks must not crash the clock thread
                pass
