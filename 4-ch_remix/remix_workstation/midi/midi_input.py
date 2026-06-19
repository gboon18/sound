# =============================================================================
# Author  : Ho San Ko
# Email   : hko@avalanche.energy
# Project : 4-Channel Music Remix Workstation
# =============================================================================

"""MIDI input listener — parses CC and Note messages via python-rtmidi."""

from typing import Callable, Optional

from midi.midi_map import MidiAddress


class MidiInput:
    """Listens on a MIDI input port and dispatches parsed messages.

    Attributes (Section 6 class diagram):
        _port            The open rtmidi.MidiIn instance.
        _callback        User-provided callback (MidiAddress, value) → None.
        _channel_filter  If set, discard messages from other channels.
    """

    def __init__(self) -> None:
        self._port = None           # rtmidi.MidiIn, set on open()
        self._callback: Optional[Callable[[MidiAddress, int], None]] = None
        self._channel_filter: Optional[int] = None

    # ── Port management ───────────────────────────────────────────────────────

    def list_ports(self) -> list[str]:
        """Return names of all available MIDI input ports."""
        import rtmidi  # deferred — optional dependency
        midi_in = rtmidi.MidiIn()
        return [midi_in.get_port_name(i) for i in range(midi_in.get_port_count())]

    def open(self, port_idx: int) -> None:
        """Open the MIDI input port at *port_idx*."""
        import rtmidi
        self._port = rtmidi.MidiIn()
        self._port.open_port(port_idx)
        self._port.set_callback(self._on_message)

    def close(self) -> None:
        if self._port is not None:
            self._port.close_port()
            self._port = None

    def set_callback(self, fn: Callable[[MidiAddress, int], None]) -> None:
        self._callback = fn

    def set_channel_filter(self, channel: Optional[int]) -> None:
        self._channel_filter = channel

    # ── Internal ──────────────────────────────────────────────────────────────

    def _on_message(self, midi_msg, timestamp) -> None:  # type: ignore[override]
        """Parse raw MIDI bytes and invoke registered callback.

        Steps (Section 7.11):
        status = msg[0] & 0xF0 → CC=0xB0, NoteOn=0x90, NoteOff=0x80
        channel = msg[0] & 0x0F
        """
        msg, _ = midi_msg if isinstance(midi_msg, tuple) else (midi_msg, None)
        if len(msg) < 2:
            return

        status = msg[0] & 0xF0
        channel = msg[0] & 0x0F

        if self._channel_filter is not None and channel != self._channel_filter:
            return

        if status == 0xB0 and len(msg) >= 3:
            addr = MidiAddress(channel=channel, msg_type="cc", number=msg[1])
            value = msg[2]
        elif status == 0x90 and len(msg) >= 3 and msg[2] > 0:
            addr = MidiAddress(channel=channel, msg_type="note", number=msg[1])
            value = msg[2]
        elif status == 0x80 and len(msg) >= 3:
            addr = MidiAddress(channel=channel, msg_type="note", number=msg[1])
            value = 0
        else:
            return

        if self._callback is not None:
            self._callback(addr, value)
