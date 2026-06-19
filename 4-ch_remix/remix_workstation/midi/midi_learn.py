# =============================================================================
# Author  : Ho San Ko
# Email   : hko@avalanche.energy
# Project : 4-Channel Music Remix Workstation
# =============================================================================

"""MIDI learn mode — captures the next incoming message and assigns a mapping."""

from typing import Callable, Optional

from midi.midi_map import MidiAddress, ControlTarget, MidiMap


class MidiLearn:
    """Captures the next MIDI message and assigns it to the pending target.

    Attributes (Section 6 class diagram):
        _active           True while waiting for a MIDI message to capture.
        _pending_target   The ControlTarget to assign on the next message.
        _midi_map         The MidiMap to write the new mapping into.
    """

    def __init__(
        self,
        midi_map: MidiMap,
        conflict_callback: Optional[Callable[[list[str]], None]] = None,
    ) -> None:
        self._active: bool = False
        self._pending_target: Optional[ControlTarget] = None
        self._midi_map = midi_map
        self._conflict_callback = conflict_callback

    # ── Public API ────────────────────────────────────────────────────────────

    def enable(self) -> None:
        self._active = True

    def disable(self) -> None:
        self._active = False
        self._pending_target = None

    def set_target(self, target: ControlTarget) -> None:
        """Declare which control the next MIDI message should be mapped to."""
        self._pending_target = target
        self._active = True

    def is_active(self) -> bool:
        return self._active

    def on_midi(self, addr: MidiAddress, value: int) -> None:
        """Handle an incoming MIDI message while learn mode is active.

        Steps (Section 7.12):
        1. Assign addr → pending_target in the map.
        2. Check conflicts; invoke conflict callback if any.
        3. Deactivate learn mode.
        """
        if not self._active or self._pending_target is None:
            return

        self._midi_map.add(addr, self._pending_target)
        conflicts = self._midi_map.check_conflicts()
        if conflicts and self._conflict_callback is not None:
            self._conflict_callback(conflicts)

        self._active = False
        self._pending_target = None
