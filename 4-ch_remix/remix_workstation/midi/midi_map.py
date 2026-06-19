# =============================================================================
# Author  : Ho San Ko
# Email   : hko@avalanche.energy
# Project : 4-Channel Music Remix Workstation
# =============================================================================

"""MIDI mapping store — JSON load/save, lookup, conflict detection (Section 7.13)."""

import json
import threading
from pathlib import Path
from typing import NamedTuple, Optional

from constants import Param


class MidiAddress(NamedTuple):
    """Uniquely identifies an incoming MIDI message."""
    channel: int       # 0-15
    msg_type: str      # "cc" or "note"
    number: int        # 0-127


class ControlTarget(NamedTuple):
    """The engine control to drive when a MidiAddress is received."""
    channel_idx: int   # 0-3 for a channel param; -1 for global
    param: object      # Param enum member or str action key
    action: str        # "set", "toggle", or "trigger"


class MidiMap:
    """Store, load, save, and query MIDI-to-control mappings.

    Attributes (Section 6 class diagram):
        _mappings   MidiAddress → ControlTarget forward map.
        _reverse    ControlTarget → MidiAddress reverse map (for conflict detect).
        _lock       Protects both dicts.
    """

    def __init__(self) -> None:
        self._mappings: dict[MidiAddress, ControlTarget] = {}
        self._reverse: dict[ControlTarget, MidiAddress] = {}
        self._lock: threading.Lock = threading.Lock()

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def add(self, addr: MidiAddress, target: ControlTarget) -> None:
        with self._lock:
            self._mappings[addr] = target
            self._reverse[target] = addr

    def remove(self, addr: MidiAddress) -> None:
        with self._lock:
            target = self._mappings.pop(addr, None)
            if target is not None:
                self._reverse.pop(target, None)

    def lookup(self, addr: MidiAddress) -> Optional[ControlTarget]:
        with self._lock:
            return self._mappings.get(addr)

    # ── Conflict detection ────────────────────────────────────────────────────

    def check_conflicts(self) -> list[str]:
        """Return human-readable descriptions of any duplicate mappings.

        A conflict occurs when one MidiAddress maps to multiple ControlTargets
        (impossible with the current dict structure, but checked for load-time
        JSON corruption) OR when one ControlTarget is bound to multiple addresses.
        """
        conflicts: list[str] = []
        with self._lock:
            # Reverse scan: one target → multiple addresses?
            target_to_addrs: dict[ControlTarget, list[MidiAddress]] = {}
            for addr, target in self._mappings.items():
                target_to_addrs.setdefault(target, []).append(addr)
            for target, addrs in target_to_addrs.items():
                if len(addrs) > 1:
                    conflicts.append(
                        f"Target {target} is assigned to multiple MIDI addresses: {addrs}"
                    )
        return conflicts

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Serialize mappings to JSON at *path* (Section 7.13)."""
        data: dict[str, dict] = {}
        with self._lock:
            for addr, target in self._mappings.items():
                key = f"{addr.channel}:{addr.msg_type}:{addr.number}"
                param_name = target.param.name if isinstance(target.param, Param) else str(target.param)
                data[key] = {
                    "channel_idx": target.channel_idx,
                    "param": param_name,
                    "action": target.action,
                }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load(self, path: str) -> None:
        """Deserialize mappings from JSON at *path*."""
        p = Path(path)
        if not p.exists():
            return
        data: dict = json.loads(p.read_text(encoding="utf-8"))
        with self._lock:
            self._mappings.clear()
            self._reverse.clear()
            for key, value in data.items():
                ch_str, msg_type, num_str = key.split(":")
                addr = MidiAddress(
                    channel=int(ch_str),
                    msg_type=msg_type,
                    number=int(num_str),
                )
                param_name = value["param"]
                try:
                    param: object = Param[param_name]
                except KeyError:
                    param = param_name  # non-Param action key
                target = ControlTarget(
                    channel_idx=int(value["channel_idx"]),
                    param=param,
                    action=value["action"],
                )
                self._mappings[addr] = target
                self._reverse[target] = addr
