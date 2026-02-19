from __future__ import annotations

from dataclasses import dataclass
from pythonosc.udp_client import SimpleUDPClient


@dataclass(frozen=True)
class Target:
    ip: str = "127.0.0.1"
    port: int = 9000


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def main() -> None:
    t = Target()
    client = SimpleUDPClient(t.ip, t.port)

    help_text = """
Commands:
  tempo <bpm>              e.g. tempo 128
  bank <0|1>               e.g. bank 1
  pause <0|1>              e.g. pause 1
  chordbeats <int>         e.g. chordbeats 4
  mgain <0..1>              e.g. mgain 0.8

  lrev <0..1>              e.g. lrev 0.25
  lgain <0..1>             e.g. lgain 0.2
  lcut <hz>                e.g. lcut 2500
  lq <q>                   e.g. lq 1.2

  rrev <0..1>              e.g. rrev 0.6
  rgain <0..1>             e.g. rgain 0.2
  rcut <hz>                e.g. rcut 600
  rq <q>                   e.g. rq 3.0

  quit
"""
    print(help_text.strip())

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not line:
            continue
        if line.lower() in {"quit", "exit"}:
            break

        parts = line.split()
        if len(parts) != 2:
            print("Expected: <cmd> <value>")
            continue

        cmd, val_s = parts[0].lower(), parts[1]

        try:
            if cmd == "tempo":
                v = float(val_s)
                client.send_message("/tempo", v)
            elif cmd == "bank":
                v = int(val_s)
                client.send_message("/bank", v)
            elif cmd == "pause":
                v = int(val_s)
                client.send_message("/pause", v)
            elif cmd in {"chordbeats", "cb"}:
                v = int(val_s)
                client.send_message("/chordBeats", v)

            elif cmd == "mgain":
                v = clamp01(float(val_s))
                client.send_message("/master/gain", v)

            elif cmd == "lrev":
                v = clamp01(float(val_s))
                client.send_message("/left/rev", v)
            elif cmd == "lgain":
                v = clamp01(float(val_s))
                client.send_message("/left/gain", v)
            elif cmd == "lcut":
                v = float(val_s)
                client.send_message("/left/cutoff", v)
            elif cmd == "lq":
                v = float(val_s)
                client.send_message("/left/q", v)

            elif cmd == "rrev":
                v = clamp01(float(val_s))
                client.send_message("/right/rev", v)
            elif cmd == "rgain":
                v = clamp01(float(val_s))
                client.send_message("/right/gain", v)
            elif cmd == "rcut":
                v = float(val_s)
                client.send_message("/right/cutoff", v)
            elif cmd == "rq":
                v = float(val_s)
                client.send_message("/right/q", v)
            else:
                print(f"Unknown cmd: {cmd}")
                continue

            print("sent")
        except ValueError:
            print("Bad value type for that command.")

if __name__ == "__main__":
    main()
