# leftright

Stereo generative melody system for ChucK with OSC control, managed by Poetry (Python 3.12).

This project contains:

- `melody_bank_stereo.ck` — ChucK stereo melody engine (major left / minor right)
- `control_osc.py` — Interactive OSC controller (terminal-based)
- `send_osc.py` — Minimal one-shot OSC sender
- `pyproject.toml` — Poetry project definition
- `poetry.lock` — Locked dependency versions
- `.venv/` — Poetry-managed virtual environment (do not commit)

The ChucK patch listens on UDP port **9000** and responds to OSC messages such as:

- `/tempo` (float)
- `/bank` (int)
- `/pause` (int)
- `/left/rev` (float 0..1)
- `/right/cutoff` (float Hz)

---

## Requirements

- Windows PowerShell
- Python 3.12
- Poetry
- ChucK installed and available in PATH

Verify Poetry:

```powershell
poetry --version
```

---

## Virtual Environment Setup (Poetry)

From the project directory:

```powershell
poetry install
```

This creates or syncs the `.venv` directory using `pyproject.toml` and `poetry.lock`.

---

## Activate the Virtual Environment (PowerShell)

```powershell
Invoke-Expression (poetry env activate)
```

Confirm activation:

```powershell
python -c "import sys; print(sys.executable)"
```

It should point inside `.venv`.

Deactivate when finished:

```powershell
deactivate
```

---

## Alternative (No Activation)

Instead of activating:

```powershell
poetry run python control_osc.py
```

---

## Running the System

### 1. Start ChucK (separate terminal)

```powershell
chuck melody_bank_stereo.ck
```

Ensure it is listening on port 9000.

### 2. Run the OSC Controller

If activated:

```powershell
python control_osc.py
```

Or:

```powershell
poetry run python control_osc.py
```

---

## Controller Commands

Examples:

```
tempo 140
bank 1
pause 0

lrev 0.2
lcut 3200
lgain 0.25

rrev 0.6
rcut 600
rgain 0.2
```

Mappings:

- `tempo <float>` → `/tempo`
- `bank <0|1>` → `/bank`
- `pause <0|1>` → `/pause`
- `lrev <0..1>` → `/left/rev`
- `lcut <hz>` → `/left/cutoff`
- `lgain <0..1>` → `/left/gain`
- `rrev <0..1>` → `/right/rev`
- `rcut <hz>` → `/right/cutoff`
- `rgain <0..1>` → `/right/gain`

---

## Network Configuration

Default target:

- IP: `127.0.0.1`
- Port: `9000`

If ChucK runs on another machine:
- Change the IP in `control_osc.py`
- Allow inbound UDP port 9000 in firewall

---

## Recommended .gitignore

Add:

```
.venv/
__pycache__/
*.pyc
```

---

## Quick Start

```powershell
poetry install
Invoke-Expression (poetry env activate)

# Terminal 1
chuck melody_bank_stereo.ck

# Terminal 2
python control_osc.py
```

You now have real-time stereo harmonic control via OSC.
