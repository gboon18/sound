# CAMP-## Analysis Environment (Windows + VS Code + Poetry + Jupyter) — Procedure

This repo uses **Poetry** for dependency management and a **project-local virtual environment** (`.venv`) so that:
- the project always runs on **Python 3.12.x**
- VS Code / Jupyter reliably detects the correct interpreter
- future “fresh starts” are repeatable

This document is written for **Windows + PowerShell + VS Code**.

---

## Requirements

- **pyenv-win** installed and working (you already have it)
- **Poetry** installed (`poetry --version` works)
- VS Code with extensions:
  - **Python** (ms-python.python)
  - **Jupyter** (ms-toolsai.jupyter)

---

## Standard Layout (per project)

Project root contains:
- `pyproject.toml`
- `poetry.lock`
- `.python-version`  (pins Python for this repo; created by `pyenv local`)
- `.venv/`           (created by Poetry; project-local environment)

---

## One-Time Setup (per repo)

> Run all commands from the **project root** (folder containing `pyproject.toml`).

### 0) Close VS Code + stop all kernels (avoid locks)
1. In VS Code: stop any running notebook kernels.
2. Close VS Code completely (recommended for the reset).
3. Close terminals that are inside this repo.

### 1) Kill Python processes (blunt reset)
```
taskkill /F /IM python.exe 2>$null
taskkill /F /IM pythonw.exe 2>$null
````

(Optional) If you suspect Jupyter is running as a separate process:

```
taskkill /F /IM jupyter.exe 2>$null
```

### 2) Ensure Python 3.12.x exists via pyenv-win

```
pyenv install 3.12.10
```

If it says “already installed”, that’s fine.

### 3) Pin this repo to Python 3.12.x (creates `.python-version`)

```
pyenv local 3.12.10
python --version
```

Expected: `Python 3.12.10`

This ensures that whenever you open a terminal **in this folder**, `python` resolves to 3.12.10.

---

## Full Clean Reset (recommended when things get weird)

### 4) Remove Poetry-managed environments for this project

```
poetry env list --full-path
```

Remove all envs Poetry has for this project:

```
poetry env remove --all
```

If you get `WinError 5` or `WinError 32`, something is still locking the env:

* close VS Code
* kill python/pythonw again
* retry `poetry env remove --all`

### 5) Remove project-local `.venv` if it exists

```
Remove-Item -Recurse -Force .\.venv -ErrorAction SilentlyContinue
```

---

## Best-Practice Configuration (local to this repo)

### 6) Force project-local virtualenv (write to repo-local Poetry config)

```
poetry config virtualenvs.in-project true --local
poetry config virtualenvs.create true --local
```

Verify:

```
poetry config --local --list | findstr virtualenvs
```

Expected:

* `virtualenvs.in-project = true`
* `virtualenvs.create = true`

---

## Create the `.venv` and Install Dependencies

### 7) Create env using the pinned Python 3.12 interpreter

```
poetry env use (Get-Command python).Path
```

### 8) Install dependencies

```
poetry install
```

### 9) Verify `.venv` exists and is being used

```
dir .venv
poetry run python -c "import sys; print(sys.version); print(sys.executable)"
```

Expected:

* `.venv\` directory exists in project root
* `sys.executable` is:
  `...\<repo>\.venv\Scripts\python.exe`

---

## Add Jupyter Support (required for notebooks)

### 10) Install ipykernel into the Poetry env

```
poetry add --group dev ipykernel
```

(If you need Excel reads via pandas)

```
poetry add openpyxl
```

Verify:

```
poetry run python -c "import openpyxl; print(openpyxl.__version__)"
```

---

## VS Code: Make Jupyter Detect the Correct Environment

### 11) Select the interpreter explicitly (once)

In VS Code:

* `Ctrl+Shift+P` → **Python: Select Interpreter**
* Choose the interpreter at:
  `...\<repo>\.venv\Scripts\python.exe`

### 12) Select Kernel in notebook

In the notebook UI:

* **Select Kernel** → choose the `.venv` Python 3.12 interpreter.

### 13) Verify in a notebook cell

```
import sys
print(sys.version)
print(sys.executable)
```

Expected:

* Python `3.12.10`
* executable path contains `.venv\Scripts\python.exe`

---

## Common Issues

### A) Poetry says “currently activated Python 3.13 is not supported…”

Your shell default is 3.13. That’s fine as long as the repo is pinned to 3.12 and Poetry env is 3.12.
To eliminate confusion:

* ensure `pyenv local 3.12.10` is set in this repo
* open a NEW terminal in the repo
* confirm `python --version` is 3.12.10

### B) `poetry env remove ...` fails with Access Denied (WinError 5)

The env is locked by a running Python/Jupyter process.
Fix:

1. Shut down notebook kernels
2. Close VS Code
3. Kill python processes:

   ```
   taskkill /F /IM python.exe
   taskkill /F /IM pythonw.exe
   ```
4. Retry removal

### C) `.venv` doesn’t appear after `poetry install`

Usually Poetry reused a cached env or the local config wasn’t applied.
Fix:

* confirm local config:

  ```
  poetry config --local --list | findstr virtualenvs
  ```
* remove envs:

  ```
  poetry env remove --all
  Remove-Item -Recurse -Force .\.venv -ErrorAction SilentlyContinue
  ```
* rerun:

  ```
  poetry env use (Get-Command python).Path
  poetry install
  ```

---

## Daily Workflow (after setup)

From project root:

```
poetry run python your_script.py
```

In notebooks:

* always use the `.venv` kernel

---

## “Fresh clone” checklist (new machine / new folder)

1. Install pyenv-win + Poetry + VS Code extensions
2. From repo root:

   ```
   pyenv install 3.12.10
   pyenv local 3.12.10
   poetry config virtualenvs.in-project true --local
   poetry config virtualenvs.create true --local
   poetry env use (Get-Command python).Path
   poetry install
   poetry add --group dev ipykernel
   poetry add openpyxl
   ```
3. In VS Code select:
   `.\.venv\Scripts\python.exe`
4. Notebook cell check:
   `sys.version` and `sys.executable`

---

```
```

````
### Emergency cleanup: Poetry cache env is locked / access denied

If `poetry env remove --all` or `poetry env remove <path>` fails with `WinError 5` (Access is denied) or `WinError 32` (in use):

1. Shut down VS Code notebook kernels and close VS Code.
2. Kill Python processes:
   ```
   taskkill /F /IM python.exe 2>$null
   taskkill /F /IM pythonw.exe 2>$null
````

3. Try again (preferred):

   ```
   poetry env list --full-path
   poetry env remove "<ENV_PATH_FROM_LIST>"
   ```

4. If Poetry still cannot remove it, delete the env folder manually (last resort):

   ```
   Remove-Item -Recurse -Force "C:\Users\hko\AppData\Local\pypoetry\Cache\virtualenvs\analysis-cZUt14xa-py3.12"
   ```

Notes:

* This manual delete is safe if you are sure it’s the correct venv folder and no process is using it.
* After deleting, run `poetry env remove --all` again to clear stale references, then recreate the env.

```

Rationale: manual deletion bypasses Poetry’s metadata, so it should be used only when Poetry cannot remove the environment.
```

