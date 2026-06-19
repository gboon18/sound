# =============================================================================
# Author  : Ho San Ko
# Email   : hko@avalanche.energy
# Project : 4-Channel Music Remix Workstation
# =============================================================================

"""pytest configuration — adds the project root to sys.path so tests can
import from remix_workstation's top-level packages (engine, dsp, loop, …).
"""

import sys
from pathlib import Path

# remix_workstation/ is the project root.  Insert its PARENT so that
# `import engine.mix_bus` resolves correctly when pytest is invoked from
# any directory.
_PROJECT_ROOT = Path(__file__).parent.parent  # → remix_workstation/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
