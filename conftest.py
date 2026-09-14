"""Make ``shared`` and every project's ``src/`` importable from anywhere.

Project directories start with a digit (``01_document_scanner``) so they cannot
be Python packages. Each project therefore keeps its code in ``src/`` under a
**uniquely named module** (``document_scanner.py``, ``portrait_mode.py``, ...),
and every ``src/`` is placed on ``sys.path``. Unique names mean the flat path
can never resolve one project's module in place of another's.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()

_paths = [ROOT] + sorted(ROOT.glob("projects/*/src"))
for _p in _paths:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
