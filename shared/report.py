"""Result reporting: markdown tables, JSON payloads, and a console that works.

Every project produces the same artefacts — a results table, a JSON file, a
printed summary — so they are built once here rather than 41 times.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

Column = tuple[str, str]


def init_console() -> None:
    """Make ``print`` safe for non-ASCII on a Windows terminal.

    The default Windows console encoding is cp1252, which raises
    ``UnicodeEncodeError`` on characters as ordinary as ``≤`` or ``—``. Files are
    written with an explicit ``encoding="utf-8"`` and are unaffected; it is only
    stdout that needs this, and only on Windows.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except (OSError, ValueError):  # pragma: no cover - exotic terminals
                pass


def markdown_table(
    rows: Sequence[dict], columns: Sequence[Column], missing: str = "n/a"
) -> str:
    """Render result rows as a GitHub-flavoured markdown table.

    ``columns`` is a sequence of ``(header, key)`` pairs. The first column is
    left-aligned (it holds the method name) and the rest are right-aligned, which
    is what makes a column of numbers readable.
    """
    if not columns:
        raise ValueError("markdown_table needs at least one column")

    head = "| " + " | ".join(label for label, _ in columns) + " |"
    rule = "|" + "|".join("---" if i == 0 else "---:" for i in range(len(columns))) + "|"

    body = []
    for r in rows:
        cells = []
        for _, key in columns:
            v = r.get(key)
            if v is None:
                cells.append(missing)
            elif isinstance(v, float):
                cells.append(f"{v:g}")
            else:
                cells.append(str(v))
        body.append("| " + " | ".join(cells) + " |")

    return "\n".join([head, rule, *body])


def write_results(
    results_dir: str | Path, project: str, payload: dict[str, Any]
) -> Path:
    """Write ``results.json`` with provenance stamped in.

    The library versions and the timestamp are recorded because a number without
    the version that produced it cannot be reproduced or challenged.
    """
    import cv2
    import numpy
    import skimage

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    full = {
        "project": project,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {
            "python": sys.version.split()[0],
            "opencv": cv2.__version__,
            "scikit_image": skimage.__version__,
            "numpy": numpy.__version__,
        },
        **payload,
    }
    path = results_dir / "results.json"
    path.write_text(json.dumps(full, indent=2), encoding="utf-8")
    return path


def write_tables(results_dir: str | Path, sections: Sequence[tuple[str, str]]) -> Path:
    """Write ``tables.md`` from ``(heading, markdown_table)`` sections."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    text = "\n\n".join(f"### {heading}\n\n{table}" for heading, table in sections) + "\n"
    path = results_dir / "tables.md"
    path.write_text(text, encoding="utf-8")
    return path
