from __future__ import annotations

import sys
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    for candidate in start.parents:
        if (candidate / "python" / "stratax").exists():
            return candidate

    raise RuntimeError("Unable to locate repository root from tests/python")


ROOT = _find_repo_root(Path(__file__).resolve())
sys.path.insert(0, str(ROOT / "python"))
