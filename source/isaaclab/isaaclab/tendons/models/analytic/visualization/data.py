"""Input/output helpers for tendon visualization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load frame-by-frame tendon debug data from a JSONL file."""
    path = Path(path)
    with path.open("r") as f:
        return [json.loads(line) for line in f if line.strip()]
