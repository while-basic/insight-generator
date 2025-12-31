from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def write_json(path: Path, rows: Iterable[object]) -> None:
    data = [row.model_dump() for row in rows]
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
