from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_rules(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
