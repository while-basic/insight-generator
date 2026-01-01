from __future__ import annotations

import hashlib
from typing import Any


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(obj: Any) -> str:
    normalized = repr(obj).encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()
