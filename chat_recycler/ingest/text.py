from __future__ import annotations

from pathlib import Path
from typing import Dict, List


def parse_text_file(path: Path) -> List[Dict[str, str]]:
    content = path.read_text(encoding="utf-8")
    return [
        {
            "conversation_id": path.stem,
            "messages": [
                {
                    "message_id": f"{path.stem}-0",
                    "role": "user",
                    "content": content,
                    "created_at": None,
                }
            ],
        }
    ]
