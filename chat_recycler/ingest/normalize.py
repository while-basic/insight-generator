from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from chat_recycler.extract.schemas import MessageRow, Role
from chat_recycler.utils.hashing import sha256_text


def _parse_datetime(value: Optional[float | str]) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value))
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def normalize_messages(
    conversation_id: str,
    messages: List[Dict[str, str]],
    source_file: str,
) -> List[MessageRow]:
    rows: List[MessageRow] = []
    for idx, msg in enumerate(messages):
        content = str(msg.get("content") or "").strip()
        if not content:
            continue
        created_at = _parse_datetime(msg.get("created_at"))
        role_value = str(msg.get("role") or "user")
        try:
            role = Role(role_value)
        except ValueError:
            role = Role.user
        content_hash = sha256_text(content)
        tokens_est = max(1, len(content) // 4)
        rows.append(
            MessageRow(
                conversation_id=conversation_id,
                message_id=str(msg.get("message_id") or f"{conversation_id}-{idx}"),
                created_at=created_at,
                role=role,
                content=content,
                content_hash=content_hash,
                tokens_est=tokens_est,
                source_file=source_file,
                order_index=idx,
            )
        )
    return rows
