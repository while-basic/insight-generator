from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from chat_recycler.utils.logging import log_warning


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _extract_messages_from_mapping(mapping: Dict[str, Any]) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    for node in mapping.values():
        message = node.get("message") or {}
        author = message.get("author") or {}
        role = author.get("role")
        content = message.get("content") or {}
        parts = content.get("parts")
        if not parts:
            continue
        text = "\n".join(str(part) for part in parts if part is not None).strip()
        if not text:
            continue
        messages.append(
            {
                "message_id": message.get("id") or node.get("id"),
                "role": role or "user",
                "content": text,
                "created_at": message.get("create_time"),
            }
        )
    messages.sort(key=lambda item: item.get("created_at") or 0)
    return messages


def parse_chatgpt_json(path: Path) -> List[Dict[str, Any]]:
    payload = _load_json(path)
    conversations: List[Dict[str, Any]] = []
    if isinstance(payload, list):
        items = payload
    else:
        items = [payload]

    for convo in items:
        convo_id = convo.get("id") or convo.get("conversation_id") or path.stem
        mapping = convo.get("mapping")
        if mapping:
            messages = _extract_messages_from_mapping(mapping)
        else:
            raw_messages = convo.get("messages") or convo.get("conversation") or []
            messages = []
            for msg in raw_messages:
                content = msg.get("content") or ""
                if isinstance(content, dict):
                    parts = content.get("parts") or []
                    content = "\n".join(str(part) for part in parts if part is not None)
                messages.append(
                    {
                        "message_id": msg.get("id") or msg.get("message_id"),
                        "role": msg.get("role") or msg.get("author") or "user",
                        "content": str(content),
                        "created_at": msg.get("create_time"),
                    }
                )
            messages = [msg for msg in messages if msg.get("content")]
            messages.sort(key=lambda item: item.get("created_at") or 0)
        if not messages:
            log_warning(f"No messages found in {path}")
            continue
        conversations.append(
            {
                "conversation_id": str(convo_id),
                "messages": messages,
            }
        )
    return conversations
