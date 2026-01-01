from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class ConversationSummary:
    conversation_id: str
    message_count: int
    event_count: int
    variable_count: int


def _connect(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(db_path)


def fetch_conversations(db_path: Path) -> List[ConversationSummary]:
    if not db_path.exists():
        return []
    conn = _connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT conversation_id,
               COUNT(*) AS message_count
        FROM messages
        GROUP BY conversation_id
        """
    )
    message_counts = {row[0]: row[1] for row in cursor.fetchall()}
    cursor.execute(
        """
        SELECT conversation_id,
               COUNT(*) AS event_count
        FROM events
        GROUP BY conversation_id
        """
    )
    event_counts = {row[0]: row[1] for row in cursor.fetchall()}
    cursor.execute(
        """
        SELECT conversation_id,
               COUNT(*) AS variable_count
        FROM variables
        GROUP BY conversation_id
        """
    )
    variable_counts = {row[0]: row[1] for row in cursor.fetchall()}
    conn.close()
    summaries: List[ConversationSummary] = []
    for convo_id, message_count in message_counts.items():
        summaries.append(
            ConversationSummary(
                conversation_id=convo_id,
                message_count=message_count,
                event_count=event_counts.get(convo_id, 0),
                variable_count=variable_counts.get(convo_id, 0),
            )
        )
    return sorted(summaries, key=lambda item: item.conversation_id)


def fetch_messages(db_path: Path, conversation_id: str) -> List[sqlite3.Row]:
    if not db_path.exists():
        return []
    conn = _connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT message_id, role, content, created_at
        FROM messages
        WHERE conversation_id = ?
        ORDER BY order_index
        """,
        (conversation_id,),
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


def fetch_events(db_path: Path, conversation_id: str) -> List[sqlite3.Row]:
    if not db_path.exists():
        return []
    conn = _connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT event_type, label, event_text, confidence
        FROM events
        WHERE conversation_id = ?
        ORDER BY created_at
        """,
        (conversation_id,),
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


def fetch_variables(db_path: Path, conversation_id: str) -> List[sqlite3.Row]:
    if not db_path.exists():
        return []
    conn = _connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT var_name, var_value, var_source_text, confidence
        FROM variables
        WHERE conversation_id = ?
        ORDER BY created_at
        """,
        (conversation_id,),
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


def fetch_clusters(db_path: Path) -> List[sqlite3.Row]:
    if not db_path.exists():
        return []
    conn = _connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT cluster_label, conversation_id, cluster_summary, centroid_repr, confidence
        FROM clusters
        ORDER BY cluster_label, conversation_id
        """
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


def fetch_metrics(db_path: Path) -> dict:
    if not db_path.exists():
        return {"messages": 0, "events": 0, "variables": 0}
    conn = _connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM messages")
    messages = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM events")
    events = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM variables")
    variables = cursor.fetchone()[0]
    conn.close()
    return {"messages": messages, "events": events, "variables": variables}
