from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

from chat_recycler.extract.schemas import ClusterRow, EventRow, MessageRow, PredictionRow, VariableRow


def init_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            conversation_id TEXT,
            message_id TEXT,
            created_at TEXT,
            role TEXT,
            content TEXT,
            content_hash TEXT,
            tokens_est INTEGER,
            source_file TEXT,
            order_index INTEGER
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            event_id TEXT,
            conversation_id TEXT,
            message_id TEXT,
            created_at TEXT,
            event_type TEXT,
            label TEXT,
            event_text TEXT,
            entities TEXT,
            confidence REAL,
            extractor TEXT,
            span TEXT
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS variables (
            var_id TEXT,
            conversation_id TEXT,
            message_id TEXT,
            created_at TEXT,
            var_name TEXT,
            var_type TEXT,
            var_value TEXT,
            var_source_text TEXT,
            confidence REAL,
            rule_id TEXT,
            extractor TEXT
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS clusters (
            cluster_id TEXT,
            conversation_id TEXT,
            cluster_label TEXT,
            cluster_summary TEXT,
            centroid_repr TEXT,
            method TEXT,
            confidence REAL
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id TEXT,
            conversation_id TEXT,
            target TEXT,
            prediction TEXT,
            model TEXT,
            features_used TEXT,
            confidence REAL
        )
        """
    )
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_convo ON messages(conversation_id, created_at)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_convo ON events(conversation_id, event_type)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_vars_convo ON variables(conversation_id, var_name)")
    conn.commit()
    return conn


def _insert_many(conn: sqlite3.Connection, query: str, rows: Iterable[tuple]) -> None:
    conn.executemany(query, list(rows))
    conn.commit()


def write_messages(conn: sqlite3.Connection, messages: Iterable[MessageRow]) -> None:
    _insert_many(
        conn,
        """
        INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            (
                m.conversation_id,
                m.message_id,
                m.created_at.isoformat() if m.created_at else None,
                m.role.value,
                m.content,
                m.content_hash,
                m.tokens_est,
                m.source_file,
                m.order_index,
            )
            for m in messages
        ),
    )


def write_events(conn: sqlite3.Connection, events: Iterable[EventRow]) -> None:
    _insert_many(
        conn,
        """
        INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            (
                e.event_id,
                e.conversation_id,
                e.message_id,
                e.created_at.isoformat() if e.created_at else None,
                e.event_type.value,
                e.label,
                e.event_text,
                str(e.entities),
                e.confidence,
                e.extractor.value,
                str(e.span),
            )
            for e in events
        ),
    )


def write_variables(conn: sqlite3.Connection, variables: Iterable[VariableRow]) -> None:
    _insert_many(
        conn,
        """
        INSERT INTO variables VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            (
                v.var_id,
                v.conversation_id,
                v.message_id,
                v.created_at.isoformat() if v.created_at else None,
                v.var_name,
                v.var_type.value,
                str(v.var_value),
                v.var_source_text,
                v.confidence,
                v.rule_id,
                v.extractor.value,
            )
            for v in variables
        ),
    )


def write_clusters(conn: sqlite3.Connection, clusters: Iterable[ClusterRow]) -> None:
    _insert_many(
        conn,
        """
        INSERT INTO clusters VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            (
                c.cluster_id,
                c.conversation_id,
                c.cluster_label,
                c.cluster_summary,
                c.centroid_repr,
                c.method.value,
                c.confidence,
            )
            for c in clusters
        ),
    )


def write_predictions(conn: sqlite3.Connection, predictions: Iterable[PredictionRow]) -> None:
    _insert_many(
        conn,
        """
        INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            (
                p.prediction_id,
                p.conversation_id,
                p.target.value,
                str(p.prediction),
                p.model,
                str(p.features_used),
                p.confidence,
            )
            for p in predictions
        ),
    )
