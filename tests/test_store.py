from pathlib import Path

from chat_recycler.extract.schemas import EventRow, EventType, ExtractorType, MessageRow, Role, VariableRow, VariableType
from chat_recycler.store.sqlite import init_db, write_events, write_messages, write_variables
from chat_recycler.utils.hashing import sha256_text


def test_sqlite_write(tmp_path: Path):
    db_path = tmp_path / "db.sqlite"
    conn = init_db(db_path)
    message = MessageRow(
        conversation_id="c1",
        message_id="m1",
        created_at=None,
        role=Role.user,
        content="hello",
        content_hash=sha256_text("hello"),
        tokens_est=1,
        source_file="test",
        order_index=0,
    )
    event = EventRow(
        conversation_id="c1",
        message_id="m1",
        created_at=None,
        event_type=EventType.intent,
        label="intent",
        event_text="hello",
        entities={},
        confidence=0.5,
        extractor=ExtractorType.heuristic,
        span={},
    )
    variable = VariableRow(
        conversation_id="c1",
        message_id="m1",
        created_at=None,
        var_name="requested_artifact",
        var_type=VariableType.enum,
        var_value="report",
        var_source_text="report",
        confidence=0.5,
        rule_id="rule-1",
        extractor=ExtractorType.heuristic,
    )
    write_messages(conn, [message])
    write_events(conn, [event])
    write_variables(conn, [variable])
    conn.close()
    assert db_path.exists()
