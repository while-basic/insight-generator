from pathlib import Path

from chat_recycler.ingest.chatgpt_json import parse_chatgpt_json
from chat_recycler.ingest.normalize import normalize_messages


def test_ingest_chatgpt_json():
    path = Path(__file__).parents[1] / "data" / "sample_export" / "conversations.json"
    conversations = parse_chatgpt_json(path)
    assert conversations
    convo = conversations[0]
    rows = normalize_messages(convo["conversation_id"], convo["messages"], source_file=str(path))
    assert len(rows) >= 2
    assert rows[0].conversation_id == "sample-convo-1"
