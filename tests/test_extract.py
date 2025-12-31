from chat_recycler.extract.heuristic import extract_from_rules
from chat_recycler.extract.rules_loader import load_rules
from chat_recycler.extract.schemas import MessageRow, Role
from chat_recycler.utils.hashing import sha256_text


def test_heuristic_extraction_rules():
    rules = load_rules(
        __import__("pathlib").Path(__file__).parents[1]
        / "chat_recycler"
        / "rules"
        / "default_rules.yaml"
    )
    message = MessageRow(
        conversation_id="c1",
        message_id="m1",
        created_at=None,
        role=Role.user,
        content="Make a report with a checklist for local-first M1 16GB. It works but error appears.",
        content_hash=sha256_text("test"),
        tokens_est=10,
        source_file="test",
        order_index=0,
    )
    events, variables = extract_from_rules([message], rules)
    assert len(events) >= 5
    assert any(var.var_name == "requested_artifact" for var in variables)
