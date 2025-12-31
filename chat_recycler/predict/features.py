from __future__ import annotations

from collections import Counter
from typing import Dict, List

import pandas as pd

from chat_recycler.extract.schemas import EventRow, MessageRow, VariableRow


SENTIMENT_LEXICON = {
    "good": 1,
    "great": 2,
    "success": 2,
    "works": 1,
    "bad": -1,
    "error": -2,
    "fail": -2,
    "issue": -1,
}


def build_feature_frame(
    messages: List[MessageRow],
    events: List[EventRow],
    variables: List[VariableRow],
) -> pd.DataFrame:
    convo_ids = {message.conversation_id for message in messages}
    rows: List[Dict[str, float]] = []
    for convo_id in convo_ids:
        convo_messages = [m for m in messages if m.conversation_id == convo_id]
        convo_events = [e for e in events if e.conversation_id == convo_id]
        convo_vars = [v for v in variables if v.conversation_id == convo_id]
        text = " ".join(m.content for m in convo_messages)
        sentiment = sum(SENTIMENT_LEXICON.get(word.lower(), 0) for word in text.split())
        event_counts = Counter(e.event_type for e in convo_events)
        var_names = {v.var_name for v in convo_vars}
        row = {
            "conversation_id": convo_id,
            "message_count": float(len(convo_messages)),
            "char_count": float(sum(len(m.content) for m in convo_messages)),
            "sentiment": float(sentiment),
        }
        for event_type, count in event_counts.items():
            row[f"event_{event_type}"] = float(count)
        row["has_constraints"] = float("constraint" in {e.event_type for e in convo_events})
        row["has_artifact_request"] = float("requested_artifact" in var_names)
        rows.append(row)
    return pd.DataFrame(rows).fillna(0.0)
