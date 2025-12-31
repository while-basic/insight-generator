from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Tuple

from chat_recycler.extract.schemas import (
    EventRow,
    EventType,
    ExtractorType,
    MessageRow,
    VariableRow,
    VariableType,
)


def _compile_patterns(patterns: Iterable[str]) -> List[re.Pattern[str]]:
    compiled: List[re.Pattern[str]] = []
    for pattern in patterns:
        compiled.append(re.compile(pattern, re.IGNORECASE))
    return compiled


def _match_patterns(patterns: List[re.Pattern[str]], text: str) -> List[re.Match[str]]:
    matches: List[re.Match[str]] = []
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            matches.append(match)
    return matches


def _span_from_match(match: re.Match[str]) -> Dict[str, int]:
    return {"start": match.start(), "end": match.end()}


def extract_from_rules(
    messages: Iterable[MessageRow],
    rules: Dict[str, Any],
) -> Tuple[List[EventRow], List[VariableRow]]:
    event_rows: List[EventRow] = []
    variable_rows: List[VariableRow] = []
    for message in messages:
        text = message.content
        for rule in rules.get("rules", []):
            patterns = rule.get("patterns") or []
            if not patterns:
                continue
            matches = _match_patterns(_compile_patterns(patterns), text)
            if not matches:
                continue
            span = _span_from_match(matches[0]) if matches else {}
            if rule.get("kind") == "event":
                event_rows.append(
                    EventRow(
                        conversation_id=message.conversation_id,
                        message_id=message.message_id,
                        created_at=message.created_at,
                        event_type=EventType(rule.get("event_type")),
                        label=rule.get("label", rule.get("event_type")),
                        event_text=matches[0].group(0),
                        entities=rule.get("entities") or {},
                        confidence=float(rule.get("confidence", 0.6)),
                        extractor=ExtractorType.heuristic,
                        span=span,
                    )
                )
            elif rule.get("kind") == "variable":
                variable_rows.append(
                    VariableRow(
                        conversation_id=message.conversation_id,
                        message_id=message.message_id,
                        created_at=message.created_at,
                        var_name=rule.get("var_name"),
                        var_type=VariableType(rule.get("var_type")),
                        var_value=rule.get("var_value"),
                        var_source_text=matches[0].group(0),
                        confidence=float(rule.get("confidence", 0.6)),
                        rule_id=rule.get("id", "unknown"),
                        extractor=ExtractorType.heuristic,
                    )
                )
    return event_rows, variable_rows
