from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

from chat_recycler.extract.schemas import ClusterRow, EventRow, MessageRow, VariableRow


def write_index(path: Path, convo_ids: Iterable[str]) -> None:
    lines = ["# Chat Recycler Report", "", "## Conversations"]
    for convo_id in convo_ids:
        lines.append(f"- [{convo_id}](convos/{convo_id}.md)")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_convo_report(
    path: Path,
    messages: List[MessageRow],
    events: List[EventRow],
    variables: List[VariableRow],
) -> None:
    convo_id = messages[0].conversation_id if messages else "unknown"
    lines = [f"# Conversation {convo_id}", "", "## Summary", ""]
    lines.append(f"Messages: {len(messages)}")
    lines.append(f"Events: {len(events)}")
    lines.append(f"Variables: {len(variables)}")
    lines.append("\n## Signals")
    for event in events:
        lines.append(f"- {event.event_type.value}: {event.label} ({event.event_text})")
    lines.append("\n## Variables")
    for var in variables:
        lines.append(f"- {var.var_name} = {var.var_value} ({var.var_source_text})")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_clusters_report(path: Path, clusters: List[ClusterRow]) -> None:
    lines = ["# Cluster Themes", ""]
    for cluster in clusters:
        lines.append(f"## {cluster.cluster_label}")
        lines.append(cluster.cluster_summary)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_metrics_report(path: Path, events: List[EventRow], variables: List[VariableRow]) -> None:
    event_types = Counter(event.event_type.value for event in events)
    var_names = Counter(var.var_name for var in variables)
    lines = ["# Metrics", "", "## Event Types"]
    for name, count in event_types.items():
        lines.append(f"- {name}: {count}")
    lines.append("\n## Top Variables")
    for name, count in var_names.most_common(10):
        lines.append(f"- {name}: {count}")
    path.write_text("\n".join(lines), encoding="utf-8")
