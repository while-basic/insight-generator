from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import typer

from chat_recycler import __version__
from chat_recycler.cluster.tfidf_kmeans import cluster_conversations
from chat_recycler.config import PipelineConfig
from chat_recycler.extract.heuristic import extract_from_rules
from chat_recycler.extract.llm_ollama import OllamaError, extract_with_ollama
from chat_recycler.extract.rules_loader import load_rules
from chat_recycler.extract.schemas import EventRow, MessageRow, VariableRow
from chat_recycler.ingest.chatgpt_json import parse_chatgpt_json
from chat_recycler.ingest.normalize import normalize_messages
from chat_recycler.ingest.text import parse_text_file
from chat_recycler.predict.features import build_feature_frame
from chat_recycler.predict.infer import generate_predictions
from chat_recycler.predict.train import train_models
from chat_recycler.report.markdown import (
    write_clusters_report,
    write_convo_report,
    write_index,
    write_metrics_report,
)
from chat_recycler.store.parquet import (
    write_clusters as write_clusters_parquet,
    write_events as write_events_parquet,
    write_messages as write_messages_parquet,
    write_predictions as write_predictions_parquet,
    write_variables as write_variables_parquet,
)
from chat_recycler.store.json_store import write_json
from chat_recycler.store.sqlite import (
    init_db,
    write_clusters as write_clusters_db,
    write_events as write_events_db,
    write_messages as write_messages_db,
    write_predictions as write_predictions_db,
    write_variables as write_variables_db,
)
from chat_recycler.utils.chunking import chunk_text
from chat_recycler.utils.hashing import sha256_text
from chat_recycler.utils.logging import log_error, log_info
from chat_recycler.utils.time import now_utc

app = typer.Typer(add_completion=False)


def _ensure_dirs(output: Path) -> Dict[str, Path]:
    dirs = {
        "root": output,
        "exports": output / "exports",
        "reports": output / "reports",
        "convos": output / "reports" / "convos",
        "meta": output / "meta",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _log_error(meta_dir: Path, error: Dict[str, str]) -> None:
    error_path = meta_dir / "errors.jsonl"
    with error_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(error) + "\n")


def _detect_format(path: Path, input_format: str) -> str:
    if input_format != "auto":
        return input_format
    if path.suffix.lower() == ".json":
        return "chatgpt_json"
    return "text"


def ingest(config: PipelineConfig) -> List[MessageRow]:
    dirs = _ensure_dirs(config.output_path)
    input_path = config.input_path
    files = [input_path] if input_path.is_file() else list(input_path.glob("**/*"))
    messages: List[MessageRow] = []
    for file_path in files:
        if file_path.is_dir():
            continue
        try:
            detected = _detect_format(file_path, config.input_format)
            if detected == "chatgpt_json":
                conversations = parse_chatgpt_json(file_path)
            else:
                conversations = parse_text_file(file_path)
            for convo in conversations:
                messages.extend(
                    normalize_messages(
                        convo["conversation_id"],
                        convo["messages"],
                        source_file=str(file_path),
                    )
                )
        except Exception as exc:  # noqa: BLE001
            _log_error(dirs["meta"], {"stage": "ingest", "file": str(file_path), "error": str(exc)})
            log_error(f"Failed to ingest {file_path}: {exc}")
    if config.db == "sqlite":
        conn = init_db(config.output_path / "db.sqlite")
        write_messages_db(conn, messages)
        conn.close()
    if config.export_format in {"parquet", "both"}:
        write_messages_parquet(dirs["exports"] / "conversations.parquet", messages)
    if config.export_format in {"json", "both"}:
        write_json(dirs["exports"] / "conversations.json", messages)
    return messages


def extract(config: PipelineConfig, messages: List[MessageRow]) -> tuple[list[EventRow], list[VariableRow], dict]:
    dirs = _ensure_dirs(config.output_path)
    rules = load_rules(Path(__file__).parent / "rules" / "default_rules.yaml")
    llm_receipts: List[dict] = []
    events: List[EventRow] = []
    variables: List[VariableRow] = []
    if config.use_llm:
        grouped: Dict[str, List[MessageRow]] = {}
        for message in messages:
            grouped.setdefault(message.conversation_id, []).append(message)
        for convo_id, convo_messages in grouped.items():
            text = "\n".join(msg.content for msg in convo_messages)
            chunks = chunk_text(text, config.max_chars_per_chunk)
            for chunk in chunks:
                try:
                    chunk_events, chunk_vars, receipt = extract_with_ollama(chunk, config.ollama_model)
                    events.extend(chunk_events or [])
                    variables.extend(chunk_vars or [])
                    llm_receipts.append({"conversation_id": convo_id, **receipt})
                except (OllamaError, ValueError, json.JSONDecodeError) as exc:
                    _log_error(dirs["meta"], {"stage": "extract", "conversation_id": convo_id, "error": str(exc)})
                    fallback_events, fallback_vars = extract_from_rules(convo_messages, rules)
                    events.extend(fallback_events)
                    variables.extend(fallback_vars)
    else:
        events, variables = extract_from_rules(messages, rules)

    if config.db == "sqlite":
        conn = init_db(config.output_path / "db.sqlite")
        write_events_db(conn, events)
        write_variables_db(conn, variables)
        conn.close()
    if config.export_format in {"parquet", "both"}:
        write_events_parquet(dirs["exports"] / "events.parquet", events)
        write_variables_parquet(dirs["exports"] / "variables.parquet", variables)
    if config.export_format in {"json", "both"}:
        write_json(dirs["exports"] / "events.json", events)
        write_json(dirs["exports"] / "variables.json", variables)
    return events, variables, {"llm_receipts": llm_receipts}


def cluster(config: PipelineConfig, messages: List[MessageRow]) -> List:
    dirs = _ensure_dirs(config.output_path)
    clusters = cluster_conversations(messages)
    if config.db == "sqlite":
        conn = init_db(config.output_path / "db.sqlite")
        write_clusters_db(conn, clusters)
        conn.close()
    if config.export_format in {"parquet", "both"}:
        write_clusters_parquet(dirs["exports"] / "clusters.parquet", clusters)
    if config.export_format in {"json", "both"}:
        write_json(dirs["exports"] / "clusters.json", clusters)
    return clusters


def predict(
    config: PipelineConfig,
    messages: List[MessageRow],
    events: List[EventRow],
    variables: List[VariableRow],
) -> List:
    dirs = _ensure_dirs(config.output_path)
    features = build_feature_frame(messages, events, variables)
    labels_path = config.input_path / "labels.json" if config.input_path.is_dir() else config.input_path.parent / "labels.json"
    labels = None
    if labels_path.exists():
        labels = pd.read_json(labels_path)
    models = train_models(features, labels)
    predictions = generate_predictions(features, models)
    if config.db == "sqlite":
        conn = init_db(config.output_path / "db.sqlite")
        write_predictions_db(conn, predictions)
        conn.close()
    if config.export_format in {"parquet", "both"}:
        write_predictions_parquet(dirs["exports"] / "predictions.parquet", predictions)
    if config.export_format in {"json", "both"}:
        write_json(dirs["exports"] / "predictions.json", predictions)
    return predictions


def report(
    config: PipelineConfig,
    messages: List[MessageRow],
    events: List[EventRow],
    variables: List[VariableRow],
    clusters: List,
) -> None:
    dirs = _ensure_dirs(config.output_path)
    grouped_messages: Dict[str, List[MessageRow]] = {}
    grouped_events: Dict[str, List[EventRow]] = {}
    grouped_vars: Dict[str, List[VariableRow]] = {}
    for message in messages:
        grouped_messages.setdefault(message.conversation_id, []).append(message)
    for event in events:
        grouped_events.setdefault(event.conversation_id, []).append(event)
    for var in variables:
        grouped_vars.setdefault(var.conversation_id, []).append(var)
    for convo_id, convo_messages in grouped_messages.items():
        write_convo_report(
            dirs["convos"] / f"{convo_id}.md",
            convo_messages,
            grouped_events.get(convo_id, []),
            grouped_vars.get(convo_id, []),
        )
    write_index(dirs["reports"] / "INDEX.md", grouped_messages.keys())
    write_clusters_report(dirs["reports"] / "clusters.md", clusters)
    write_metrics_report(dirs["reports"] / "metrics.md", events, variables)


def write_meta(config: PipelineConfig, messages: List[MessageRow], receipts: Dict[str, list]) -> None:
    dirs = _ensure_dirs(config.output_path)
    run_data = {
        "version": __version__,
        "timestamp": now_utc().isoformat(),
        "config": config.__dict__,
        "hashes": {
            "messages": sha256_text("".join(m.content_hash for m in messages)),
        },
        "llm_receipts": receipts.get("llm_receipts", []),
    }
    (dirs["meta"] / "run.json").write_text(json.dumps(run_data, indent=2), encoding="utf-8")

    schema = {
        "MessageRow": MessageRow.model_json_schema(),
        "EventRow": EventRow.model_json_schema(),
        "VariableRow": VariableRow.model_json_schema(),
    }
    (dirs["meta"] / "schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")


@app.command()
def ingest_cmd(
    input_path: Path = typer.Option(..., "--in", help="Input file or directory"),
    output_path: Path = typer.Option("out", "--out"),
    input_format: str = typer.Option("auto", "--format"),
    db: str = typer.Option("sqlite", "--db"),
    export_format: str = typer.Option("both", "--export"),
) -> None:
    config = PipelineConfig(input_path, output_path, input_format=input_format, db=db, export_format=export_format)
    ingest(config)
    log_info("Ingest complete")


@app.command()
def extract_cmd(
    input_path: Path = typer.Option(..., "--in", help="Input file or directory"),
    output_path: Path = typer.Option("out", "--out"),
    use_llm: bool = typer.Option(False, "--use-llm"),
    ollama_model: str = typer.Option("qwen2.5:7b", "--ollama-model"),
    max_chars_per_chunk: int = typer.Option(6000, "--max-chars-per-chunk"),
    db: str = typer.Option("sqlite", "--db"),
    export_format: str = typer.Option("both", "--export"),
) -> None:
    config = PipelineConfig(
        input_path,
        output_path,
        use_llm=use_llm,
        ollama_model=ollama_model,
        max_chars_per_chunk=max_chars_per_chunk,
        db=db,
        export_format=export_format,
    )
    messages = ingest(config)
    extract(config, messages)
    log_info("Extract complete")


@app.command()
def cluster_cmd(
    input_path: Path = typer.Option(..., "--in", help="Input file or directory"),
    output_path: Path = typer.Option("out", "--out"),
    db: str = typer.Option("sqlite", "--db"),
    export_format: str = typer.Option("both", "--export"),
) -> None:
    config = PipelineConfig(input_path, output_path, db=db, export_format=export_format)
    messages = ingest(config)
    cluster(config, messages)
    log_info("Cluster complete")


@app.command()
def predict_cmd(
    input_path: Path = typer.Option(..., "--in", help="Input file or directory"),
    output_path: Path = typer.Option("out", "--out"),
    db: str = typer.Option("sqlite", "--db"),
    export_format: str = typer.Option("both", "--export"),
) -> None:
    config = PipelineConfig(input_path, output_path, db=db, export_format=export_format)
    messages = ingest(config)
    events, variables, _ = extract(config, messages)
    predict(config, messages, events, variables)
    log_info("Predict complete")


@app.command()
def report_cmd(
    input_path: Path = typer.Option(..., "--in", help="Input file or directory"),
    output_path: Path = typer.Option("out", "--out"),
    db: str = typer.Option("sqlite", "--db"),
    export_format: str = typer.Option("both", "--export"),
) -> None:
    config = PipelineConfig(input_path, output_path, db=db, export_format=export_format)
    messages = ingest(config)
    events, variables, _ = extract(config, messages)
    clusters = cluster(config, messages)
    report(config, messages, events, variables, clusters)
    log_info("Report complete")


@app.command()
def run(
    input_path: Path = typer.Option(..., "--in", help="Input file or directory"),
    output_path: Path = typer.Option("out", "--out"),
    input_format: str = typer.Option("auto", "--format"),
    use_llm: bool = typer.Option(False, "--use-llm"),
    ollama_model: str = typer.Option("qwen2.5:7b", "--ollama-model"),
    max_chars_per_chunk: int = typer.Option(6000, "--max-chars-per-chunk"),
    db: str = typer.Option("sqlite", "--db"),
    export_format: str = typer.Option("both", "--export"),
    report_flag: bool = typer.Option(True, "--report"),
) -> None:
    config = PipelineConfig(
        input_path,
        output_path,
        input_format=input_format,
        use_llm=use_llm,
        ollama_model=ollama_model,
        max_chars_per_chunk=max_chars_per_chunk,
        db=db,
        export_format=export_format,
        report=report_flag,
    )
    messages = ingest(config)
    events, variables, receipts = extract(config, messages)
    clusters = cluster(config, messages)
    predictions = predict(config, messages, events, variables)
    if config.report:
        report(config, messages, events, variables, clusters)
    write_meta(config, messages, receipts)
    log_info(f"Run complete. Predictions: {len(predictions)}")


if __name__ == "__main__":
    app()
