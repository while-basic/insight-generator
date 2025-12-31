# chat-recycler

Local-first Python system that recycles exported ChatGPT conversations into structured datasets for signal mining, variable extraction, clustering, and baseline prediction.

## Features

- Deterministic heuristic extraction with optional local Ollama LLM path
- Typed Pydantic schemas with JSON schema export
- SQLite + Parquet exports
- Cluster summaries + baseline predictions
- CLI-first workflows with an importable library

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quickstart

```bash
chat-recycler run --in data/sample_export --out out
```

Outputs:

```
out/
  db.sqlite
  exports/
    conversations.parquet
    events.parquet
    variables.parquet
    clusters.parquet
    predictions.parquet
  reports/
    INDEX.md
    clusters.md
    metrics.md
    convos/
      <conversation_id>.md
  meta/
    run.json
    schema.json
    errors.jsonl
```

## CLI

```bash
chat-recycler ingest --in data/export --out out/
chat-recycler extract --in out/ --use-llm false
chat-recycler cluster --in out/
chat-recycler predict --in out/
chat-recycler report --in out/
chat-recycler run --in data/export --out out/
```

### LLM extractor (optional)

```bash
chat-recycler run --in data/export --out out --use-llm true --ollama-model qwen2.5:7b
```

## Minimal sample input

`data/sample_export/conversations.json` contains a minimal ChatGPT export example.

## Troubleshooting

- If Ollama is unavailable, `--use-llm true` will fall back to heuristic extraction and log an error in `meta/errors.jsonl`.
- Ensure you have write permissions to the output directory.

## How to add a new variable

1. Add a new rule in `chat_recycler/rules/default_rules.yaml` with `kind: variable`.
2. Set `var_name`, `var_type`, and `var_value` plus regex `patterns`.
3. Run `chat-recycler extract --in data/export --out out/`.
4. Confirm the new variable appears in `exports/variables.parquet` and `db.sqlite`.

## Roadmap

### Phase 1
- Labeling UI (simple TUI) to mark outcomes + “shipped”
- Better embeddings + semantic clustering
- Plug-in extractor interface

### Phase 2
- Time-series view: state transitions (exploring → building → shipping)
- Personalized “next step recommender”
- Export to Notion/Obsidian formats

### Phase 3
- Optional online sync while keeping local-first default
