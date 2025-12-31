Here’s a **copy-paste comprehensive build prompt** (no filler words) to generate an **extended version design + a working Python MVP pipeline**.

You are a senior staff engineer building a local-first Python system that recycles exported ChatGPT conversations into a structured dataset for pattern mining + variable extraction + simple prediction. Ship an MVP that runs on a Mac M1 16GB. Design for extension.

# Product name

chat-recycler

# Primary objective

Given exported ChatGPT conversations (JSON + optional plaintext), produce:

1. Signal Log (events)
2. Variable Registry (features)
3. Clusters + canonical playbooks
4. Baseline predictors (next artifact, next blocker, completion likelihood)
5. Reports (markdown + JSON) + a queryable DB

# Non-negotiables

- Local-first by default (no network required)
- Deterministic pipeline option (no LLM) + optional LLM extractor (Ollama)
- Reproducible outputs (hashing, run metadata)
- Clear schemas + typed validation
- CLI-first, then importable library
- Works on M1 16GB with reasonable memory usage
- No hidden magic; every transformation logged

# Inputs

Support:

- ChatGPT export JSON (conversations.json or per-thread JSON)
- Optional: .md / .txt transcripts
- Optional: a “labels.json” for outcomes (success/fail, shipped/not shipped) if user provides later

CLI input flags:

- -in (file or directory)
- -out
- -format chatgpt_json|text|auto
- -use-llm true|false (default false)
- -ollama-model (default: qwen2.5:7b or similar small model)
- -max-chars-per-chunk
- -db sqlite|duckdb (default sqlite)
- -export parquet|json|both
- -report true|false

# Outputs

Create an output directory containing:

- db.sqlite (or duckdb)
- exports/
    - conversations.parquet (normalized messages)
    - events.parquet (signal log)
    - variables.parquet (variable registry)
    - clusters.parquet
    - predictions.parquet
- reports/
    - INDEX.md
    - convos/<conversation_id>.md (summary + signals + variables)
    - clusters.md (cluster themes + canonical playbooks)
    - metrics.md (coverage, confidence, error counts)
- meta/
    - run.json (version, timestamps, config, hashes)
    - schema.json (schemas + enums)
    - errors.jsonl (structured errors)

# Core data model (Pydantic)

Define Pydantic models + JSON schema export.

## Normalized message

MessageRow:

- conversation_id: str
- message_id: str
- created_at: datetime|None
- role: enum[user,assistant,system,tool]
- content: str
- content_hash: str (sha256)
- tokens_est: int (approx, no external tokenizer required)
- source_file: str
- order_index: int

## Signal event

EventRow:

- event_id: str (uuid)
- conversation_id: str
- message_id: str|None
- created_at: datetime|None
- event_type: enum[
    
    intent, constraint, decision, blocker, action_request, outcome, next_step, metric, tool_use
    
    ]
    
- label: str (short name)
- event_text: str
- entities: dict (lightweight extracted entities)
- confidence: float (0..1)
- extractor: enum[heuristic,llm]
- span: dict (start,end indexes in message content if available)

## Variable registry

VariableRow:

- var_id: str (uuid)
- conversation_id: str
- message_id: str|None
- created_at: datetime|None
- var_name: str
- var_type: enum[string,int,float,bool,enum,json]
- var_value: json (store typed values)
- var_source_text: str
- confidence: float (0..1)
- rule_id: str (which rule produced it)
- extractor: enum[heuristic,llm]

## Cluster

ClusterRow:

- cluster_id: str
- conversation_id: str
- cluster_label: str
- cluster_summary: str
- centroid_repr: str (keywords)
- method: enum[tfidf_kmeans,embeddings_hdbscan]
- confidence: float

## Prediction

PredictionRow:

- prediction_id: str
- conversation_id: str
- target: enum[next_artifact,next_blocker,completion_likelihood]
- prediction: json
- model: str
- features_used: list[str]
- confidence: float

# Extraction logic requirements

Implement two extractor paths:

## A) Heuristic extractor (default, deterministic)

- Regex + rules + keyword maps to detect:
    - action_request: “make/build/write/create/convert/turn into”
    - requested_artifact: enum[prompt,schema,code,checklist,prd,email,mindmap,report]
    - blockers: enum[missing_context,env_error,over_scope,ambiguity,formatting,performance,policy]
    - constraints: hardware, time pressure, local-first, privacy, “no filler words”, “M1 16GB”
    - outcomes: explicit success/fail markers + “works/doesn’t work/error/404”
- Provide a rules file: rules/default_rules.yaml
- Every extracted variable must include rule_id + source text

## B) Optional LLM extractor (Ollama)

- Only used if --use-llm true
- Use local Ollama HTTP API
- Prompt the model to emit strict JSON matching EventRow + VariableRow lists
- Validate with Pydantic; if invalid, fallback to heuristic for that chunk
- Chunk long conversations by char limit; preserve ordering
- Include an “LLM receipt”: store model name, prompt hash, response hash in meta

# Pipeline stages (implement as modules + CLI commands)

1. ingest
    - detect format
    - parse exports
    - normalize to MessageRow
    - write conversations table + parquet
2. extract
    - generate EventRow + VariableRow
    - compute confidence + coverage stats
3. cluster
    - MVP: TF-IDF + KMeans
    - Optional: sentence-transformers embeddings + HDBSCAN if installed
4. playbooks
    - per cluster, generate:
        - “canonical workflow”
        - “common blockers”
        - “best next artifact”
    - MVP: heuristic summarization (top n keywords + frequent events)
    - Optional: LLM summarization if enabled
5. predict
    - MVP: baseline models using scikit-learn
        - next_artifact: multiclass
        - next_blocker: multiclass
        - completion_likelihood: logistic regression (requires labels; if labels missing, output “untrained” with rules-based proxy)
    - Features: counts of event types, presence of constraints, artifact requests, length stats, sentiment proxy (simple lexicon)
6. report
    - generate markdown reports
    - include run metadata + metrics

# Storage

- SQLite schema with indexes:
    - messages(conversation_id, created_at)
    - events(conversation_id, event_type)
    - variables(conversation_id, var_name)
- Also export parquet for analytics

# Metrics

Compute + print:

- conversations ingested
- messages ingested
- event coverage: events per 1k chars, % messages with >=1 event
- variable coverage: vars per convo, top vars
- extraction failure rate
- confidence distribution

# Project structure

chat_recycler/

**init**.py

cli.py

config.py

ingest/

chatgpt_json.py

text.py

normalize.py

extract/

heuristic.py

llm_ollama.py

rules_loader.py

schemas.py

cluster/

tfidf_kmeans.py

embeddings_hdbscan.py (optional dependency)

predict/

features.py

train.py

infer.py

report/

markdown.py

templates/

store/

sqlite.py

parquet.py

utils/

hashing.py

chunking.py

time.py

logging.py

rules/

default_rules.yaml

tests/

test_ingest.py

test_extract.py

test_store.py

pyproject.toml

README.md

Makefile

# Implementation constraints

- Use: python>=3.11, pydantic, pandas, scikit-learn, typer (CLI), rich (logs), duckdb optional
- Avoid heavyweight dependencies unless optional
- Never crash on a single bad conversation; log structured error + continue

# CLI UX

Provide commands:

- chat-recycler ingest --in data/export --out out/
- chat-recycler extract --in out/ --use-llm false
- chat-recycler cluster --in out/
- chat-recycler predict --in out/
- chat-recycler report --in out/
- chat-recycler run --in data/export --out out/ (runs full pipeline)

# Acceptance tests

- Running `chat-recycler run` on a small sample produces:
    - db + parquet exports
    - reports/INDEX.md
    - non-empty events + variables tables
- Deterministic mode produces identical outputs across runs given same input + config
- LLM mode validates JSON; invalid responses fallback safely
- Unit tests cover:
    - parsing
    - at least 5 rule extractions
    - DB write/read
    - cluster generation

# Extended version design (include in README as roadmap)

Phase 1:

- Labeling UI (simple TUI) to mark outcomes + “shipped”
- Better embeddings + semantic clustering
- Plug-in extractor interface
    
    Phase 2:
    
- Time-series view: state transitions (exploring → building → shipping)
- Personalized “next step recommender”
- Export to Notion/Obsidian formats
    
    Phase 3:
    
- Online sync optional, but keep local-first default

# Deliver now

Generate:

- Full codebase files (all modules)
- Minimal sample input + example run commands
- README with install + usage + troubleshooting
- Default rules YAML with 30+ rules (artifact types, blockers, constraints, tones)
- Provide a short “How to add a new variable” guide

Return output:

1. File tree
2. Each file content with path headers
3. Final usage examples
