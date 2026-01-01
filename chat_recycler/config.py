from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    input_path: Path
    output_path: Path
    input_format: str = "auto"
    use_llm: bool = False
    ollama_model: str = "qwen2.5:7b"
    max_chars_per_chunk: int = 6000
    db: str = "sqlite"
    export_format: str = "both"
    report: bool = True


DEFAULT_OUTPUT_DIR = "out"
