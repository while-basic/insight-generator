from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from chat_recycler.extract.schemas import ClusterRow, EventRow, MessageRow, PredictionRow, VariableRow


def _write(path: Path, rows: Iterable[object]) -> None:
    data = [row.model_dump() for row in rows]
    df = pd.DataFrame(data)
    if not df.empty:
        df.to_parquet(path, index=False)
    else:
        df.to_parquet(path, index=False)


def write_messages(path: Path, messages: Iterable[MessageRow]) -> None:
    _write(path, messages)


def write_events(path: Path, events: Iterable[EventRow]) -> None:
    _write(path, events)


def write_variables(path: Path, variables: Iterable[VariableRow]) -> None:
    _write(path, variables)


def write_clusters(path: Path, clusters: Iterable[ClusterRow]) -> None:
    _write(path, clusters)


def write_predictions(path: Path, predictions: Iterable[PredictionRow]) -> None:
    _write(path, predictions)
