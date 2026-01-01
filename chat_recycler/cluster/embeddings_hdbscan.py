from __future__ import annotations

from typing import List

from chat_recycler.extract.schemas import ClusterRow, MessageRow


def cluster_with_embeddings(messages: List[MessageRow]) -> List[ClusterRow]:
    raise ImportError("Optional embeddings dependencies not installed")
