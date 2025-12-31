from __future__ import annotations

from typing import Iterable, List


def chunk_text(text: str, max_chars: int) -> List[str]:
    if max_chars <= 0:
        return [text]
    chunks: List[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(length, start + max_chars)
        chunks.append(text[start:end])
        start = end
    return chunks


def join_chunks(chunks: Iterable[str]) -> str:
    return "".join(chunks)
