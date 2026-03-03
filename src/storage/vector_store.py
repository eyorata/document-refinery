from __future__ import annotations

import math
from collections import Counter

from src.models import LDU


class SimpleVectorStore:
    def __init__(self) -> None:
        self._chunks: list[LDU] = []
        self._tf: list[Counter[str]] = []

    def ingest(self, chunks: list[LDU]) -> None:
        for ch in chunks:
            self._chunks.append(ch)
            self._tf.append(Counter(self._tokenize(ch.content)))

    def search(self, query: str, top_k: int = 3, filter_pages: set[int] | None = None) -> list[LDU]:
        q_tf = Counter(self._tokenize(query))
        scored: list[tuple[float, int]] = []
        for i, tf in enumerate(self._tf):
            chunk = self._chunks[i]
            if filter_pages and chunk.page_refs[0].page_number not in filter_pages:
                continue
            score = self._cosine(q_tf, tf)
            scored.append((score, i))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [self._chunks[i] for score, i in scored[:top_k] if score > 0]

    def _tokenize(self, txt: str) -> list[str]:
        return [t.lower() for t in txt.split() if t.strip()]

    def _cosine(self, a: Counter[str], b: Counter[str]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(a[k] * b.get(k, 0) for k in a)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)
