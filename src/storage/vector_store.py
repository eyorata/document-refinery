from __future__ import annotations

import math
from collections import Counter
from typing import Protocol

from src.models import LDU
from src.utils.hashing import stable_hash


class VectorStore(Protocol):
    def ingest(self, chunks: list[LDU]) -> None: ...
    def search(self, query: str, top_k: int = 3, filter_pages: set[int] | None = None) -> list[LDU]: ...
    def get_by_hashes(self, hashes: list[str]) -> list[LDU]: ...


class SimpleVectorStore:
    def __init__(self) -> None:
        self._chunks: list[LDU] = []
        self._tf: list[Counter[str]] = []
        self._by_hash: dict[str, LDU] = {}

    def ingest(self, chunks: list[LDU]) -> None:
        for ch in chunks:
            self._chunks.append(ch)
            self._tf.append(Counter(self._tokenize(ch.content)))
            self._by_hash[ch.content_hash] = ch

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

    def get_by_hashes(self, hashes: list[str]) -> list[LDU]:
        out: list[LDU] = []
        for h in hashes:
            chunk = self._by_hash.get(h)
            if chunk is not None:
                out.append(chunk)
        return out

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


class FaissVectorStore:
    """
    Optional FAISS-backed vector store using lightweight hashed-token embeddings.
    Falls back to SimpleVectorStore via `build_vector_store` when FAISS is unavailable.
    """

    def __init__(self, embedding_dim: int = 256, similarity: str = "cosine") -> None:
        try:
            import faiss  # type: ignore[import-not-found]
            import numpy as np  # type: ignore[import-not-found]
        except Exception as exc:
            raise RuntimeError("FAISS backend requested but `faiss`/`numpy` are unavailable.") from exc
        self.faiss = faiss
        self.np = np
        self.embedding_dim = max(32, int(embedding_dim))
        self.similarity = similarity.lower().strip()
        self._chunks: list[LDU] = []
        self._by_hash: dict[str, LDU] = {}

        if self.similarity == "l2":
            self.index = self.faiss.IndexFlatL2(self.embedding_dim)
            self._normalize = False
        else:
            # cosine similarity via normalized inner product
            self.index = self.faiss.IndexFlatIP(self.embedding_dim)
            self._normalize = True

    def ingest(self, chunks: list[LDU]) -> None:
        if not chunks:
            return
        embs = [self._embed(ch.content) for ch in chunks]
        mat = self.np.vstack(embs).astype("float32")
        if self._normalize:
            self.faiss.normalize_L2(mat)
        self.index.add(mat)
        for ch in chunks:
            self._chunks.append(ch)
            self._by_hash[ch.content_hash] = ch

    def search(self, query: str, top_k: int = 3, filter_pages: set[int] | None = None) -> list[LDU]:
        if not self._chunks or self.index.ntotal == 0:
            return []
        top_k = max(1, int(top_k))
        q = self._embed(query).astype("float32").reshape(1, -1)
        if self._normalize:
            self.faiss.normalize_L2(q)
        # Pull extra candidates then filter by pages.
        k_scan = min(max(top_k * 5, top_k), len(self._chunks))
        scores, idxs = self.index.search(q, k_scan)
        out: list[LDU] = []
        for idx in idxs[0]:
            if idx < 0 or idx >= len(self._chunks):
                continue
            ch = self._chunks[int(idx)]
            if filter_pages and ch.page_refs[0].page_number not in filter_pages:
                continue
            out.append(ch)
            if len(out) >= top_k:
                break
        return out

    def get_by_hashes(self, hashes: list[str]) -> list[LDU]:
        out: list[LDU] = []
        for h in hashes:
            chunk = self._by_hash.get(h)
            if chunk is not None:
                out.append(chunk)
        return out

    def _embed(self, text: str):
        vec = self.np.zeros((self.embedding_dim,), dtype=self.np.float32)
        for tok in self._tokenize(text):
            h = stable_hash(tok)
            idx = int(h[:8], 16) % self.embedding_dim
            sign = -1.0 if (int(h[8:10], 16) % 2) else 1.0
            vec[idx] += sign
        return vec

    def _tokenize(self, txt: str) -> list[str]:
        return [t.lower() for t in txt.split() if t.strip()]


def build_vector_store(storage_cfg: dict | None = None) -> VectorStore:
    cfg = storage_cfg or {}
    vs_cfg = cfg.get("vector_store", cfg)
    backend = str(vs_cfg.get("backend", "simple")).lower().strip()
    if backend == "faiss":
        try:
            return FaissVectorStore(
                embedding_dim=int(vs_cfg.get("embedding_dim", 256)),
                similarity=str(vs_cfg.get("similarity", "cosine")),
            )
        except Exception:
            return SimpleVectorStore()
    return SimpleVectorStore()
