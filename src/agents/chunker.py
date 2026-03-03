from __future__ import annotations

from collections import defaultdict

from src.models import BoundingBox, ExtractedDocument, LDU, PageRef
from src.utils.hashing import stable_hash


class ChunkValidator:
    def validate(self, chunks: list[LDU]) -> None:
        for ch in chunks:
            if ch.chunk_type == "table" and "headers" not in ch.metadata:
                raise ValueError("Chunk rule violation: table chunk missing headers metadata")
            if ch.chunk_type == "figure" and "caption" not in ch.metadata:
                raise ValueError("Chunk rule violation: figure chunk missing caption metadata")
            if ch.token_count <= 0:
                raise ValueError("Chunk rule violation: token_count must be positive")


class ChunkingEngine:
    def __init__(self, max_tokens: int = 500) -> None:
        self.max_tokens = max_tokens
        self.validator = ChunkValidator()

    def chunk(self, doc: ExtractedDocument) -> list[LDU]:
        chunks: list[LDU] = []

        for block in doc.text_blocks:
            toks = self._token_count(block.content)
            chunks.append(
                self._build_chunk(
                    content=block.content,
                    chunk_type="list" if self._is_numbered_list(block.content) and toks <= self.max_tokens else "text",
                    page=block.page_number,
                    bbox=block.bbox,
                    parent_section=block.section_hint,
                    metadata={"section_hint": block.section_hint or "unknown"},
                )
            )

        for table in doc.tables:
            flat_rows = [" | ".join(table.headers)] + [" | ".join(r) for r in table.rows]
            content = "\n".join(flat_rows)
            chunks.append(
                self._build_chunk(
                    content=content,
                    chunk_type="table",
                    page=table.page_number,
                    bbox=table.bbox,
                    parent_section=table.title,
                    metadata={"headers": ",".join(table.headers), "title": table.title or "table"},
                )
            )

        for fig in doc.figures:
            caption = fig.caption or "[no caption]"
            chunks.append(
                self._build_chunk(
                    content=caption,
                    chunk_type="figure",
                    page=fig.page_number,
                    bbox=fig.bbox,
                    parent_section="figure",
                    metadata={"caption": caption},
                )
            )

        self._link_cross_references(chunks)
        self.validator.validate(chunks)
        return chunks

    def _build_chunk(
        self,
        content: str,
        chunk_type: str,
        page: int,
        bbox: BoundingBox,
        parent_section: str | None,
        metadata: dict[str, str],
    ) -> LDU:
        tok = self._token_count(content)
        h = stable_hash(f"{page}:{bbox.model_dump_json()}:{content}")
        return LDU(
            content=content,
            chunk_type=chunk_type,  # type: ignore[arg-type]
            page_refs=[PageRef(document_name="", page_number=page, bbox=bbox)],
            bounding_box=bbox,
            parent_section=parent_section,
            token_count=tok,
            content_hash=h,
            metadata=metadata,
        )

    def _token_count(self, txt: str) -> int:
        return max(1, len(txt.split()))

    def _is_numbered_list(self, txt: str) -> bool:
        lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        return any(ln[:2].isdigit() and ln[2:3] in {".", ")"} for ln in lines)

    def _link_cross_references(self, chunks: list[LDU]) -> None:
        table_by_idx = {i + 1: ch.content_hash for i, ch in enumerate(ch for ch in chunks if ch.chunk_type == "table")}
        for ch in chunks:
            low = ch.content.lower()
            for idx, table_hash in table_by_idx.items():
                if f"table {idx}" in low or f"see table {idx}" in low:
                    ch.related_chunk_hashes.append(table_hash)
