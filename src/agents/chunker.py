from __future__ import annotations

from src.models import BoundingBox, ExtractedDocument, LDU, PageRef, TextBlock
from src.utils.hashing import stable_hash


class ChunkValidator:
    def validate(self, chunks: list[LDU]) -> None:
        by_hash = {ch.content_hash: ch for ch in chunks}
        for ch in chunks:
            if ch.chunk_type == "table" and "headers" not in ch.metadata:
                raise ValueError("Chunk rule violation: table chunk missing headers metadata")
            if ch.chunk_type == "figure" and "caption" not in ch.metadata:
                raise ValueError("Chunk rule violation: figure chunk missing caption metadata")
            if ch.chunk_type in {"table", "figure"} and not ch.parent_section:
                raise ValueError("Chunk rule violation: structural chunks must carry parent_section metadata")
            if ch.chunk_type == "table":
                header = ch.metadata.get("headers", "").strip()
                if header:
                    header_parts = [h.strip() for h in header.split(",") if h.strip()]
                    if any(part not in ch.content for part in header_parts):
                        raise ValueError("Chunk rule violation: table content must include header row")
            for rel in ch.related_chunk_hashes:
                if rel not in by_hash:
                    raise ValueError("Chunk rule violation: cross-reference points to unknown chunk hash")
            if ch.token_count <= 0:
                raise ValueError("Chunk rule violation: token_count must be positive")


class ChunkingEngine:
    def __init__(self, max_tokens: int = 500, enabled_rules: list[str] | None = None) -> None:
        self.max_tokens = max_tokens
        self.validator = ChunkValidator()
        self.enabled_rules = set(enabled_rules or [])

    def chunk(self, doc: ExtractedDocument) -> list[LDU]:
        chunks: list[LDU] = []
        page_section_map: dict[int, str] = {}

        for block in doc.text_blocks:
            section = block.section_hint or page_section_map.get(block.page_number) or "Uncategorized"
            page_section_map[block.page_number] = section
            chunks.extend(self._chunks_from_text_block(doc=doc, block=block, section=section))

        for table in doc.tables:
            flat_rows = [" | ".join(table.headers)] + [" | ".join(r) for r in table.rows]
            content = "\n".join(flat_rows)
            section = page_section_map.get(table.page_number) or table.title or "Uncategorized"
            chunks.append(
                self._build_chunk(
                    document_name=doc.document_name,
                    content=content,
                    chunk_type="table",
                    page=table.page_number,
                    bbox=table.bbox,
                    parent_section=section,
                    metadata={
                        "headers": ",".join(table.headers),
                        "title": table.title or "table",
                        "source_table_title": table.title or "table",
                    },
                )
            )

        for fig in doc.figures:
            caption = fig.caption or "[no caption]"
            section = page_section_map.get(fig.page_number) or "figure"
            chunks.append(
                self._build_chunk(
                    document_name=doc.document_name,
                    content=caption,
                    chunk_type="figure",
                    page=fig.page_number,
                    bbox=fig.bbox,
                    parent_section=section,
                    metadata={"caption": caption},
                )
            )

        self._link_cross_references(chunks)
        self.validator.validate(chunks)
        return chunks

    def _chunks_from_text_block(self, doc: ExtractedDocument, block: TextBlock, section: str) -> list[LDU]:
        toks = self._token_count(block.content)
        is_list = self._is_numbered_list(block.content)
        if is_list and toks > self.max_tokens:
            # Keep numbered lists coherent when possible; split only when they exceed the max token budget.
            return self._split_text_into_chunks(
                document_name=doc.document_name,
                text=block.content,
                page=block.page_number,
                bbox=block.bbox,
                parent_section=section,
                chunk_type="list",
                metadata={"section_hint": section, "split_reason": "max_tokens"},
            )
        chunk_type = "list" if is_list else "text"
        return [
            self._build_chunk(
                document_name=doc.document_name,
                content=block.content,
                chunk_type=chunk_type,
                page=block.page_number,
                bbox=block.bbox,
                parent_section=section,
                metadata={"section_hint": section},
            )
        ]

    def _split_text_into_chunks(
        self,
        document_name: str,
        text: str,
        page: int,
        bbox: BoundingBox,
        parent_section: str,
        chunk_type: str,
        metadata: dict[str, str],
    ) -> list[LDU]:
        words = text.split()
        out: list[LDU] = []
        for i in range(0, len(words), self.max_tokens):
            part = " ".join(words[i : i + self.max_tokens]).strip()
            if not part:
                continue
            out.append(
                self._build_chunk(
                    document_name=document_name,
                    content=part,
                    chunk_type=chunk_type,
                    page=page,
                    bbox=bbox,
                    parent_section=parent_section,
                    metadata=metadata,
                )
            )
        return out

    def _build_chunk(
        self,
        document_name: str,
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
            page_refs=[PageRef(document_name=document_name, page_number=page, bbox=bbox)],
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
        figure_by_idx = {i + 1: ch.content_hash for i, ch in enumerate(ch for ch in chunks if ch.chunk_type == "figure")}
        for ch in chunks:
            low = ch.content.lower()
            for idx, table_hash in table_by_idx.items():
                if f"table {idx}" in low or f"see table {idx}" in low:
                    ch.related_chunk_hashes.append(table_hash)
            for idx, fig_hash in figure_by_idx.items():
                if f"figure {idx}" in low or f"fig. {idx}" in low or f"see figure {idx}" in low:
                    ch.related_chunk_hashes.append(fig_hash)
