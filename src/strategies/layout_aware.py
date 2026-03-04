from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from src.config import LayoutConfig
from src.models import BoundingBox, DocumentProfile, ExtractedDocument, TableObject, TextBlock
from src.strategies.fast_text import FastTextExtractor


class LayoutToolAdapter(Protocol):
    name: str

    def promote_tables(self, blocks: list[TextBlock], document_path: str, profile: DocumentProfile) -> list[TableObject]:
        ...


class HeuristicLayoutAdapter:
    name = "heuristic"

    def promote_tables(self, blocks: list[TextBlock], document_path: str, profile: DocumentProfile) -> list[TableObject]:
        out: list[TableObject] = []
        for block in blocks:
            low = block.content.lower()
            if "balance sheet" in low or "income statement" in low or "table" in low:
                out.append(
                    TableObject(
                        page_number=block.page_number,
                        bbox=block.bbox,
                        headers=["label", "value"],
                        rows=[["snippet", block.content[:200]]],
                        title="Heuristic layout table",
                    )
                )
        return out


class ExternalTablePayload(BaseModel):
    page_number: int = 1
    headers: list[str]
    rows: list[list[str]]
    title: str | None = None
    bbox: BoundingBox | None = None


def normalize_external_tables(raw: object) -> list[ExternalTablePayload]:
    if isinstance(raw, dict):
        if "tables" in raw:
            raw = raw["tables"]
        else:
            raw = [raw]
    if not isinstance(raw, list):
        raise ValueError("External table payload must be a list or an object with `tables` list.")

    out: list[ExternalTablePayload] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        out.append(ExternalTablePayload.model_validate(item))
    return out


class ExternalPayloadLayoutAdapter:
    name = "external_payload"

    def __init__(self, options: dict | None = None, fallback: LayoutToolAdapter | None = None) -> None:
        self.options = options or {}
        self.fallback = fallback or HeuristicLayoutAdapter()

    def promote_tables(self, blocks: list[TextBlock], document_path: str, profile: DocumentProfile) -> list[TableObject]:
        strict = bool(self.options.get("strict", False))
        path_str = str(self.options.get("payload_json_path", "")).strip()
        if not path_str:
            if strict:
                raise ValueError("external_payload adapter requires `payload_json_path` in options.")
            return self.fallback.promote_tables(blocks, document_path, profile)

        payload_path = Path(path_str)
        try:
            raw = json.loads(payload_path.read_text(encoding="utf-8"))
            normalized = normalize_external_tables(raw)
        except Exception:
            if strict:
                raise
            return self.fallback.promote_tables(blocks, document_path, profile)

        if not normalized:
            return self.fallback.promote_tables(blocks, document_path, profile)
        return self._to_tables(normalized)

    def _to_tables(self, payloads: list[ExternalTablePayload]) -> list[TableObject]:
        default_w = float(self.options.get("default_table_bbox_width", 612.0))
        default_h = float(self.options.get("default_table_bbox_height", 250.0))
        out: list[TableObject] = []
        for payload in payloads:
            bbox = payload.bbox or BoundingBox(x0=0.0, y0=0.0, x1=default_w, y1=default_h)
            out.append(
                TableObject(
                    page_number=max(1, payload.page_number),
                    bbox=bbox,
                    headers=[str(h) for h in payload.headers],
                    rows=[[str(c) for c in row] for row in payload.rows],
                    title=payload.title or "External table",
                )
            )
        return out


class DoclingLayoutAdapter:
    name = "docling"

    def __init__(self, options: dict | None = None, fallback: LayoutToolAdapter | None = None) -> None:
        self.options = options or {}
        self.fallback = fallback or HeuristicLayoutAdapter()

    def promote_tables(self, blocks: list[TextBlock], document_path: str, profile: DocumentProfile) -> list[TableObject]:
        strict = bool(self.options.get("strict", False))
        try:
            rows = self._extract_rows_with_docling(document_path)
            if rows:
                return self._to_tables(rows)
        except Exception:
            if strict:
                raise
        return self.fallback.promote_tables(blocks, document_path, profile)

    def _extract_rows_with_docling(self, document_path: str) -> list[dict]:
        # Optional dependency: if docling is not installed, fallback behavior is used.
        from docling.document_converter import DocumentConverter  # type: ignore[import-not-found]

        converter = DocumentConverter()
        result = converter.convert(document_path)
        document = getattr(result, "document", None)
        if document is None:
            return []

        tables = getattr(document, "tables", None)
        if not tables:
            return []

        out: list[dict] = []
        for table in tables:
            page_no = self._infer_page_number(table)
            title = getattr(table, "caption", None) or getattr(table, "title", None)
            headers, data_rows = self._extract_table_rows(table)
            if not headers and not data_rows:
                continue
            out.append(
                {
                    "page_number": page_no,
                    "headers": headers or ["col_1"],
                    "rows": data_rows or [[""]],
                    "title": str(title) if title else "Docling table",
                }
            )
        return out

    def _extract_table_rows(self, table: object) -> tuple[list[str], list[list[str]]]:
        for attr in ("export_to_dataframe", "to_dataframe", "as_dataframe"):
            fn = getattr(table, attr, None)
            if callable(fn):
                try:
                    df = fn()
                    if df is not None and not getattr(df, "empty", True):
                        headers = [str(c).strip() for c in list(df.columns)]
                        rows = [[str(v).strip() for v in row] for row in df.fillna("").values.tolist()]
                        return headers, rows
                except Exception:
                    pass

        text = ""
        for attr in ("text", "markdown", "md", "content"):
            val = getattr(table, attr, None)
            if isinstance(val, str) and val.strip():
                text = val
                break
            if callable(val):
                try:
                    maybe = val()
                    if isinstance(maybe, str) and maybe.strip():
                        text = maybe
                        break
                except Exception:
                    pass

        if text:
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            pipe_lines = [ln for ln in lines if "|" in ln]
            if len(pipe_lines) >= 2:
                parsed = [[c.strip() for c in ln.split("|") if c.strip()] for ln in pipe_lines]
                parsed = [row for row in parsed if row]
                if parsed:
                    return parsed[0], parsed[1:] or [[""]]

        cells = getattr(table, "cells", None)
        if isinstance(cells, list) and cells:
            grid: dict[tuple[int, int], str] = {}
            max_r = 0
            max_c = 0
            for cell in cells:
                r = int(getattr(cell, "row", 0))
                c = int(getattr(cell, "col", 0))
                val = str(getattr(cell, "text", "") or "").strip()
                grid[(r, c)] = val
                max_r = max(max_r, r)
                max_c = max(max_c, c)
            rows: list[list[str]] = []
            for r in range(max_r + 1):
                rows.append([grid.get((r, c), "") for c in range(max_c + 1)])
            if rows:
                return rows[0], rows[1:] or [[""]]

        return [], []

    def _infer_page_number(self, table: object) -> int:
        for attr in ("page_no", "page", "page_number"):
            v = getattr(table, attr, None)
            if isinstance(v, int) and v >= 1:
                return v
            if isinstance(v, str) and re.fullmatch(r"\d+", v):
                return int(v)
        prov = getattr(table, "provenance", None)
        if prov is not None:
            for attr in ("page_no", "page", "page_number"):
                v = getattr(prov, attr, None)
                if isinstance(v, int) and v >= 1:
                    return v
        return 1

    def _to_tables(self, rows: list[dict]) -> list[TableObject]:
        bbox_height = float(self.options.get("default_table_bbox_height", 250.0))
        bbox_width = float(self.options.get("default_table_bbox_width", 612.0))
        payloads: list[ExternalTablePayload] = []
        for row in rows:
            payloads.append(
                ExternalTablePayload(
                    page_number=max(1, int(row.get("page_number", 1))),
                    headers=[str(h) for h in row.get("headers", ["col_1"])],
                    rows=[[str(c) for c in r] for r in row.get("rows", [[""]])],
                    title=str(row.get("title", "Docling table")),
                    bbox=BoundingBox(x0=0.0, y0=0.0, x1=bbox_width, y1=bbox_height),
                )
            )
        return ExternalPayloadLayoutAdapter(options=self.options)._to_tables(payloads)


class MineruLayoutAdapter:
    name = "mineru"

    def promote_tables(self, blocks: list[TextBlock], document_path: str, profile: DocumentProfile) -> list[TableObject]:
        raise NotImplementedError(
            "MinerU adapter selected but not wired yet. "
            "Provide integration in MineruLayoutAdapter.promote_tables."
        )


def build_layout_adapter(layout_cfg: dict | None) -> LayoutToolAdapter:
    cfg = layout_cfg or {}
    adapter_cfg = cfg.get("adapter", cfg)
    provider = str(adapter_cfg.get("provider", "heuristic")).lower()
    options = adapter_cfg.get("options", {}) or {}
    if provider == "heuristic":
        return HeuristicLayoutAdapter()
    if provider == "docling":
        return DoclingLayoutAdapter(options=options, fallback=HeuristicLayoutAdapter())
    if provider == "external_payload":
        return ExternalPayloadLayoutAdapter(options=options, fallback=HeuristicLayoutAdapter())
    if provider == "mineru":
        return MineruLayoutAdapter()
    raise ValueError(f"Unsupported layout adapter provider: {provider}")


class LayoutExtractor(FastTextExtractor):
    name = "layout_aware"

    def __init__(self, thresholds: dict[str, float], layout_cfg: dict | None = None) -> None:
        super().__init__(thresholds=thresholds)
        self.layout_cfg = LayoutConfig.model_validate(layout_cfg or {}).model_dump(mode="python")
        self.adapter = build_layout_adapter(self.layout_cfg)

    def extract(self, document_path: str, profile: DocumentProfile) -> tuple[ExtractedDocument, float, float]:
        base, base_conf, _ = super().extract(document_path, profile)
        promoted_tables = self.adapter.promote_tables(base.text_blocks, document_path, profile)
        all_tables = base.tables + promoted_tables
        conf_if_tables = float(self.layout_cfg["confidence_if_tables_present"])
        conf = max(base_conf, conf_if_tables if all_tables else base_conf)
        extracted = ExtractedDocument(
            doc_id=base.doc_id,
            document_name=Path(document_path).name,
            strategy_used=self.name,
            confidence_score=conf,
            text_blocks=base.text_blocks,
            tables=all_tables,
            figures=base.figures,
        )
        estimated_cost = float(self.layout_cfg["estimated_cost_usd"])
        return extracted, conf, estimated_cost
