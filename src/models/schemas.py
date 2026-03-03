from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class OriginType(str, Enum):
    NATIVE_DIGITAL = "native_digital"
    SCANNED_IMAGE = "scanned_image"
    MIXED = "mixed"
    FORM_FILLABLE = "form_fillable"


class LayoutComplexity(str, Enum):
    SINGLE_COLUMN = "single_column"
    MULTI_COLUMN = "multi_column"
    TABLE_HEAVY = "table_heavy"
    FIGURE_HEAVY = "figure_heavy"
    MIXED = "mixed"


class CostTier(str, Enum):
    FAST_TEXT_SUFFICIENT = "fast_text_sufficient"
    NEEDS_LAYOUT_MODEL = "needs_layout_model"
    NEEDS_VISION_MODEL = "needs_vision_model"


class BoundingBox(BaseModel):
    x0: float = 0.0
    y0: float = 0.0
    x1: float = 0.0
    y1: float = 0.0


class PageRef(BaseModel):
    document_name: str
    page_number: int = Field(ge=1)
    bbox: BoundingBox


class ProvenanceItem(BaseModel):
    document_name: str
    page_number: int = Field(ge=1)
    bbox: BoundingBox
    content_hash: str


class ProvenanceChain(BaseModel):
    citations: list[ProvenanceItem]


class DocumentProfile(BaseModel):
    doc_id: str
    document_name: str
    origin_type: OriginType
    layout_complexity: LayoutComplexity
    language_code: str
    language_confidence: float = Field(ge=0.0, le=1.0)
    domain_hint: Literal["financial", "legal", "technical", "medical", "general"]
    estimated_extraction_cost: CostTier
    avg_char_density: float = 0.0
    avg_image_ratio: float = 0.0


class TextBlock(BaseModel):
    content: str
    page_number: int
    bbox: BoundingBox
    section_hint: str | None = None
    reading_order: int = 0


class TableObject(BaseModel):
    page_number: int
    bbox: BoundingBox
    headers: list[str]
    rows: list[list[str]]
    title: str | None = None


class FigureObject(BaseModel):
    page_number: int
    bbox: BoundingBox
    caption: str | None = None


class ExtractedDocument(BaseModel):
    doc_id: str
    document_name: str
    strategy_used: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    text_blocks: list[TextBlock]
    tables: list[TableObject]
    figures: list[FigureObject]


class LDU(BaseModel):
    content: str
    chunk_type: Literal["text", "table", "figure", "list"]
    page_refs: list[PageRef]
    bounding_box: BoundingBox
    parent_section: str | None = None
    token_count: int
    content_hash: str
    related_chunk_hashes: list[str] = []
    metadata: dict[str, str] = {}


class PageIndexNode(BaseModel):
    title: str
    page_start: int
    page_end: int
    child_sections: list["PageIndexNode"] = []
    key_entities: list[str] = []
    summary: str
    data_types_present: list[str] = []


class ExtractionLedgerEntry(BaseModel):
    doc_id: str
    document_name: str
    strategy_used: str
    confidence_score: float
    cost_estimate_usd: float
    processing_time_sec: float
    escalated_from: str | None = None


class QueryAnswer(BaseModel):
    answer: str
    provenance: ProvenanceChain


PageIndexNode.model_rebuild()
