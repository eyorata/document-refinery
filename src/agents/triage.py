from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pdfplumber
from pypdf import PdfReader

from src.config import TriageThresholds
from src.models import CostTier, DocumentProfile, LayoutComplexity, OriginType
from src.utils.hashing import stable_hash


@dataclass
class PageStat:
    char_count: int
    page_area: float
    image_ratio: float
    whitespace_ratio: float
    font_metadata_present: bool


class DomainClassifier(Protocol):
    def classify(self, text: str) -> str:  # pragma: no cover - simple protocol
        ...


class KeywordDomainClassifier:
    def __init__(self, domain_keywords: dict[str, list[str]]) -> None:
        self.domain_keywords = domain_keywords

    def classify(self, text: str) -> str:
        low = text.lower()
        best_domain = "general"
        best_score = 0
        for domain, kws in self.domain_keywords.items():
            score = sum(low.count(k.lower()) for k in kws)
            if score > best_score:
                best_domain = domain
                best_score = score
        return best_domain


class TriageAgent:
    def __init__(
        self,
        domain_keywords: dict[str, list[str]],
        thresholds: dict[str, float],
        domain_classifier: DomainClassifier | None = None,
    ) -> None:
        self.thresholds = TriageThresholds.model_validate(thresholds).model_dump(mode="python")
        self.domain_classifier = domain_classifier or KeywordDomainClassifier(domain_keywords)

    def profile(self, document_path: str) -> DocumentProfile:
        start = time.time()
        path = Path(document_path)
        doc_id = stable_hash(str(path.resolve()))[:16]
        text_by_page, page_stats = self._read_pdf(path)
        flat_text = "\n".join(text_by_page)

        avg_density = self._avg_char_density(page_stats)
        avg_image_ratio = self._avg_image_ratio(page_stats)
        font_signal_present = any(s.font_metadata_present for s in page_stats)
        origin = self._origin_type(avg_density, avg_image_ratio, self._is_form_fillable(path), font_signal_present)
        complexity = self._layout_complexity(text_by_page)
        domain = self._domain_hint(flat_text)
        cost = self._cost_tier(origin, complexity)

        triage_conf = self._triage_confidence(origin, complexity)

        _ = time.time() - start
        return DocumentProfile(
            doc_id=doc_id,
            document_name=path.name,
            page_count=len(page_stats) if page_stats else 1,
            origin_type=origin,
            layout_complexity=complexity,
            language_code="en",
            language_confidence=float(self.thresholds["default_language_confidence"]),
            domain_hint=domain,
            estimated_extraction_cost=cost,
            avg_char_density=avg_density,
            avg_image_ratio=avg_image_ratio,
            triage_confidence=triage_conf,
        )

    def _read_pdf(self, path: Path) -> tuple[list[str], list[PageStat]]:
        texts: list[str] = []
        stats: list[PageStat] = []
        try:
            with pdfplumber.open(str(path)) as pdf:
                for page in pdf.pages:
                    width = float(page.width or 612)
                    height = float(page.height or 792)
                    page_area = max(width * height, 1.0)
                    text = page.extract_text() or ""
                    chars = page.chars or []
                    images = page.images or []

                    image_area = 0.0
                    for img in images:
                        x0 = float(img.get("x0", 0.0) or 0.0)
                        x1 = float(img.get("x1", x0) or x0)
                        top = float(img.get("top", 0.0) or 0.0)
                        bottom = float(img.get("bottom", top) or top)
                        image_area += max(0.0, (x1 - x0) * (bottom - top))
                    image_ratio = max(0.0, min(1.0, image_area / page_area))

                    words = page.extract_words() or []
                    word_chars = sum(len(str(w.get("text", ""))) for w in words)
                    whitespace_ratio = 1.0 if len(text) == 0 else max(0.0, min(1.0, 1.0 - (word_chars / max(len(text), 1))))
                    font_present = any(bool(c.get("fontname")) for c in chars) if chars else False

                    texts.append(text)
                    stats.append(
                        PageStat(
                            char_count=len(text),
                            page_area=page_area,
                            image_ratio=image_ratio,
                            whitespace_ratio=whitespace_ratio,
                            font_metadata_present=font_present,
                        )
                    )
            return texts, stats
        except Exception:
            # Fallback to pypdf when pdfplumber parsing fails.
            reader = PdfReader(str(path))
            for page in reader.pages:
                text = page.extract_text() or ""
                width = float(page.mediabox.width or 612)
                height = float(page.mediabox.height or 792)
                page_area = max(width * height, 1.0)
                images = len(getattr(page, "images", []))
                image_ratio = min(images / max(int(self.thresholds["max_images_for_ratio"]), 1), 1.0)
                texts.append(text)
                stats.append(
                    PageStat(
                        char_count=len(text),
                        page_area=page_area,
                        image_ratio=image_ratio,
                        whitespace_ratio=0.0,
                        font_metadata_present=False,
                    )
                )
            return texts, stats

    def _avg_char_density(self, stats: list[PageStat]) -> float:
        if not stats:
            return 0.0
        return sum(s.char_count / s.page_area for s in stats) / len(stats)

    def _avg_image_ratio(self, stats: list[PageStat]) -> float:
        if not stats:
            return 0.0
        return sum(s.image_ratio for s in stats) / len(stats)

    def _origin_type(
        self,
        avg_density: float,
        avg_image_ratio: float,
        form_fillable: bool,
        font_signal_present: bool = True,
    ) -> OriginType:
        if form_fillable:
            return OriginType.FORM_FILLABLE
        low_density = float(self.thresholds["low_density_threshold"])
        high_density = float(self.thresholds["high_density_threshold"])
        image_heavy = float(self.thresholds["image_heavy_threshold"])

        if (avg_density < low_density and avg_image_ratio >= image_heavy) or (
            avg_density < low_density and not font_signal_present
        ):
            return OriginType.SCANNED_IMAGE
        native_max_image_ratio = float(self.thresholds["native_digital_max_image_ratio"])
        if avg_density >= high_density and avg_image_ratio < native_max_image_ratio and font_signal_present:
            return OriginType.NATIVE_DIGITAL
        return OriginType.MIXED

    def _layout_complexity(self, page_texts: list[str]) -> LayoutComplexity:
        text = "\n".join(page_texts)
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        pipe_like = sum(1 for ln in lines if "|" in ln)
        short_min = int(self.thresholds["short_line_min_chars"])
        short_max = int(self.thresholds["short_line_max_chars"])
        many_short_lines = sum(1 for ln in lines if short_min < len(ln) < short_max)
        table_heavy_min = int(self.thresholds["table_heavy_min_pipe_lines"])
        if pipe_like > table_heavy_min:
            return LayoutComplexity.TABLE_HEAVY
        short_ratio = float(self.thresholds["multi_column_short_line_ratio"])
        short_min_count = int(self.thresholds["multi_column_short_line_min_count"])
        if many_short_lines > max(len(lines) * short_ratio, short_min_count):
            return LayoutComplexity.MULTI_COLUMN
        if "figure" in text.lower() or "chart" in text.lower():
            return LayoutComplexity.FIGURE_HEAVY
        return LayoutComplexity.SINGLE_COLUMN

    def _domain_hint(self, text: str) -> str:
        return self.domain_classifier.classify(text)

    def _is_form_fillable(self, path: Path) -> bool:
        try:
            reader = PdfReader(str(path))
            root = reader.trailer.get("/Root", {})
            return "/AcroForm" in root
        except Exception:  # noqa: BLE001
            return False

    def _triage_confidence(self, origin: OriginType, complexity: LayoutComplexity) -> float:
        """
        Heuristic confidence score for the overall profile classification.

        - Start from 0.5 (unknown).
        - Add 0.25 if origin_type is not MIXED.
        - Add 0.25 if layout_complexity is not MIXED.
        """
        score = float(self.thresholds["triage_confidence_base"])
        if origin != OriginType.MIXED:
            score += float(self.thresholds["triage_confidence_origin_bonus"])
        if complexity != LayoutComplexity.MIXED:
            score += float(self.thresholds["triage_confidence_layout_bonus"])
        return max(0.0, min(1.0, score))

    def _cost_tier(self, origin: OriginType, complexity: LayoutComplexity) -> CostTier:
        if origin == OriginType.SCANNED_IMAGE:
            return CostTier.NEEDS_VISION_MODEL
        if complexity in {LayoutComplexity.TABLE_HEAVY, LayoutComplexity.MULTI_COLUMN, LayoutComplexity.MIXED}:
            return CostTier.NEEDS_LAYOUT_MODEL
        return CostTier.FAST_TEXT_SUFFICIENT
