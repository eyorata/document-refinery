from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from src.models import CostTier, DocumentProfile, LayoutComplexity, OriginType
from src.utils.hashing import stable_hash


@dataclass
class PageStat:
    char_count: int
    page_area: float
    image_count: int


class TriageAgent:
    def __init__(self, domain_keywords: dict[str, list[str]], thresholds: dict[str, float]) -> None:
        self.domain_keywords = domain_keywords
        self.thresholds = thresholds

    def profile(self, document_path: str) -> DocumentProfile:
        start = time.time()
        path = Path(document_path)
        doc_id = stable_hash(str(path.resolve()))[:16]
        text_by_page, page_stats = self._read_pdf(path)
        flat_text = "\n".join(text_by_page)

        avg_density = self._avg_char_density(page_stats)
        avg_image_ratio = self._avg_image_ratio(page_stats)
        origin = self._origin_type(avg_density, avg_image_ratio)
        complexity = self._layout_complexity(text_by_page)
        domain = self._domain_hint(flat_text)
        cost = self._cost_tier(origin, complexity)

        _ = time.time() - start
        return DocumentProfile(
            doc_id=doc_id,
            document_name=path.name,
            origin_type=origin,
            layout_complexity=complexity,
            language_code="en",
            language_confidence=0.85,
            domain_hint=domain,
            estimated_extraction_cost=cost,
            avg_char_density=avg_density,
            avg_image_ratio=avg_image_ratio,
        )

    def _read_pdf(self, path: Path) -> tuple[list[str], list[PageStat]]:
        reader = PdfReader(str(path))
        texts: list[str] = []
        stats: list[PageStat] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            width = float(page.mediabox.width or 612)
            height = float(page.mediabox.height or 792)
            images = len(getattr(page, "images", []))
            texts.append(text)
            stats.append(PageStat(char_count=len(text), page_area=max(width * height, 1.0), image_count=images))
        return texts, stats

    def _avg_char_density(self, stats: list[PageStat]) -> float:
        if not stats:
            return 0.0
        return sum(s.char_count / s.page_area for s in stats) / len(stats)

    def _avg_image_ratio(self, stats: list[PageStat]) -> float:
        if not stats:
            return 0.0
        max_images = max(self.thresholds.get("max_images_for_ratio", 10), 1)
        return sum(min(s.image_count / max_images, 1.0) for s in stats) / len(stats)

    def _origin_type(self, avg_density: float, avg_image_ratio: float) -> OriginType:
        low_density = self.thresholds.get("low_density_threshold", 0.0002)
        high_density = self.thresholds.get("high_density_threshold", 0.001)
        image_heavy = self.thresholds.get("image_heavy_threshold", 0.6)

        if avg_density < low_density and avg_image_ratio >= image_heavy:
            return OriginType.SCANNED_IMAGE
        if avg_density >= high_density and avg_image_ratio < 0.3:
            return OriginType.NATIVE_DIGITAL
        return OriginType.MIXED

    def _layout_complexity(self, page_texts: list[str]) -> LayoutComplexity:
        text = "\n".join(page_texts)
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        pipe_like = sum(1 for ln in lines if "|" in ln)
        many_short_lines = sum(1 for ln in lines if 8 < len(ln) < 32)
        if pipe_like > 5:
            return LayoutComplexity.TABLE_HEAVY
        if many_short_lines > max(len(lines) * 0.45, 5):
            return LayoutComplexity.MULTI_COLUMN
        if "figure" in text.lower() or "chart" in text.lower():
            return LayoutComplexity.FIGURE_HEAVY
        return LayoutComplexity.SINGLE_COLUMN

    def _domain_hint(self, text: str) -> str:
        low = text.lower()
        best_domain = "general"
        best_score = 0
        for domain, kws in self.domain_keywords.items():
            score = sum(low.count(k.lower()) for k in kws)
            if score > best_score:
                best_domain = domain
                best_score = score
        return best_domain

    def _cost_tier(self, origin: OriginType, complexity: LayoutComplexity) -> CostTier:
        if origin == OriginType.SCANNED_IMAGE:
            return CostTier.NEEDS_VISION_MODEL
        if complexity in {LayoutComplexity.TABLE_HEAVY, LayoutComplexity.MULTI_COLUMN, LayoutComplexity.MIXED}:
            return CostTier.NEEDS_LAYOUT_MODEL
        return CostTier.FAST_TEXT_SUFFICIENT
