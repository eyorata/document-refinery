from __future__ import annotations

import base64
import io
import json
import math
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable, List

from pypdf import PdfReader
import yaml

from src.agents.errors import BudgetExceededError
from src.config import VisionConfig, VlmBudgetConfig
from src.models import BoundingBox, DocumentProfile, ExtractedDocument, FigureObject, TextBlock
from src.strategies.fast_text import FastTextExtractor


class VisionExtractor(FastTextExtractor):
    name = "vision_augmented"

    def __init__(self, thresholds: dict[str, float], vlm_budget: dict | None = None, vision_cfg: dict | None = None) -> None:
        super().__init__(thresholds=thresholds)
        budget = VlmBudgetConfig.model_validate(vlm_budget or {}).model_dump(mode="python")
        raw_cfg = vision_cfg or {}
        base_cfg = VisionConfig.model_validate(raw_cfg).model_dump(mode="python")
        self.vision_cfg = self._merge_with_strategy_config(base_cfg, raw_cfg)
        self.vlm_enabled = bool(budget["enabled"])
        self.max_pages = int(budget["max_pages_per_document"])
        self.cost_per_page_usd = float(budget["cost_per_page_usd"])
        self.max_total_cost_usd = float(budget["max_total_cost_usd"])
        self.stop_on_budget_exceeded = bool(budget["stop_on_budget_exceeded"])
        self.allow_partial_processing = bool(budget["allow_partial_processing"])
        self.providers = self._build_provider_chain()
        self.min_confidence_for_accept = float(self.vision_cfg.get("min_confidence_for_accept", 0.78))
        self.escalate_on_low_confidence = bool(self.vision_cfg.get("escalate_on_low_confidence", True))
        self.allow_best_effort_on_low_confidence = bool(
            self.vision_cfg.get("allow_best_effort_on_low_confidence", True)
        )
        self.max_image_dimension = int(self.vision_cfg.get("max_image_dimension", 1600))
        self.require_model_for_ocr = bool(self.vision_cfg.get("require_model_for_ocr", True))
        self.page_extraction_prompt = str(self.vision_cfg.get("page_extraction_prompt", "")).strip()
        self.last_token_usage = 0
        self.last_provider = "placeholder"
        self.last_provider_attempts: list[str] = []
        self.last_ocr_confidence = 0.0
        self.last_cost_per_page_used = 0.0

    def extract(self, document_path: str, profile: DocumentProfile) -> tuple[ExtractedDocument, float, float]:
        base, _, _ = super().extract(document_path, profile)
        # Do not short-circuit to fast-text for scanned/needs-vision documents.
        # Some scanned PDFs expose tiny text streams that are not usable extraction output.
        origin = self._enum_value(getattr(profile, "origin_type", ""))
        estimated_cost = self._enum_value(getattr(profile, "estimated_extraction_cost", ""))
        should_force_ocr = origin in {"scanned_image", "mixed"} or estimated_cost == "needs_vision_model"
        if base.text_blocks and not should_force_ocr:
            conf = max(base.confidence_score, float(self.vision_cfg["confidence_if_text_present_min"]))
            extracted = ExtractedDocument(
                doc_id=base.doc_id,
                document_name=Path(document_path).name,
                strategy_used=self.name,
                confidence_score=conf,
                text_blocks=base.text_blocks,
                tables=base.tables,
                figures=base.figures,
            )
            return extracted, conf, 0.0

        if not self.vlm_enabled:
            raise BudgetExceededError("VLM extraction is disabled by configuration (`extraction.vlm_budget.enabled=false`).")

        reader = PdfReader(document_path)
        total_pages = len(reader.pages)
        affordable_pages = self._affordable_pages(total_pages)
        if affordable_pages <= 0:
            raise BudgetExceededError("VLM budget cap allows 0 pages; vision processing is blocked.")

        if (total_pages > affordable_pages) and (self.stop_on_budget_exceeded or not self.allow_partial_processing):
            raise BudgetExceededError(
                f"VLM cap exceeded: document has {total_pages} pages but only {affordable_pages} "
                f"pages are allowed by budget/page caps."
            )

        pages_to_process = min(total_pages, affordable_pages)
        self.last_token_usage = 0
        self.last_cost_per_page_used = 0.0
        ocr_blocks = self._ocr_pages_with_vision(
            reader,
            document_path=document_path,
            max_pages=pages_to_process,
            profile=profile,
        )
        ocr_tables = self._extract_tables_from_ocr_blocks(ocr_blocks)
        estimated_cost = pages_to_process * float(self.last_cost_per_page_used or 0.0)
        if pages_to_process < total_pages:
            ocr_blocks.append(
                TextBlock(
                    content=(
                        f"[vision-budget-stop] Processed {pages_to_process}/{total_pages} pages due to "
                        "VLM budget/page caps."
                    ),
                    page_number=max(1, pages_to_process),
                    bbox=BoundingBox(x0=0.0, y0=0.0, x1=612.0, y1=80.0),
                    section_hint="vision budget cap",
                    reading_order=pages_to_process + 1,
                )
            )
        if not ocr_blocks:
            ocr_blocks = [
                TextBlock(
                    content="[vision-placeholder] OCR text would be produced by a multimodal endpoint here.",
                    page_number=1,
                    bbox=BoundingBox(x0=0.0, y0=0.0, x1=612.0, y1=792.0),
                    section_hint="vision extraction",
                    reading_order=1,
                )
            ]

        conf = max(float(self.vision_cfg["confidence_if_ocr_only"]), float(self.last_ocr_confidence or 0.0))
        figure_bbox_height = float(self.vision_cfg["figure_bbox_height"])
        extracted = ExtractedDocument(
            doc_id=base.doc_id,
            document_name=Path(document_path).name,
            strategy_used=self.name,
            confidence_score=conf,
            text_blocks=ocr_blocks,
            tables=base.tables + ocr_tables,
            figures=base.figures
            + [
                FigureObject(
                    page_number=1,
                    bbox=BoundingBox(x0=0.0, y0=0.0, x1=612.0, y1=figure_bbox_height),
                    caption="Vision analysis region",
                )
            ],
        )
        return extracted, conf, estimated_cost

    def _ocr_pages_with_vision(
        self, reader: PdfReader, document_path: str, max_pages: int, profile: DocumentProfile
    ) -> List[TextBlock]:
        self.last_provider_attempts = []
        enabled_providers = [p for p in self.providers if bool(p.get("enabled", False))]
        best_blocks: List[TextBlock] = []
        best_conf = 0.0
        best_provider = "placeholder"
        for provider in self.providers:
            if not bool(provider.get("enabled", False)):
                continue
            provider_name = str(provider.get("name", "provider")).strip() or "provider"
            try:
                blocks, usage_tokens = self._ocr_pages_with_provider(
                    document_path=document_path,
                    max_pages=max_pages,
                    provider=provider,
                )
                self.last_token_usage += usage_tokens
                conf = self._estimate_ocr_confidence(blocks=blocks, profile=profile, max_pages=max_pages)
                self.last_provider_attempts.append(f"{provider_name}:ok:{conf:.3f}")
                if conf > best_conf:
                    best_conf = conf
                    best_blocks = blocks
                    best_provider = provider_name
                if (conf >= self.min_confidence_for_accept) or (not self.escalate_on_low_confidence):
                    self.last_provider = provider_name
                    self.last_ocr_confidence = conf
                    self.last_cost_per_page_used = self._provider_cost_per_page(provider)
                    return blocks
            except Exception as exc:
                detail = str(exc).strip().replace("\n", " ")
                if len(detail) > 180:
                    detail = detail[:177] + "..."
                self.last_provider_attempts.append(f"{provider_name}:error:{exc.__class__.__name__}:{detail}")
                continue

        if best_blocks and self.allow_best_effort_on_low_confidence:
            self.last_provider = best_provider
            self.last_ocr_confidence = best_conf
            # Best provider name maps back to an enabled provider object for cost attribution.
            for provider in self.providers:
                if str(provider.get("name", "")).strip() == best_provider:
                    self.last_cost_per_page_used = self._provider_cost_per_page(provider)
                    break
            return best_blocks

        all_local_enabled = bool(enabled_providers) and all(self._is_local_provider(p) for p in enabled_providers)
        if all_local_enabled:
            # Local model stack should not hard-fail the pipeline; preserve progress with placeholder fallback.
            self.last_provider = "local_provider_failed_placeholder"
            self.last_ocr_confidence = 0.35
            self.last_cost_per_page_used = float(self.cost_per_page_usd)
            return self._ocr_pages_with_placeholder(reader=reader, max_pages=max_pages)

        if self.require_model_for_ocr:
            attempts = ", ".join(self.last_provider_attempts) if self.last_provider_attempts else "no providers enabled"
            raise BudgetExceededError(
                "Vision model required to extract this document, but all providers failed or were low confidence. "
                f"Attempts: {attempts}"
            )
        self.last_provider = "placeholder"
        self.last_ocr_confidence = 0.0
        # Placeholder path represents synthetic OCR work in tests/dev mode.
        self.last_cost_per_page_used = float(self.cost_per_page_usd)
        return self._ocr_pages_with_placeholder(reader=reader, max_pages=max_pages)

    def _ocr_pages_with_provider(self, document_path: str, max_pages: int, provider: dict[str, Any]) -> tuple[List[TextBlock], int]:
        blocks: List[TextBlock] = []
        usage_total = 0
        for i, img in self._iter_pages_as_pil(document_path=document_path, max_pages=max_pages):
            img = self._normalize_image_for_vlm(img)
            width, height = img.size
            b = io.BytesIO()
            img.save(b, format="PNG")
            image_data_url = f"data:image/png;base64,{base64.b64encode(b.getvalue()).decode('ascii')}"

            prompt = self.page_extraction_prompt
            text, usage_tokens = self._openai_compatible_chat_completion(
                prompt=prompt,
                image_data_url=image_data_url,
                provider=provider,
            )
            usage_total += usage_tokens
            blocks.append(
                TextBlock(
                    content=text.strip() or f"[openrouter-empty] Page {i}: no OCR text returned.",
                    page_number=i,
                    bbox=BoundingBox(x0=0.0, y0=0.0, x1=width, y1=height),
                    section_hint=f"vision extraction ({provider.get('name', 'provider')})",
                    reading_order=i,
                )
            )
        return blocks, usage_total

    def _iter_pages_as_pil(self, document_path: str, max_pages: int) -> Iterable[tuple[int, Any]]:
        # Preferred path: pdf2image if available.
        try:
            from pdf2image import convert_from_path  # type: ignore[import-not-found]

            for i in range(1, max_pages + 1):
                pages = convert_from_path(document_path, first_page=i, last_page=i, fmt="png")
                if not pages:
                    break
                yield i, pages[0]
            return
        except Exception:
            pass

        # Fallback path: pypdfium2 (avoids Poppler dependency).
        try:
            import pypdfium2 as pdfium  # type: ignore[import-not-found]

            doc = pdfium.PdfDocument(document_path)
            count = min(len(doc), max_pages)
            for i in range(count):
                page = doc.get_page(i)
                bitmap = page.render(scale=2.0)
                pil = bitmap.to_pil()
                page.close()
                yield i + 1, pil
            doc.close()
            return
        except Exception as exc:
            raise RuntimeError("Vision OCR page rendering failed (need pdf2image+poppler or pypdfium2).") from exc

    def _ocr_pages_with_placeholder(self, reader: PdfReader, max_pages: int) -> List[TextBlock]:
        blocks: List[TextBlock] = []
        for i, page in enumerate(reader.pages, start=1):
            if i > max_pages:
                break
            width = float(page.mediabox.width or 612)
            height = float(page.mediabox.height or 792)
            blocks.append(
                TextBlock(
                    content=f"[vision-placeholder] Page {i}: OCR text would be inserted here from the vision model.",
                    page_number=i,
                    bbox=BoundingBox(x0=0.0, y0=0.0, x1=width, y1=height),
                    section_hint="vision extraction",
                    reading_order=i,
                )
            )
        return blocks

    def _openai_compatible_chat_completion(
        self,
        prompt: str,
        image_data_url: str,
        provider: dict[str, Any],
    ) -> tuple[str, int]:
        model = str(provider.get("model", "openai/gpt-4o-mini"))
        temperature = float(provider.get("temperature", 0.0))
        max_tokens = int(provider.get("max_output_tokens", 700))
        api_base = str(provider.get("api_base", "https://openrouter.ai/api/v1")).rstrip("/")
        api_key_env = str(provider.get("api_key_env", "OPENROUTER_API_KEY"))
        api_key = os.environ.get(api_key_env, "").strip()
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        # Try common OpenAI-compatible multimodal payload variants used by local servers.
        variants = [
            {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_data_url}},
                        ],
                    }
                ],
            },
            {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": image_data_url},
                        ],
                    }
                ],
            },
            {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image", "image_url": image_data_url},
                        ],
                    }
                ],
            },
        ]
        last_error = ""
        for payload in variants:
            for _ in range(2):
                req = urllib.request.Request(
                    f"{api_base}/chat/completions",
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers,
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(req, timeout=180) as resp:
                        body = json.loads(resp.read().decode("utf-8", errors="replace"))
                    return self._extract_chat_content_and_usage(body)
                except urllib.error.HTTPError as exc:
                    raw = ""
                    try:
                        raw = exc.read().decode("utf-8", errors="replace")
                    except Exception:
                        raw = ""
                    last_error = f"http_{exc.code}:{raw[:240]}"
                    continue
                except Exception as exc:
                    last_error = f"{exc.__class__.__name__}:{str(exc)[:240]}"
                    continue
        raise RuntimeError(f"chat_completions_failed:{last_error}")

    def _normalize_image_for_vlm(self, img: Any) -> Any:
        """
        Downscale very large page renders to reduce payload size and improve local VLM stability.
        """
        try:
            max_dim = max(256, int(self.max_image_dimension))
            w, h = img.size
            if max(w, h) <= max_dim:
                return img
            resized = img.copy()
            resized.thumbnail((max_dim, max_dim))
            return resized
        except Exception:
            return img

    def _extract_chat_content_and_usage(self, body: object) -> tuple[str, int]:
        if not isinstance(body, dict):
            return "", 0
        choices = body.get("choices", [])
        if not choices:
            return "", int(body.get("usage", {}).get("total_tokens", 0) or 0)

        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, list):
            content = "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
        usage = body.get("usage", {}) if isinstance(body, dict) else {}
        return str(content), int(usage.get("total_tokens", 0) or 0)

    def _estimate_ocr_confidence(self, blocks: List[TextBlock], profile: DocumentProfile, max_pages: int) -> float:
        if not blocks or max_pages <= 0:
            return 0.0
        lengths = [len((b.content or "").strip()) for b in blocks]
        non_empty = sum(1 for n in lengths if n > 20)
        non_empty_ratio = non_empty / max_pages
        avg_chars = sum(lengths) / max(1, len(lengths))
        avg_chars_score = min(1.0, avg_chars / 1200.0)
        base = (0.65 * non_empty_ratio) + (0.35 * avg_chars_score)
        if self._enum_value(getattr(profile, "origin_type", "")) == "scanned_image":
            base += 0.05
        return max(0.0, min(1.0, base))

    def _extract_tables_from_ocr_blocks(self, blocks: List[TextBlock]) -> list:
        out = []
        for b in blocks:
            txt = (b.content or "").strip()
            if not txt:
                continue
            if txt.startswith("[vision-placeholder]") or txt.startswith("[vision-budget-stop]"):
                continue
            width = max(1.0, float(b.bbox.x1 - b.bbox.x0))
            height = max(1.0, float(b.bbox.y1 - b.bbox.y0))
            out.extend(self._detect_pipe_tables(txt, b.page_number, width, height))
        return out

    def _build_provider_chain(self) -> list[dict[str, Any]]:
        providers_raw = self.vision_cfg.get("providers")
        providers: list[dict[str, Any]] = []
        if isinstance(providers_raw, list):
            for idx, p in enumerate(providers_raw):
                if not isinstance(p, dict):
                    continue
                providers.append(
                    {
                        "name": str(p.get("name", f"provider_{idx + 1}")),
                        "enabled": bool(p.get("enabled", False)),
                        "api_base": str(p.get("api_base", "https://openrouter.ai/api/v1")).rstrip("/"),
                        "model": str(p.get("model", "openai/gpt-4o-mini")),
                        "api_key_env": str(p.get("api_key_env", "OPENROUTER_API_KEY")),
                        "max_output_tokens": int(p.get("max_output_tokens", 700)),
                        "temperature": float(p.get("temperature", 0.0)),
                    }
                )
        if providers:
            return providers

        # Backward-compatible single-provider config.
        openrouter_cfg = self.vision_cfg.get("openrouter") or {}
        return [
            {
                "name": "openrouter",
                "enabled": bool(openrouter_cfg.get("enabled", False)),
                "api_base": str(openrouter_cfg.get("api_base", "https://openrouter.ai/api/v1")).rstrip("/"),
                "model": str(openrouter_cfg.get("model", "openai/gpt-4o-mini")),
                "api_key_env": str(openrouter_cfg.get("api_key_env", "OPENROUTER_API_KEY")),
                "max_output_tokens": int(openrouter_cfg.get("max_output_tokens", 700)),
                "temperature": float(openrouter_cfg.get("temperature", 0.0)),
            }
        ]

    def _provider_cost_per_page(self, provider: dict[str, Any]) -> float:
        """Local endpoints are treated as zero-cost for budget accounting."""
        if self._is_local_provider(provider):
            return 0.0
        return float(self.cost_per_page_usd)

    @staticmethod
    def _enum_value(v: Any) -> str:
        """Normalize enums/objects to their semantic value for robust comparisons."""
        if hasattr(v, "value"):
            return str(getattr(v, "value")).strip()
        return str(v).strip()

    @staticmethod
    def _is_local_provider(provider: dict[str, Any]) -> bool:
        name = str(provider.get("name", "")).lower()
        api_base = str(provider.get("api_base", "")).lower()
        local_markers = ("localhost", "127.0.0.1", "192.168.", "10.", "172.16.")
        if any(marker in name for marker in ("lmstudio", "local")):
            return True
        return any(marker in api_base for marker in local_markers)

    def _affordable_pages(self, total_pages: int) -> int:
        if self.cost_per_page_usd <= 0:
            cost_cap_pages = total_pages
        else:
            cost_cap_pages = int(math.floor((self.max_total_cost_usd / self.cost_per_page_usd) + 1e-9))
        return max(0, min(total_pages, self.max_pages, cost_cap_pages))

    def _merge_with_strategy_config(self, base_cfg: dict, raw_cfg: dict) -> dict:
        cfg = dict(base_cfg)
        strategy_path = str(cfg.get("strategy_config_path", "")).strip()
        if not strategy_path:
            return cfg
        p = Path(strategy_path)
        if not p.exists():
            return cfg
        try:
            loaded = yaml.safe_load(p.read_text(encoding="utf-8"))
        except Exception:
            return cfg
        if not isinstance(loaded, dict):
            return cfg

        strategy_cfg = loaded.get("vision", loaded)
        if not isinstance(strategy_cfg, dict):
            return cfg

        for key in (
            "require_model_for_ocr",
            "page_extraction_prompt",
            "confidence_if_ocr_only",
            "confidence_if_text_present_min",
            "min_confidence_for_accept",
            "escalate_on_low_confidence",
            "allow_best_effort_on_low_confidence",
            "providers",
        ):
            if key in strategy_cfg:
                cfg[key] = strategy_cfg[key]

        openrouter_cfg = dict(cfg.get("openrouter", {}))
        extra_openrouter = strategy_cfg.get("openrouter")
        if isinstance(extra_openrouter, dict):
            openrouter_cfg.update(extra_openrouter)
        cfg["openrouter"] = openrouter_cfg

        # Explicit runtime config takes precedence over file settings.
        for key, val in raw_cfg.items():
            if key == "openrouter" and isinstance(val, dict):
                merged_openrouter = dict(cfg.get("openrouter", {}))
                merged_openrouter.update(val)
                cfg["openrouter"] = merged_openrouter
            else:
                cfg[key] = val
        # If caller explicitly passes openrouter config but not provider chain,
        # avoid unintentionally inheriting strategy-file providers.
        if ("openrouter" in raw_cfg) and ("providers" not in raw_cfg):
            cfg["providers"] = []
        return cfg
