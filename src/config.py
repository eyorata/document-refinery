from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional at import-time for robustness
    load_dotenv = None


_ENV_LOADED = False


def _load_env_file_fallback(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        val = val.strip().strip("'").strip('"')
        # Normalize accidental literal PowerShell escape remnants like "`n".
        if val.endswith("`n"):
            val = val[:-2]
        if key not in os.environ:
            os.environ[key] = val


class TriageThresholds(BaseModel):
    low_density_threshold: float = 0.0002
    high_density_threshold: float = 0.001
    image_heavy_threshold: float = 0.6
    native_digital_max_image_ratio: float = 0.3
    max_images_for_ratio: int = 10
    target_chars_per_page: int = 350
    target_density: float = 0.001
    short_line_min_chars: int = 8
    short_line_max_chars: int = 32
    table_heavy_min_pipe_lines: int = 5
    multi_column_short_line_ratio: float = 0.45
    multi_column_short_line_min_count: int = 5
    triage_confidence_base: float = 0.5
    triage_confidence_origin_bonus: float = 0.25
    triage_confidence_layout_bonus: float = 0.25
    default_language_confidence: float = 0.85


class TriageConfig(BaseModel):
    thresholds: TriageThresholds
    domain_keywords: dict[str, list[str]] = Field(default_factory=dict)


class VlmBudgetConfig(BaseModel):
    enabled: bool = True
    max_pages_per_document: int = Field(default=25, ge=1)
    cost_per_page_usd: float = Field(default=0.01, ge=0.0)
    max_total_cost_usd: float = Field(default=0.20, ge=0.0)
    stop_on_budget_exceeded: bool = True
    allow_partial_processing: bool = False


class FastTextConfig(BaseModel):
    estimated_cost_usd: float = Field(default=0.002, ge=0.0)
    confidence_weight_chars: float = Field(default=0.45, ge=0.0, le=1.0)
    confidence_weight_density: float = Field(default=0.35, ge=0.0, le=1.0)
    confidence_weight_image_penalty: float = Field(default=0.20, ge=0.0, le=1.0)
    confidence_weight_font_metadata: float = Field(default=0.10, ge=0.0, le=1.0)
    table_bbox_height_ratio: float = Field(default=0.45, ge=0.0, le=1.0)
    table_min_columns: int = Field(default=3, ge=2)
    table_min_lines: int = Field(default=2, ge=2)


class VisionConfig(BaseModel):
    confidence_if_text_present_min: float = Field(default=0.9, ge=0.0, le=1.0)
    confidence_if_ocr_only: float = Field(default=0.88, ge=0.0, le=1.0)
    figure_bbox_height: float = Field(default=260.0, ge=0.0)
    require_model_for_ocr: bool = True
    strategy_config_path: str = "rubric/vision_strategy.yaml"
    page_extraction_prompt: str = (
        "Extract visible document text and tables from this page. "
        "Return plain text with table rows represented using pipe separators."
    )
    openrouter: dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": False,
            "api_base": "https://openrouter.ai/api/v1",
            "model": "openai/gpt-4o-mini",
            "api_key_env": "OPENROUTER_API_KEY",
            "max_output_tokens": 700,
            "temperature": 0.0,
        }
    )


class LayoutAdapterConfig(BaseModel):
    provider: str = "heuristic"
    options: dict[str, Any] = Field(default_factory=dict)


class LayoutConfig(BaseModel):
    confidence_if_tables_present: float = Field(default=0.72, ge=0.0, le=1.0)
    estimated_cost_usd: float = Field(default=0.02, ge=0.0)
    adapter: LayoutAdapterConfig = Field(default_factory=LayoutAdapterConfig)


class EscalationConfig(BaseModel):
    continue_on_strategy_error: bool = True
    require_human_review_on_low_confidence: bool = True
    chains: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "fast_text": ["fast_text", "layout_aware", "vision_augmented"],
            "layout_aware": ["layout_aware", "vision_augmented"],
            "vision_augmented": ["vision_augmented"],
        }
    )


class ExtractionConfig(BaseModel):
    confidence_minimum: float = Field(default=0.65, ge=0.0, le=1.0)
    budget_per_document_usd: float = Field(default=0.20, ge=0.0)
    strategy_budgets_usd: dict[str, float] = Field(
        default_factory=lambda: {"fast_text": 0.05, "layout_aware": 0.15, "vision_augmented": 0.20}
    )
    strategy_estimated_costs_usd: dict[str, float] = Field(
        default_factory=lambda: {"fast_text": 0.002, "layout_aware": 0.02, "vision_augmented": 0.10}
    )
    enforce_hard_caps: bool = True
    vlm_budget: VlmBudgetConfig = Field(default_factory=VlmBudgetConfig)
    fast_text: FastTextConfig = Field(default_factory=FastTextConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    layout: LayoutConfig = Field(default_factory=LayoutConfig)
    escalation: EscalationConfig = Field(default_factory=EscalationConfig)


class ChunkingConfig(BaseModel):
    max_tokens: int = Field(default=500, ge=1)
    rules: list[str] = Field(default_factory=list)


class PageIndexConfig(BaseModel):
    llm_summaries_enabled: bool = True
    llm: dict[str, Any] = Field(
        default_factory=lambda: {
            "provider": "openrouter",  # openrouter | gemini
            "enabled": False,
            "api_base": "https://openrouter.ai/api/v1",  # used for openrouter
            "model": "openai/gpt-4o-mini",
            "api_key_env": "OPENROUTER_API_KEY",
            "max_output_tokens": 140,
            "temperature": 0.1,
        }
    )


class QueryAgentConfig(BaseModel):
    router: dict[str, Any] = Field(
        default_factory=lambda: {
            "provider": "heuristic",  # heuristic | openrouter | gemini
            "enabled": False,
            "api_base": "https://openrouter.ai/api/v1",  # used for openrouter
            "model": "openai/gpt-4o-mini",
            "api_key_env": "OPENROUTER_API_KEY",
            "max_output_tokens": 250,
            "temperature": 0.0,
        }
    )


class StorageConfig(BaseModel):
    vector_store: dict[str, Any] = Field(
        default_factory=lambda: {
            "backend": "simple",  # simple | faiss
            "embedding_dim": 256,
            "similarity": "cosine",
        }
    )


class RefineryConfig(BaseModel):
    triage: TriageConfig
    extraction: ExtractionConfig
    chunking: ChunkingConfig
    pageindex: PageIndexConfig = Field(default_factory=PageIndexConfig)
    query_agent: QueryAgentConfig = Field(default_factory=QueryAgentConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)


def load_config(path: str | Path) -> dict[str, Any]:
    global _ENV_LOADED
    if not _ENV_LOADED:
        # Load .env from project root/current working directory.
        if load_dotenv is not None:
            load_dotenv()
        else:
            _load_env_file_fallback(".env")
        _ENV_LOADED = True

    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a mapping")
    try:
        validated = RefineryConfig.model_validate(cfg).model_dump(mode="python")
        return _apply_env_overrides(validated)
    except ValidationError as exc:
        raise ValueError(f"Invalid configuration in {p}: {exc}") from exc


def _apply_env_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Optional env-based overrides for local model serving (e.g., LM Studio).
    """
    out = cfg
    lm_base = os.getenv("LMSTUDIO_API_BASE", "").strip().rstrip("/")
    lm_api_key_env = os.getenv("LMSTUDIO_API_KEY_ENV", "").strip()
    vision_model = os.getenv("LMSTUDIO_VISION_MODEL", "").strip()
    reasoning_model = os.getenv("LMSTUDIO_REASONING_MODEL", "").strip()
    use_lm_for_vision = os.getenv("USE_LMSTUDIO_FOR_VISION", "false").strip().lower() in {"1", "true", "yes"}
    use_lm_for_pageindex = os.getenv("USE_LMSTUDIO_FOR_PAGEINDEX", "false").strip().lower() in {"1", "true", "yes"}
    use_lm_for_router = os.getenv("USE_LMSTUDIO_FOR_ROUTER", "false").strip().lower() in {"1", "true", "yes"}

    if lm_base:
        if use_lm_for_vision:
            vis_or = out["extraction"]["vision"]["openrouter"]
            vis_or["enabled"] = True
            vis_or["api_base"] = lm_base
            if vision_model:
                vis_or["model"] = vision_model
            if lm_api_key_env:
                vis_or["api_key_env"] = lm_api_key_env
        if use_lm_for_pageindex:
            llm_cfg = out["pageindex"]["llm"]
            llm_cfg["provider"] = "openrouter"
            llm_cfg["enabled"] = True
            llm_cfg["api_base"] = lm_base
            if reasoning_model:
                llm_cfg["model"] = reasoning_model
            if lm_api_key_env:
                llm_cfg["api_key_env"] = lm_api_key_env
        if use_lm_for_router:
            router_cfg = out["query_agent"]["router"]
            router_cfg["provider"] = "openrouter"
            router_cfg["enabled"] = True
            router_cfg["api_base"] = lm_base
            if reasoning_model:
                router_cfg["model"] = reasoning_model
            if lm_api_key_env:
                router_cfg["api_key_env"] = lm_api_key_env
    return out
