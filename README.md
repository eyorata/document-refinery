# Document Intelligence Refinery

Production-oriented baseline for the Week 3 TRP1 challenge: a typed, multi-stage, strategy-routed document intelligence pipeline.

## What this implements

- 5-stage architecture:
1. `TriageAgent` (`src/agents/triage.py`)
2. `ExtractionRouter` + strategies (`src/agents/extractor.py`, `src/strategies/*`)
3. `ChunkingEngine` + `ChunkValidator` (`src/agents/chunker.py`)
4. `PageIndexBuilder` (`src/agents/indexer.py`)
5. `QueryInterfaceAgent` (`src/agents/query_agent.py`)

- Core typed models in `src/models/schemas.py`:
- `DocumentProfile`, `ExtractedDocument`, `LDU`, `PageIndexNode`, `ProvenanceChain`, and related schemas.

- Externalized routing and chunking rules:
- `rubric/extraction_rules.yaml`

- Artifacts emitted to `.refinery/`:
- `profiles/{doc_id}.json`
- `extraction_ledger.jsonl`
- `pageindex/{doc_id}.json`
- `facts.db`

## Setup

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -e .[dev]
```

## Run

```bash
refinery "path/to/document.pdf"
```

or

```bash
python -m src.cli "path/to/document.pdf"
```

## Test

```bash
pytest
```

## Design notes

- Strategy A/B/C are implemented with confidence-gated escalation and budget guard.
- Budget guard now enforces both post-run spend and preflight projected spend before expensive strategies run.
- Provenance includes page + bbox + content hash.
- Query flow uses PageIndex navigation before semantic search.
- Fact extraction persists key-value signals to SQLite for structured queries.

## Configuration reference

All runtime behavior is driven from `rubric/extraction_rules.yaml`, validated through typed Pydantic models at startup (`src/config.py`).

### `triage.thresholds`
- `low_density_threshold` (`float`): below this density and image-heavy implies scanned.
- `high_density_threshold` (`float`): above this density and low-image implies native-digital.
- `image_heavy_threshold` (`float`): ratio threshold for scanned-image signal.
- `max_images_for_ratio` (`int`): normalization denominator for image ratio.
- `target_chars_per_page` (`int`): fast-text confidence calibration.
- `target_density` (`float`): fast-text confidence calibration.

### `extraction`
- `confidence_minimum` (`0..1`): minimum confidence to avoid human-review escalation.
- `budget_per_document_usd` (`float`): hard total cost cap per document.
- `enforce_hard_caps` (`bool`): enables projected preflight caps before each strategy attempt.
- `strategy_budgets_usd` (`map[str,float]`): per-strategy hard caps.
- `strategy_estimated_costs_usd` (`map[str,float]`): projected costs for preflight checks.
- `vlm_budget.enabled` (`bool`): toggle vision strategy execution.
- `vlm_budget.max_pages_per_document` (`int`): hard max pages eligible for VLM.
- `vlm_budget.cost_per_page_usd` (`float`): page-based VLM estimate used for preflight/post-checks.
- `vlm_budget.max_total_cost_usd` (`float`): hard VLM cap per document.
- `vlm_budget.stop_on_budget_exceeded` (`bool`): hard-stop vision when caps are exceeded.
- `vlm_budget.allow_partial_processing` (`bool`): allow partial page processing when hard-stop is disabled.
- `fast_text.*`: confidence weights and table-detection thresholds for strategy A.
- `layout.confidence_if_tables_present` (`float`): confidence floor for strategy B when tables are found.
- `layout.estimated_cost_usd` (`float`): returned cost estimate for strategy B.
- `layout.adapter.provider` (`heuristic|docling|external_payload|mineru`): selected layout tool adapter.
- `layout.adapter.options.strict` (`bool`): when `provider=docling`, fail fast if Docling is unavailable/errors instead of fallback.
- `layout.adapter.options.payload_json_path` (`str`): for `external_payload`, path to external tool JSON tables.
- `vision.*`: confidence tuning and synthetic provenance geometry for strategy C.
- `escalation.continue_on_strategy_error` (`bool`): continue chain on strategy failure.
- `escalation.require_human_review_on_low_confidence` (`bool`): enforce review for low-confidence final output.
- `escalation.chains` (`map[str,list[str]]`): explicit per-entry strategy chain order.

### `chunking`
- `max_tokens` (`int`): max tokens per LDU chunk.
- `rules` (`list[str]`): rule toggles for chunk assembly behavior.

## Current limitations

- `docling` is wired with optional dependency + heuristic fallback; `mineru` remains an extension point.
- Vision extraction keeps an explicit budget/paging guard but OCR/VLM calls are still placeholder by default.
- Semantic retrieval currently uses a local cosine-over-token baseline vector store.
- Section summaries are heuristic; swap in cheap LLM call if available.

## Next hardening steps

1. Integrate Docling or MinerU adapter for robust layout extraction.
2. Add OpenRouter-backed multimodal vision extractor with token accounting.
3. Add evaluation harness for retrieval precision and table extraction metrics.
4. Add Dockerfile and scripted demo workflow.
