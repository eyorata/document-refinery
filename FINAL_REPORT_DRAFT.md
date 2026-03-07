# Document Intelligence Refinery - Final Submission Report (Draft)

## 1. Executive Summary

This final submission extends the interim delivery into a full 5-stage document refinery pipeline with typed schemas, confidence-gated extraction routing, semantic chunking, PageIndex navigation, provenance-first Q&A, and auditable artifacts.

The system now supports:

- Stage 1: Triage profiling with origin/layout/domain/cost-tier classification
- Stage 2: Multi-strategy extraction with escalation (`A -> B -> C`) and budget guards
- Stage 3: Semantic chunking with rule validation and content hashing
- Stage 4: PageIndex construction + retrieval precision comparison (`with_pageindex` vs `without_pageindex`)
- Stage 5: Query agent with three tools (`pageindex_navigate`, `semantic_search`, `structured_query`) and provenance citations

Artifacts are persisted under `.refinery/`:

- `profiles/{doc_id}.json`
- `extracted/{doc_id}.json`
- `chunks/{doc_id}.json`
- `pageindex/{doc_id}.json`
- `pageindex_metrics/{doc_id}.json`
- `extraction_ledger.jsonl`
- `facts.db`

---

## 2. Interim to Final: What Was Refined

Compared to interim, this final iteration adds:

- Explicit strategy orchestration and escalation tracing (`initial_strategy`, `routing_trace`, `escalated_from`)
- Vision strategy provider chaining and provider-attempt diagnostics
- Local-model support for LM Studio (`qwen3-vl-32b-instruct`, `google/gemma-3-27b`)
- Stage 3 artifact persistence (`extracted`, `chunks`, `pageindex_metrics`)
- PageIndex precision evaluation helper persisted per processed document
- Query-agent output cleanup and improved pageindex navigation scoring
- Fact table ingestion for structured numeric retrieval
- UI filtering to current-upload artifacts and chat workflow improvements

---

## 3. Architecture (Final)

### Stage 1 - Triage Agent

Input document is profiled into `DocumentProfile`:

- `origin_type`: `native_digital | scanned_image | mixed | form_fillable`
- `layout_complexity`: `single_column | multi_column | table_heavy | figure_heavy | mixed`
- `language_code` + confidence
- `domain_hint`
- `estimated_extraction_cost`

Output is saved at `.refinery/profiles/{doc_id}.json`.

### Stage 2 - Multi-Strategy Extraction

- Strategy A: `fast_text` (`pdfplumber` + confidence score)
- Strategy B: `layout_aware` (Docling/MinerU adapter chain + heuristic fallback)
- Strategy C: `vision_augmented` (local/cloud multimodal OCR with budget guard)

Every run is logged to `.refinery/extraction_ledger.jsonl` including:

- strategy and initial strategy
- confidence, cost estimate, processing time
- escalation trace
- provider and token usage
- provider attempts

### Stage 3 - Semantic Chunking + Validation

`ChunkingEngine` emits LDUs with:

- content, `chunk_type`, `page_refs`, `bounding_box`
- `parent_section`, `token_count`, `content_hash`
- relationships (`related_chunk_hashes`)

`ChunkValidator` enforces rule constraints (table/figure/list/header/cross-reference integrity).

### Stage 4 - PageIndex Builder

Section-level tree is generated with:

- title + page range
- entities
- summary (LLM-backed when enabled, heuristic fallback)
- data types present

`top_sections(topic, k=3)` and precision evaluation (`with_pageindex` vs `without_pageindex`) are available.

### Stage 5 - Query Interface Agent

Three-tool flow:

1. `pageindex_navigate`
2. `semantic_search`
3. `structured_query`

Answers include `ProvenanceChain` with document/page/bbox/content_hash citations.

---

## 4. Extraction Quality Analysis

### 4.1 Ledger-level Snapshot (current workspace)

From `.refinery/extraction_ledger.jsonl`:

- Total extractions logged: **82**
- Strategy distribution:
  - `fast_text`: **44**
  - `layout_aware`: **17**
  - `vision_augmented`: **21**
- Provider distribution (non-null provider rows):
  - `placeholder`: **18**
  - `local_provider_failed_placeholder`: **2**
  - `lmstudio_qwen`: **1**

Interpretation:

- Strategy A/B are stable for many docs.
- Strategy C works on some runs but still has intermittent local-VLM execution reliability.

### 4.2 Table Extraction Quality (Proxy Evaluation)

Because full corpus gold table annotations are not yet packaged in this repo, a **proxy/silver** evaluation was run on processed documents:

- Ground truth proxy: `layout_complexity == table_heavy` implies table expected
- Prediction: extracted document contains `tables[]` entries

Evaluated set: **5** processed docs with saved extracted artifacts.

Confusion counts:

- TP = 2
- FP = 1
- FN = 0
- TN = 2

Metrics:

- Precision: **0.6667**
- Recall: **1.0000**
- F1: **0.8000**

Notes:

- Recall is currently high on this small set.
- Precision is reduced by one over-detection case (`single_column` doc with predicted tables).
- This should be replaced by corpus-wide gold-annotated evaluation before final grading submission.

---

## 5. PageIndex and Retrieval Effectiveness

From persisted metrics in `.refinery/pageindex_metrics/*.json` (current generated subset):

- Metric files: **4**
- Avg precision with PageIndex filtering: **0.25**
- Avg precision without PageIndex filtering: **0.25**

Interpretation:

- No improvement yet on the sampled subset.
- Main bottleneck is extraction quality and section granularity for scanned/complex docs.
- Further gains expected after improving OCR quality and section segmentation depth.

---

## 6. Lessons Learned (Failures and Fixes)

### Case 1 - Vision strategy reported success but produced placeholder text

Failure:

- Ledger rows showed `strategy_used=vision_augmented` but `provider=placeholder` and `token_usage=0`.

Root causes:

- Enum-value comparison bug prevented forced OCR for scanned docs in some routes.
- Missing render dependency path caused runtime failures before model call.

Fixes:

- Normalized enum comparisons for robust routing checks.
- Added rendering fallback to `pypdfium2`.
- Added provider-attempt diagnostics in ledger.
- Added multimodal payload variants for OpenAI-compatible endpoints.

### Case 2 - Config/profile mismatch in strategy selection

Failure:

- Profile indicated `needs_layout_model`, but extraction still remained on Strategy A.

Root cause:

- `initial_strategy_mode: always_fast_text` forced Strategy A start and high A-confidence prevented escalation.

Fixes:

- Added explicit routing override for `table_heavy` documents to start Strategy B.
- Clarified routing policy and config behavior.

### Case 3 - QA answers leaked routing/process text

Failure:

- Some answers included internal operational text instead of concise user-facing responses.

Fix:

- Updated synthesis prompting and post-processing cleanup to return final answer text only while preserving provenance.

---

## 7. Remaining Gaps Before Final PDF Submission

1. Replace proxy table evaluation with full-corpus precision/recall using gold labels.
2. Complete robustness tuning for local vision OCR to eliminate placeholder fallback in scanned cases.
3. Produce required evidence package:
   - At least 12 PageIndex artifacts (3 per class minimum)
   - 12 Q&A examples with full provenance (3 per class)
4. Consolidate this report into a single submission PDF with screenshots and demo evidence.

---

## 8. Appendix: Reproducibility Notes

- Config: `rubric/extraction_rules.yaml`, `rubric/vision_strategy.yaml`
- Runtime env: `.env` (LM Studio/OpenAI/Gemini toggles)
- Main entry points:
  - `src/pipeline.py`
  - `src/agents/*`
  - `streamlit_app.py`
- Tests: `pytest` (all passing in current workspace)

