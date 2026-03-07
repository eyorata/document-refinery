from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from src.models import BoundingBox, LDU, PageIndexNode, ProvenanceChain, ProvenanceItem, QueryAnswer
from src.storage import FactTableStore
from src.storage.vector_store import VectorStore
from src.utils.hashing import stable_hash
from src.utils.llm_client import call_chat_text_openai_compatible, should_use_langchain_wrapper


@dataclass
class ToolDecision:
    next_tool: str
    sql: str | None = None
    reason: str = ""


class HeuristicToolRouter:
    def decide(self, question: str, state: dict[str, Any]) -> ToolDecision:
        q = question.lower()
        if state.get("step_count", 0) == 0:
            return ToolDecision(next_tool="pageindex_navigate", reason="Start with section navigation.")
        if not state.get("semantic_attempted", False):
            return ToolDecision(next_tool="semantic_search", reason="Run vector retrieval.")
        if state.get("semantic_hits"):
            return ToolDecision(next_tool="finish", reason="Semantic evidence collected.")
        if (not state.get("structured_attempted", False)) and any(
            tok in q for tok in ["sql", "table", "revenue", "amount", "q1", "q2", "q3", "q4"]
        ):
            sql = self._default_fact_sql(question)
            return ToolDecision(next_tool="structured_query", sql=sql, reason="Numerical question routed to SQL.")
        return ToolDecision(next_tool="finish")

    def _default_fact_sql(self, question: str) -> str:
        safe = "".join(ch for ch in question.lower() if ch.isalnum() or ch.isspace()).strip()
        terms = [t for t in safe.split() if len(t) > 2][:4]
        if not terms:
            return "SELECT key, value, page, content_hash FROM facts LIMIT 5"
        clauses = " OR ".join([f"key LIKE '%{t}%'" for t in terms])
        return f"SELECT key, value, page, content_hash FROM facts WHERE {clauses} LIMIT 5"


class OpenRouterToolRouter:
    def __init__(
        self,
        api_key_env: str = "OPENROUTER_API_KEY",
        api_base: str = "https://openrouter.ai/api/v1",
        model: str = "openai/gpt-4o-mini",
    ) -> None:
        self.api_key_env = api_key_env
        self.api_base = api_base.rstrip("/")
        self.model = model

    def decide(self, question: str, state: dict[str, Any]) -> ToolDecision:
        api_key = os.environ.get(self.api_key_env, "").strip()

        prompt = (
            "You are routing a query agent with tools: pageindex_navigate, semantic_search, structured_query, finish.\n"
            "Return strict JSON: {\"next_tool\":\"...\",\"sql\":\"...|null\",\"reason\":\"...\"}.\n"
            "Choose one tool only.\n"
            f"Question: {question}\n"
            f"State: {json.dumps(self._state_snapshot(state), ensure_ascii=True)}"
        )
        payload = {
            "model": self.model,
            "temperature": 0.0,
            "max_tokens": 250,
            "messages": [{"role": "user", "content": prompt}],
        }
        if should_use_langchain_wrapper():
            try:
                content = call_chat_text_openai_compatible(
                    prompt,
                    model=self.model,
                    api_base=self.api_base,
                    api_key=api_key,
                    max_tokens=250,
                    temperature=0.0,
                )
                data = self._parse_json(content)
                next_tool = str(data.get("next_tool", "semantic_search")).strip().lower()
                if next_tool not in {"pageindex_navigate", "semantic_search", "structured_query", "finish"}:
                    next_tool = "semantic_search"
                sql = data.get("sql")
                return ToolDecision(next_tool=next_tool, sql=str(sql) if sql else None, reason=str(data.get("reason", "")))
            except Exception:
                pass

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        req = urllib.request.Request(
            f"{self.api_base}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"OpenRouter routing HTTP error: {exc.code}") from exc

        content = ""
        choices = body.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            content = msg.get("content", "") if isinstance(msg, dict) else ""
        if not isinstance(content, str):
            content = str(content)
        data = self._parse_json(content)
        next_tool = str(data.get("next_tool", "semantic_search")).strip().lower()
        if next_tool not in {"pageindex_navigate", "semantic_search", "structured_query", "finish"}:
            next_tool = "semantic_search"
        sql = data.get("sql")
        return ToolDecision(next_tool=next_tool, sql=str(sql) if sql else None, reason=str(data.get("reason", "")))

    def _state_snapshot(self, state: dict[str, Any]) -> dict[str, Any]:
        return {
            "step_count": state.get("step_count", 0),
            "selected_pages": state.get("selected_pages", []),
            "has_semantic_hits": bool(state.get("semantic_hits")),
            "has_structured_rows": bool(state.get("structured_rows")),
        }

    def _parse_json(self, text: str) -> dict[str, Any]:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`")
            if "\n" in stripped:
                stripped = stripped.split("\n", 1)[1]
        if "{" in stripped and "}" in stripped:
            stripped = stripped[stripped.find("{") : stripped.rfind("}") + 1]
        try:
            obj = json.loads(stripped)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        return {}


class GeminiToolRouter:
    def __init__(
        self,
        api_key_env: str = "GEMINI_API_KEY",
        model: str = "gemini-1.5-flash",
        max_output_tokens: int = 250,
        temperature: float = 0.0,
    ) -> None:
        self.api_key_env = api_key_env
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature

    def decide(self, question: str, state: dict[str, Any]) -> ToolDecision:
        api_key = os.environ.get(self.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"Missing Gemini API key in env var `{self.api_key_env}`.")
        prompt = (
            "Route to one tool: pageindex_navigate, semantic_search, structured_query, finish.\n"
            "Return strict JSON object with fields next_tool, sql (or null), reason.\n"
            f"Question: {question}\n"
            f"State: {json.dumps(OpenRouterToolRouter()._state_snapshot(state), ensure_ascii=True)}"
        )
        req = urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={api_key}",
            data=json.dumps(
                {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": self.temperature,
                        "maxOutputTokens": self.max_output_tokens,
                    },
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Gemini routing HTTP error: {exc.code}") from exc

        text = self._extract_text(body)
        data = OpenRouterToolRouter()._parse_json(text)
        next_tool = str(data.get("next_tool", "semantic_search")).strip().lower()
        if next_tool not in {"pageindex_navigate", "semantic_search", "structured_query", "finish"}:
            next_tool = "semantic_search"
        sql = data.get("sql")
        return ToolDecision(next_tool=next_tool, sql=str(sql) if sql else None, reason=str(data.get("reason", "")))

    def _extract_text(self, body: object) -> str:
        if not isinstance(body, dict):
            return ""
        cands = body.get("candidates", [])
        if not cands:
            return ""
        cand0 = cands[0] if isinstance(cands[0], dict) else {}
        content = cand0.get("content", {}) if isinstance(cand0, dict) else {}
        parts = content.get("parts", []) if isinstance(content, dict) else []
        return " ".join(str(p.get("text", "")) for p in parts if isinstance(p, dict)).strip()


class QueryInterfaceAgent:
    def __init__(self, vector_store: VectorStore, fact_table: FactTableStore, router_cfg: dict | None = None) -> None:
        self.vector_store = vector_store
        self.fact_table = fact_table
        self.router_cfg = router_cfg or {}

    def pageindex_navigate(self, index: PageIndexNode, topic: str, k: int = 3) -> list[PageIndexNode]:
        q_tokens = self._tokens(topic)
        scored = []
        for node in index.child_sections:
            node_blob = " ".join([node.title, node.summary, " ".join(node.key_entities)]).lower()
            n_tokens = self._tokens(node_blob)
            score = len(q_tokens.intersection(n_tokens))
            scored.append((score, node))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [n for s, n in scored[:k]]

    def semantic_search(self, query: str, top_k: int = 3, allowed_pages: set[int] | None = None) -> list[LDU]:
        return self.vector_store.search(query, top_k=top_k, filter_pages=allowed_pages)

    def structured_query(self, sql: str):
        return self.fact_table.query(sql)

    def tools(self) -> dict[str, object]:
        return {
            "pageindex_navigate": self.pageindex_navigate,
            "semantic_search": self.semantic_search,
            "structured_query": self.structured_query,
        }

    def build_langgraph(self, question: str, page_index: PageIndexNode, max_steps: int = 4):
        try:
            from langgraph.graph import END, StateGraph  # type: ignore[import-not-found]
        except Exception:
            return None

        router = self._build_router()

        def merged(state: dict[str, Any], **updates: Any) -> dict[str, Any]:
            out = dict(state)
            out.update(updates)
            return out

        def route_node(state: dict[str, Any]) -> dict[str, Any]:
            if int(state.get("step_count", 0)) >= max_steps:
                return merged(state, next_tool="finish")
            try:
                decision = router.decide(state["question"], state)
            except Exception:
                decision = HeuristicToolRouter().decide(state["question"], state)
            update: dict[str, Any] = {
                "next_tool": decision.next_tool,
                "step_count": int(state.get("step_count", 0)) + 1,
                "routing_reasons": list(state.get("routing_reasons", [])) + [decision.reason],
            }
            if decision.sql:
                update["sql"] = decision.sql
            return merged(state, **update)

        def pageindex_node(state: dict[str, Any]) -> dict[str, Any]:
            nav = self.pageindex_navigate(state["page_index"], state["question"], k=3)
            pages: set[int] = set()
            for node in nav:
                pages.update(range(node.page_start, node.page_end + 1))
            return merged(state, selected_pages=sorted(pages))

        def semantic_node(state: dict[str, Any]) -> dict[str, Any]:
            pages = set(state.get("selected_pages", []))
            hits = self.semantic_search(state["question"], top_k=3, allowed_pages=pages if pages else None)
            return merged(state, semantic_hits=hits, semantic_attempted=True)

        def structured_node(state: dict[str, Any]) -> dict[str, Any]:
            sql = state.get("sql")
            if not sql:
                sql = HeuristicToolRouter()._default_fact_sql(state["question"])
            try:
                rows = self.structured_query(sql)
            except Exception:
                rows = []
            return merged(state, structured_rows=rows, sql=sql, structured_attempted=True)

        def synth_node(state: dict[str, Any]) -> dict[str, Any]:
            answer = self._synthesize_answer(state["semantic_hits"], state.get("structured_rows", []))
            prov = self._build_provenance(state["semantic_hits"], state.get("structured_rows", []))
            return merged(state, final_answer=answer, final_provenance=prov)

        def route_cond(state: dict[str, Any]) -> str:
            nxt = str(state.get("next_tool", "finish"))
            if nxt in {"pageindex_navigate", "semantic_search", "structured_query", "finish"}:
                return nxt
            return "finish"

        graph = StateGraph(dict)
        graph.add_node("route", route_node)
        graph.add_node("pageindex_navigate", pageindex_node)
        graph.add_node("semantic_search", semantic_node)
        graph.add_node("structured_query", structured_node)
        graph.add_node("synthesize", synth_node)
        graph.set_entry_point("route")
        graph.add_conditional_edges(
            "route",
            route_cond,
            {
                "pageindex_navigate": "pageindex_navigate",
                "semantic_search": "semantic_search",
                "structured_query": "structured_query",
                "finish": "synthesize",
            },
        )
        graph.add_edge("pageindex_navigate", "route")
        graph.add_edge("semantic_search", "route")
        graph.add_edge("structured_query", "route")
        graph.add_edge("synthesize", END)
        return graph.compile()

    def answer(self, question: str, page_index: PageIndexNode) -> QueryAnswer:
        app = self.build_langgraph(question=question, page_index=page_index, max_steps=4)
        if app is None:
            # Fallback if langgraph is not available.
            return self._fallback_answer(question, page_index)
        state = {
            "question": question,
            "page_index": page_index,
            "selected_pages": [],
            "semantic_hits": [],
            "structured_rows": [],
            "step_count": 0,
            "next_tool": "pageindex_navigate",
            "routing_reasons": [],
            "sql": None,
            "semantic_attempted": False,
            "structured_attempted": False,
        }
        result = app.invoke(state)
        return QueryAnswer(
            answer=str(result.get("final_answer", "No verifiable answer found.")),
            provenance=result.get(
                "final_provenance",
                ProvenanceChain(bbox=BoundingBox(x0=0.0, y0=0.0, x1=0.0, y1=0.0), content_hash="", citations=[]),
            ),
        )

    def _build_router(self):
        provider = str(self.router_cfg.get("provider", "heuristic")).lower().strip()
        enabled = bool(self.router_cfg.get("enabled", False))
        if not enabled or provider == "heuristic":
            return HeuristicToolRouter()
        if provider == "openrouter":
            return OpenRouterToolRouter(
                api_key_env=str(self.router_cfg.get("api_key_env", "OPENROUTER_API_KEY")),
                api_base=str(self.router_cfg.get("api_base", "https://openrouter.ai/api/v1")),
                model=str(self.router_cfg.get("model", "openai/gpt-4o-mini")),
            )
        if provider == "gemini":
            return GeminiToolRouter(
                api_key_env=str(self.router_cfg.get("api_key_env", "GEMINI_API_KEY")),
                model=str(self.router_cfg.get("model", "gemini-1.5-flash")),
                max_output_tokens=int(self.router_cfg.get("max_output_tokens", 250)),
                temperature=float(self.router_cfg.get("temperature", 0.0)),
            )
        return HeuristicToolRouter()

    def audit_claim(self, claim: str, page_index: PageIndexNode) -> dict[str, object]:
        ans = self.answer(claim, page_index)
        status = "verified" if ans.provenance.citations else "unverifiable"
        return {"claim": claim, "status": status, "answer": ans.answer, "provenance": ans.provenance.model_dump()}

    def _fallback_answer(self, question: str, page_index: PageIndexNode) -> QueryAnswer:
        nav = self.pageindex_navigate(page_index, question, k=3)
        pages = set()
        for n in nav:
            pages.update(range(n.page_start, n.page_end + 1))
        hits = self.semantic_search(question, top_k=3, allowed_pages=pages if pages else None)
        answer = self._synthesize_answer(hits, [])
        provenance = self._build_provenance(hits, [])
        return QueryAnswer(answer=answer, provenance=provenance)

    def _synthesize_answer(self, hits: list[LDU], rows: list[tuple]) -> str:
        llm_answer = self._llm_answer(hits, rows)
        if llm_answer:
            return llm_answer
        parts: list[str] = []
        if hits:
            parts.append(" ".join(h.content[:220] for h in hits)[:600])
        if rows:
            kvs = []
            for row in rows[:5]:
                if len(row) >= 2:
                    kvs.append(f"{row[0]}={row[1]}")
            if kvs:
                parts.append("Structured facts: " + "; ".join(kvs))
        if not parts:
            return "No verifiable answer found."
        return " ".join(parts)[:900]

    def _llm_answer(self, hits: list[LDU], rows: list[tuple]) -> str | None:
        provider = str(self.router_cfg.get("provider", "heuristic")).lower().strip()
        enabled = bool(self.router_cfg.get("enabled", False))
        if not enabled or provider not in {"openrouter", "gemini"}:
            return None
        if not hits and not rows:
            return None

        context_parts: list[str] = []
        for h in hits[:6]:
            if h.page_refs:
                context_parts.append(f"[p{h.page_refs[0].page_number}] {h.content[:500]}")
            else:
                context_parts.append(h.content[:500])
        if rows:
            facts = []
            for row in rows[:8]:
                if len(row) >= 2:
                    facts.append(f"{row[0]}={row[1]}")
            if facts:
                context_parts.append("Structured facts: " + "; ".join(facts))
        context = "\n".join(context_parts)[:6000]
        instruction = (
            "Answer the user question using only provided context. "
            "Return only the final answer in 2-5 sentences. "
            "Do not include reasoning steps, tool/routing notes, or internal process text. "
            "If the context is insufficient, say so briefly."
        )
        prompt = f"{instruction}\n\nContext:\n{context}"

        try:
            if provider == "openrouter":
                api_base = str(self.router_cfg.get("api_base", "https://openrouter.ai/api/v1")).rstrip("/")
                model = str(self.router_cfg.get("model", "openai/gpt-4o-mini"))
                api_key_env = str(self.router_cfg.get("api_key_env", "OPENROUTER_API_KEY"))
                api_key = os.environ.get(api_key_env, "").strip()
                max_tokens = int(self.router_cfg.get("max_output_tokens", 250))
                temperature = float(self.router_cfg.get("temperature", 0.0))
                if should_use_langchain_wrapper():
                    text = call_chat_text_openai_compatible(
                        prompt,
                        model=model,
                        api_base=api_base,
                        api_key=api_key,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    if isinstance(text, str) and text.strip():
                        return self._clean_answer_text(text)
                    return None

                headers = {"Content-Type": "application/json"}
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                req = urllib.request.Request(
                    f"{api_base}/chat/completions",
                    data=json.dumps(
                        {
                            "model": model,
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                            "messages": [{"role": "user", "content": prompt}],
                        }
                    ).encode("utf-8"),
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    body = json.loads(resp.read().decode("utf-8", errors="replace"))
                choices = body.get("choices", [])
                if choices and isinstance(choices[0], dict):
                    content = choices[0].get("message", {}).get("content", "")
                    text = str(content).strip()
                    return self._clean_answer_text(text) if text else None
                return None

            api_key_env = str(self.router_cfg.get("api_key_env", "GEMINI_API_KEY"))
            api_key = os.environ.get(api_key_env, "").strip()
            if not api_key:
                return None
            model = str(self.router_cfg.get("model", "gemini-1.5-flash"))
            max_output_tokens = int(self.router_cfg.get("max_output_tokens", 250))
            temperature = float(self.router_cfg.get("temperature", 0.0))
            req = urllib.request.Request(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
                data=json.dumps(
                    {
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "temperature": temperature,
                            "maxOutputTokens": max_output_tokens,
                        },
                    }
                ).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read().decode("utf-8", errors="replace"))
            cands = body.get("candidates", [])
            if not cands:
                return None
            cand0 = cands[0] if isinstance(cands[0], dict) else {}
            content = cand0.get("content", {}) if isinstance(cand0, dict) else {}
            parts = content.get("parts", []) if isinstance(content, dict) else []
            text = " ".join(str(p.get("text", "")) for p in parts if isinstance(p, dict)).strip()
            return self._clean_answer_text(text) if text else None
        except Exception:
            return None

    def _build_provenance(self, hits: list[LDU], rows: list[tuple]) -> ProvenanceChain:
        cites: list[ProvenanceItem] = []
        for h in hits:
            if not h.page_refs:
                continue
            ref = h.page_refs[0]
            cites.append(
                ProvenanceItem(
                    document_name=ref.document_name,
                    page_number=ref.page_number,
                    bbox=ref.bbox,
                    content_hash=h.content_hash,
                )
            )

        # If semantic retrieval misses but SQL rows contain content_hash, recover citations from vector-store chunks.
        if not cites and rows:
            hashes: list[str] = []
            for row in rows:
                if len(row) >= 4 and isinstance(row[3], str):
                    hashes.append(row[3])
            for h in self.vector_store.get_by_hashes(hashes):
                if not h.page_refs:
                    continue
                ref = h.page_refs[0]
                cites.append(
                    ProvenanceItem(
                        document_name=ref.document_name,
                        page_number=ref.page_number,
                        bbox=ref.bbox,
                        content_hash=h.content_hash,
                    )
                )

        if not cites:
            return ProvenanceChain(bbox=BoundingBox(x0=0.0, y0=0.0, x1=0.0, y1=0.0), content_hash="", citations=[])
        return ProvenanceChain(
            bbox=self._aggregate_bbox(cites),
            content_hash=stable_hash("|".join(c.content_hash for c in cites)),
            citations=cites,
        )

    def _aggregate_bbox(self, citations: list[ProvenanceItem]) -> BoundingBox:
        if not citations:
            return BoundingBox(x0=0.0, y0=0.0, x1=0.0, y1=0.0)
        x0 = min(c.bbox.x0 for c in citations)
        y0 = min(c.bbox.y0 for c in citations)
        x1 = max(c.bbox.x1 for c in citations)
        y1 = max(c.bbox.y1 for c in citations)
        return BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)

    def _tokens(self, text: str) -> set[str]:
        return {t.strip(".,:;()[]{}\"'").lower() for t in text.split() if t.strip(".,:;()[]{}\"'")}

    def _clean_answer_text(self, text: str) -> str:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        drop_prefixes = (
            "route:",
            "step ",
            "provenancechain",
            "tool",
            "reasoning",
            "restrict semantic search",
            "collect top chunks",
        )
        kept = [ln for ln in lines if not ln.lower().startswith(drop_prefixes)]
        out = " ".join(kept).strip()
        return out or text.strip()
