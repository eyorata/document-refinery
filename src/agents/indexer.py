from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections import defaultdict
from typing import Callable

from src.models import LDU, PageIndexNode


class PageIndexBuilder:
    def __init__(self, pageindex_cfg: dict | None = None) -> None:
        cfg = pageindex_cfg or {}
        self.llm_summaries_enabled = bool(cfg.get("llm_summaries_enabled", True))
        openrouter = cfg.get("openrouter", {}) or {}
        self.openrouter_enabled = bool(openrouter.get("enabled", False))
        self.openrouter_api_base = str(openrouter.get("api_base", "https://openrouter.ai/api/v1")).rstrip("/")
        self.openrouter_model = str(openrouter.get("model", "openai/gpt-4o-mini"))
        self.openrouter_api_key_env = str(openrouter.get("api_key_env", "OPENROUTER_API_KEY"))
        self.openrouter_max_output_tokens = int(openrouter.get("max_output_tokens", 140))

    def build(self, chunks: list[LDU]) -> PageIndexNode:
        if not chunks:
            return PageIndexNode(
                title="Document",
                page_start=1,
                page_end=1,
                child_sections=[],
                key_entities=[],
                summary="Empty document",
                data_types_present=[],
            )

        section_groups: dict[str, list[LDU]] = defaultdict(list)
        for ch in chunks:
            sec = ch.parent_section or "Uncategorized"
            section_groups[sec].append(ch)

        children: list[PageIndexNode] = []
        for sec, sec_chunks in section_groups.items():
            pages = [r.page_number for c in sec_chunks for r in c.page_refs]
            data_types = sorted(set(c.chunk_type for c in sec_chunks))
            content_blob = " ".join(c.content[:320] for c in sec_chunks[:3])
            children.append(
                PageIndexNode(
                    title=sec,
                    page_start=min(pages),
                    page_end=max(pages),
                    child_sections=[],
                    key_entities=self._extract_entities(content_blob),
                    summary=self._summary(content_blob, section_title=sec),
                    data_types_present=data_types,
                )
            )

        all_pages = [r.page_number for c in chunks for r in c.page_refs]
        return PageIndexNode(
            title="Document",
            page_start=min(all_pages),
            page_end=max(all_pages),
            child_sections=children,
            key_entities=[],
            summary="Top-level page index over extracted sections",
            data_types_present=sorted(set(c.chunk_type for c in chunks)),
        )

    def top_sections(self, tree: PageIndexNode, topic: str, k: int = 3) -> list[PageIndexNode]:
        q_tokens = self._tokens(topic)
        scored = []
        for node in tree.child_sections:
            node_blob = " ".join([node.title, node.summary, " ".join(node.key_entities)])
            n_tokens = self._tokens(node_blob)
            score = len(q_tokens.intersection(n_tokens))
            scored.append((score, node))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [n for _, n in scored[:k]]

    def evaluate_retrieval_precision(
        self,
        topic: str,
        chunks: list[LDU],
        search_fn: Callable[[str, int, set[int] | None], list[LDU]],
        page_index: PageIndexNode,
        top_k: int = 3,
    ) -> dict[str, float]:
        if not chunks:
            return {"with_pageindex": 0.0, "without_pageindex": 0.0}

        target_pages = set()
        for node in self.top_sections(page_index, topic, k=3):
            target_pages.update(range(node.page_start, node.page_end + 1))

        with_nav = search_fn(topic, top_k, target_pages if target_pages else None)
        baseline = search_fn(topic, top_k, None)

        q_tokens = self._tokens(topic)
        with_precision = self._precision_at_k(with_nav, q_tokens)
        baseline_precision = self._precision_at_k(baseline, q_tokens)
        return {"with_pageindex": with_precision, "without_pageindex": baseline_precision}

    def _summary(self, content: str, section_title: str) -> str:
        if not content.strip():
            return "No content."
        if not self.llm_summaries_enabled:
            return self._heuristic_summary(content)
        if self.openrouter_enabled:
            llm = self._llm_summary(content, section_title=section_title)
            if llm:
                return llm
        return self._heuristic_summary(content)

    def _heuristic_summary(self, content: str) -> str:
        words = content.split()
        if not words:
            return "No content."
        return " ".join(words[:40])

    def _llm_summary(self, content: str, section_title: str) -> str | None:
        api_key = os.environ.get(self.openrouter_api_key_env, "").strip()
        if not api_key:
            return None
        prompt = (
            f"Summarize document section '{section_title}' in 2-3 concise sentences. "
            "Focus on key facts and entities.\n\n"
            f"Section content:\n{content}"
        )
        payload = {
            "model": self.openrouter_model,
            "temperature": 0.1,
            "max_tokens": self.openrouter_max_output_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        req = urllib.request.Request(
            f"{self.openrouter_api_base}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                body = json.loads(resp.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError:
            return None
        choices = body.get("choices", []) if isinstance(body, dict) else []
        if not choices:
            return None
        msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = msg.get("content", "") if isinstance(msg, dict) else ""
        text = str(content).strip()
        return text or None

    def _extract_entities(self, content: str) -> list[str]:
        out = []
        for token in content.split():
            if token[:1].isupper() and token[1:].islower() and len(token) > 3:
                out.append(token.strip(".,:;()"))
        return list(dict.fromkeys(out))[:10]

    def _tokens(self, text: str) -> set[str]:
        return {t.strip(".,:;()[]{}").lower() for t in text.split() if t.strip(".,:;()[]{}")}

    def _precision_at_k(self, hits: list[LDU], query_tokens: set[str]) -> float:
        if not hits:
            return 0.0
        relevant = 0
        for hit in hits:
            if query_tokens.intersection(self._tokens(hit.content)):
                relevant += 1
        return relevant / len(hits)
