from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections import defaultdict
from typing import Callable

from src.models import LDU, PageIndexNode
from src.utils.llm_client import call_chat_text_openai_compatible, should_use_langchain_wrapper


class PageIndexBuilder:
    def __init__(self, pageindex_cfg: dict | None = None) -> None:
        cfg = pageindex_cfg or {}
        self.llm_summaries_enabled = bool(cfg.get("llm_summaries_enabled", True))
        llm = cfg.get("llm")
        # Backward compatibility: support old `pageindex.openrouter` config shape.
        if not isinstance(llm, dict):
            old = cfg.get("openrouter", {}) or {}
            llm = {
                "provider": "openrouter",
                "enabled": bool(old.get("enabled", False)),
                "api_base": old.get("api_base", "https://openrouter.ai/api/v1"),
                "model": old.get("model", "openai/gpt-4o-mini"),
                "api_key_env": old.get("api_key_env", "OPENROUTER_API_KEY"),
                "max_output_tokens": old.get("max_output_tokens", 140),
                "temperature": 0.1,
            }
        self.llm_provider = str(llm.get("provider", "openrouter")).lower().strip()
        self.llm_enabled = bool(llm.get("enabled", False))
        self.llm_api_base = str(llm.get("api_base", "https://openrouter.ai/api/v1")).rstrip("/")
        self.llm_model = str(llm.get("model", "openai/gpt-4o-mini"))
        self.llm_api_key_env = str(llm.get("api_key_env", "OPENROUTER_API_KEY"))
        self.llm_max_output_tokens = int(llm.get("max_output_tokens", 140))
        self.llm_temperature = float(llm.get("temperature", 0.1))

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
        if not self.llm_enabled:
            return None
        api_key = os.environ.get(self.llm_api_key_env, "").strip()
        if not api_key:
            return None
        prompt = (
            f"Summarize document section '{section_title}' in 2-3 concise sentences. "
            "Focus on key facts and entities.\n\n"
            f"Section content:\n{content}"
        )
        if self.llm_provider == "gemini":
            req = urllib.request.Request(
                f"https://generativelanguage.googleapis.com/v1beta/models/{self.llm_model}:generateContent?key={api_key}",
                data=json.dumps(
                    {
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "temperature": self.llm_temperature,
                            "maxOutputTokens": self.llm_max_output_tokens,
                        },
                    }
                ).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
        else:
            if should_use_langchain_wrapper():
                try:
                    return (
                        call_chat_text_openai_compatible(
                            prompt,
                            model=self.llm_model,
                            api_base=self.llm_api_base,
                            api_key=api_key,
                            max_tokens=self.llm_max_output_tokens,
                            temperature=self.llm_temperature,
                        )
                        or None
                    )
                except Exception:
                    pass
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            req = urllib.request.Request(
                f"{self.llm_api_base}/chat/completions",
                data=json.dumps(
                    {
                        "model": self.llm_model,
                        "temperature": self.llm_temperature,
                        "max_tokens": self.llm_max_output_tokens,
                        "messages": [{"role": "user", "content": prompt}],
                    }
                ).encode("utf-8"),
                headers=headers,
                method="POST",
            )
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                body = json.loads(resp.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError:
            return None
        text = self._extract_text_from_llm_body(body)
        return text or None

    def _extract_text_from_llm_body(self, body: object) -> str:
        if not isinstance(body, dict):
            return ""
        if "choices" in body:
            choices = body.get("choices", [])
            if choices:
                msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
                content = msg.get("content", "") if isinstance(msg, dict) else ""
                return str(content).strip()
        candidates = body.get("candidates", [])
        if candidates:
            cand0 = candidates[0] if isinstance(candidates[0], dict) else {}
            content = cand0.get("content", {}) if isinstance(cand0, dict) else {}
            parts = content.get("parts", []) if isinstance(content, dict) else []
            texts = [str(p.get("text", "")) for p in parts if isinstance(p, dict)]
            return " ".join(t for t in texts if t).strip()
        return ""

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
