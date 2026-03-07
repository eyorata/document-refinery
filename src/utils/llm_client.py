from __future__ import annotations

import os
from typing import Any


def call_chat_text_openai_compatible(
    prompt: str,
    *,
    model: str,
    api_base: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """
    Optional LangChain ChatOpenAI wrapper for OpenAI-compatible endpoints
    (OpenRouter, LM Studio, local gateways).
    Falls back by raising RuntimeError if langchain-openai is unavailable.
    """
    try:
        from langchain_openai import ChatOpenAI  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - dependency optional
        raise RuntimeError("langchain-openai is not installed") from exc

    client = ChatOpenAI(
        model=model,
        base_url=api_base.rstrip("/"),
        api_key=api_key or os.getenv("OPENAI_API_KEY", "not-required"),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    msg = client.invoke(prompt)
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        out: list[str] = []
        for part in content:
            if isinstance(part, dict):
                txt = str(part.get("text", "")).strip()
                if txt:
                    out.append(txt)
        return " ".join(out).strip()
    return str(content).strip()


def should_use_langchain_wrapper() -> bool:
    return os.getenv("USE_LANGCHAIN_OPENAI_WRAPPER", "false").strip().lower() in {"1", "true", "yes"}
