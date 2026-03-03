from __future__ import annotations

import hashlib


def stable_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
