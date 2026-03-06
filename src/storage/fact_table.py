from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from src.models import LDU


class FactTableStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                page INTEGER NOT NULL,
                content_hash TEXT NOT NULL
            )
            """
        )
        con.commit()
        con.close()

    def ingest(self, chunks: list[LDU]) -> int:
        pattern = re.compile(r"([A-Za-z][A-Za-z ]{2,30})\s*[:=]\s*([$]?[0-9][0-9,\.A-Za-z]*)")
        rows = []
        for c in chunks:
            for m in pattern.finditer(c.content):
                rows.append((m.group(1).strip().lower(), m.group(2).strip(), c.page_refs[0].page_number, c.content_hash))

        if not rows:
            return 0

        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        cur.executemany("INSERT INTO facts(key, value, page, content_hash) VALUES (?,?,?,?)", rows)
        con.commit()
        con.close()
        return len(rows)

    def query(self, sql: str) -> list[tuple]:
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        cur.execute(sql)
        res = cur.fetchall()
        con.close()
        return res

    def clear(self) -> None:
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        cur.execute("DELETE FROM facts")
        con.commit()
        con.close()
