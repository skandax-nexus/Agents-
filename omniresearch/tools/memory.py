"""Persistent research memory (SQLite + FTS5).

Agents save findings here during a run, and later runs can recall them, so the
system accumulates knowledge over time - entirely offline.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from .base import Tool, ToolResult
from .textutil import fts_query


class Memory:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.executescript(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS notes USING fts5(
                topic, content, tags, session UNINDEXED, created UNINDEXED, tokenize='porter unicode61'
            );
            """
        )
        self.conn.commit()

    def save(self, topic: str, content: str, tags: str = "", session: str = "") -> None:
        self.conn.execute("INSERT INTO notes(topic, content, tags, session, created) VALUES (?,?,?,?,?)",
                          (topic, content, tags, session, time.time()))
        self.conn.commit()

    def search(self, query: str, limit: int = 8) -> list[dict]:
        rows = self.conn.execute(
            "SELECT topic, content, tags, session, created, bm25(notes) FROM notes WHERE notes MATCH ? "
            "ORDER BY bm25(notes) LIMIT ?", (fts_query(query), limit)).fetchall()
        return [{"topic": r[0], "content": r[1], "tags": r[2], "session": r[3], "created": r[4]} for r in rows]

    def recent(self, limit: int = 20) -> list[dict]:
        rows = self.conn.execute(
            "SELECT topic, content, tags, session, created FROM notes ORDER BY created DESC LIMIT ?", (limit,)).fetchall()
        return [{"topic": r[0], "content": r[1], "tags": r[2], "session": r[3], "created": r[4]} for r in rows]

    def count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0])

    def close(self) -> None:
        self.conn.close()


def memory_tools(memory: Memory, session: str) -> list[Tool]:
    def save(topic: str, content: str, tags: str = "") -> ToolResult:
        memory.save(str(topic), str(content), str(tags or ""), session)
        return ToolResult(f"Saved note on '{topic}'.")

    def recall(query: str) -> ToolResult:
        hits = memory.search(str(query))
        if not hits:
            return ToolResult("No prior notes match.")
        lines = [f"- [{time.strftime('%Y-%m-%d', time.localtime(h['created']))}] {h['topic']}: {h['content'][:600]}"
                 for h in hits]
        return ToolResult("\n".join(lines))

    return [
        Tool("memory_save", "Persist an important finding, fact or source so future research runs can reuse it.",
             {"topic": "string", "content": "string", "tags": "string, optional comma-separated"}, save,
             required=("topic", "content")),
        Tool("memory_recall", "Search notes saved by earlier research runs.",
             {"query": "string"}, recall, required=("query",)),
    ]
