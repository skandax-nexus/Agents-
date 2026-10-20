"""Local knowledge base: a SQLite FTS5 index over the user's own documents.

This is the heart of offline research.  Index textbooks, papers, notes, data
exports or an entire documentation tree once, then every sub-agent can search
them with no network at all.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from .base import Tool, ToolResult
from .textutil import chunk_text, extract_text, fts_query

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", ".omniresearch", "research_output"}


class KnowledgeBase:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._init()

    def _init(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                path TEXT UNIQUE,
                title TEXT,
                mtime REAL,
                n_chunks INTEGER,
                indexed_at REAL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
                doc_id UNINDEXED, chunk_no UNINDEXED, title, content, tokenize='porter unicode61'
            );
            """
        )
        self.conn.commit()

    # -- indexing -----------------------------------------------------------
    def index_path(self, root: Path, *, verbose: bool = False) -> tuple[int, int]:
        """Index a file or directory. Returns (files_indexed, files_skipped)."""
        root = Path(root).expanduser().resolve()
        files = [root] if root.is_file() else sorted(
            p for p in root.rglob("*") if p.is_file() and not any(part in SKIP_DIRS for part in p.parts)
        )
        indexed = skipped = 0
        for path in files:
            try:
                if self._index_file(path):
                    indexed += 1
                    if verbose:
                        print(f"  indexed {path}")
                else:
                    skipped += 1
            except Exception as exc:  # noqa: BLE001
                skipped += 1
                if verbose:
                    print(f"  skipped {path}: {exc}")
        self.conn.commit()
        return indexed, skipped

    def _index_file(self, path: Path) -> bool:
        mtime = path.stat().st_mtime
        row = self.conn.execute("SELECT id, mtime FROM documents WHERE path=?", (str(path),)).fetchone()
        if row and abs(row[1] - mtime) < 1e-6:
            return False  # unchanged
        text = extract_text(path)
        if not text.strip():
            return False
        if row:
            self.conn.execute("DELETE FROM chunks WHERE doc_id=?", (row[0],))
            self.conn.execute("DELETE FROM documents WHERE id=?", (row[0],))
        title = path.stem.replace("_", " ").replace("-", " ")
        chunks = chunk_text(text)
        cur = self.conn.execute(
            "INSERT INTO documents(path, title, mtime, n_chunks, indexed_at) VALUES (?,?,?,?,?)",
            (str(path), title, mtime, len(chunks), time.time()),
        )
        doc_id = cur.lastrowid
        self.conn.executemany(
            "INSERT INTO chunks(doc_id, chunk_no, title, content) VALUES (?,?,?,?)",
            [(doc_id, i, title, c) for i, c in enumerate(chunks)],
        )
        return True

    def add_text(self, title: str, text: str, path: str | None = None) -> int:
        """Index an in-memory document (e.g. a fetched web page) so it is searchable later."""
        key = path or f"memory://{title}"
        row = self.conn.execute("SELECT id FROM documents WHERE path=?", (key,)).fetchone()
        if row:
            self.conn.execute("DELETE FROM chunks WHERE doc_id=?", (row[0],))
            self.conn.execute("DELETE FROM documents WHERE id=?", (row[0],))
        chunks = chunk_text(text)
        cur = self.conn.execute(
            "INSERT INTO documents(path, title, mtime, n_chunks, indexed_at) VALUES (?,?,?,?,?)",
            (key, title, time.time(), len(chunks), time.time()),
        )
        doc_id = cur.lastrowid
        self.conn.executemany(
            "INSERT INTO chunks(doc_id, chunk_no, title, content) VALUES (?,?,?,?)",
            [(doc_id, i, c and title, c) for i, c in enumerate(chunks)],
        )
        self.conn.commit()
        return int(doc_id or 0)

    # -- querying -----------------------------------------------------------
    def search(self, query: str, limit: int = 6) -> list[dict]:
        q = fts_query(query)
        rows = self.conn.execute(
            """
            SELECT d.path, d.title, c.chunk_no, snippet(chunks, 3, '[', ']', ' ... ', 48) AS snip,
                   c.content, bm25(chunks) AS score
            FROM chunks c JOIN documents d ON d.id = c.doc_id
            WHERE chunks MATCH ? ORDER BY score LIMIT ?
            """,
            (q, limit),
        ).fetchall()
        return [
            {"path": r[0], "title": r[1], "chunk": r[2], "snippet": r[3], "content": r[4], "score": r[5]}
            for r in rows
        ]

    def read(self, path: str, chunk: int | None = None, max_chars: int = 6000) -> str:
        row = self.conn.execute("SELECT id, title FROM documents WHERE path=? OR title=?", (path, path)).fetchone()
        if not row:
            return ""
        if chunk is None:
            rows = self.conn.execute(
                "SELECT content FROM chunks WHERE doc_id=? ORDER BY chunk_no", (row[0],)).fetchall()
            return "\n".join(r[0] for r in rows)[:max_chars]
        rows = self.conn.execute(
            "SELECT content FROM chunks WHERE doc_id=? AND chunk_no BETWEEN ? AND ? ORDER BY chunk_no",
            (row[0], chunk, chunk + 1)).fetchall()
        return "\n".join(r[0] for r in rows)[:max_chars]

    def stats(self) -> dict:
        docs = self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunks = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return {"documents": docs, "chunks": chunks, "path": str(self.db_path)}

    def close(self) -> None:
        self.conn.close()


def knowledge_tools(kb: KnowledgeBase) -> list[Tool]:
    def search(query: str, limit: int = 6) -> ToolResult:
        try:
            limit = max(1, min(int(limit), 15))
        except (TypeError, ValueError):
            limit = 6
        hits = kb.search(str(query), limit=limit)
        if not hits:
            return ToolResult("No matches in the local knowledge base. (Index documents with `omniresearch index <dir>`.)")
        lines = []
        sources = []
        for h in hits:
            ref = f"{h['path']}#chunk{h['chunk']}"
            sources.append(ref)
            lines.append(f"[{ref}] ({h['title']})\n{h['content'][:900]}")
        return ToolResult("\n\n".join(lines), sources=sources)

    def read(path: str, chunk: int | None = None) -> ToolResult:
        try:
            chunk_i = int(chunk) if chunk not in (None, "") else None
        except (TypeError, ValueError):
            chunk_i = None
        text = kb.read(str(path), chunk_i)
        if not text:
            return ToolResult(f"No indexed document named '{path}'. Use knowledge_search to find exact paths.", error=True)
        return ToolResult(text, sources=[str(path)])

    return [
        Tool("knowledge_search",
             "Full-text search over the local, offline knowledge base of indexed documents (papers, books, notes, data).",
             {"query": "string - keywords or a phrase", "limit": "int, optional (default 6)"},
             search, required=("query",)),
        Tool("knowledge_read",
             "Read a larger span of an indexed document by its path (from knowledge_search results), optionally from a chunk number.",
             {"path": "string - document path as returned by knowledge_search", "chunk": "int, optional"},
             read, required=("path",)),
    ]
