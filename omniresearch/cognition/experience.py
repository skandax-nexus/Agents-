"""Long-term learning: the Experience store and the Reflector.

The store keeps lessons (method), facts (knowledge), skills (reusable code) and user
feedback, all searchable offline.  The Reflector turns a finished run into new entries.
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..llm import ChatMessage, LLMBackend, LLMError
from ..tools.base import Tool, ToolResult
from ..tools.textutil import fts_query
from ..agents.protocol import parse_json_object
from .prompts import REFLECT_PROMPT, REFLECT_SYSTEM

KINDS = ("lesson", "fact", "skill", "feedback")


@dataclass
class Entry:
    id: int
    kind: str
    title: str
    content: str
    domains: str
    tags: str
    session: str
    score: float
    uses: int
    created: float

    def render(self) -> str:
        label = {"lesson": "Lesson", "fact": "Fact", "skill": "Skill", "feedback": "User feedback"}[self.kind]
        return f"[{label}] {self.title}: {self.content}"


class Experience:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS entries (
                id INTEGER PRIMARY KEY, kind TEXT, title TEXT, content TEXT, domains TEXT, tags TEXT,
                session TEXT, score REAL, uses INTEGER, created REAL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts USING fts5(
                title, content, domains, tags, kind, content='entries', content_rowid='id',
                tokenize='porter unicode61'
            );
            CREATE TRIGGER IF NOT EXISTS entries_ai AFTER INSERT ON entries BEGIN
                INSERT INTO entries_fts(rowid, title, content, domains, tags, kind)
                VALUES (new.id, new.title, new.content, new.domains, new.tags, new.kind);
            END;
            CREATE TRIGGER IF NOT EXISTS entries_ad AFTER DELETE ON entries BEGIN
                INSERT INTO entries_fts(entries_fts, rowid, title, content, domains, tags, kind)
                VALUES ('delete', old.id, old.title, old.content, old.domains, old.tags, old.kind);
            END;
            """
        )
        self.conn.commit()

    # -- writing -------------------------------------------------------------
    def add(self, kind: str, title: str, content: str, *, domains: str = "", tags: str = "",
            session: str = "", score: float = 1.0) -> int:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        title, content = str(title).strip()[:200], str(content).strip()
        if not title or not content:
            return 0
        dup = self.conn.execute("SELECT id FROM entries WHERE kind=? AND title=? AND content=?",
                                (kind, title, content)).fetchone()
        if dup:
            return int(dup[0])
        cur = self.conn.execute(
            "INSERT INTO entries(kind, title, content, domains, tags, session, score, uses, created) VALUES (?,?,?,?,?,?,?,0,?)",
            (kind, title, content, domains or "", tags or "", session, float(score), time.time()))
        self.conn.commit()
        return int(cur.lastrowid or 0)

    def add_skill(self, name: str, description: str, code: str, *, session: str = "", domains: str = "") -> int:
        content = f"{description.strip()}\n```python\n{code.strip()}\n```"
        return self.add("skill", name.strip(), content, session=session, domains=domains, score=1.5)

    def add_feedback(self, text: str, *, session: str = "") -> int:
        return self.add("feedback", text.strip()[:80], text.strip(), session=session, score=3.0)

    def delete(self, entry_id: int) -> None:
        self.conn.execute("DELETE FROM entries WHERE id=?", (entry_id,))
        self.conn.commit()

    # -- reading -------------------------------------------------------------
    def _rows(self, sql: str, params: tuple) -> list[Entry]:
        return [Entry(*r) for r in self.conn.execute(sql, params).fetchall()]

    def recall(self, query: str, *, kinds: Iterable[str] | None = None, limit: int = 6,
               domains: Iterable[str] = ()) -> list[Entry]:
        """Most relevant entries: FTS relevance, boosted by score, use count and domain match."""
        q = fts_query(query)
        doms = [d for d in domains if d]
        if doms:
            q = f"({q}) OR " + " OR ".join(f'domains:"{d}"' for d in doms)
        rows = self._rows(
            """SELECT e.id, e.kind, e.title, e.content, e.domains, e.tags, e.session, e.score, e.uses, e.created
               FROM entries_fts f JOIN entries e ON e.id = f.rowid WHERE entries_fts MATCH ?
               ORDER BY bm25(entries_fts) / (e.score + 0.2 * e.uses + 1) LIMIT ?""",
            (q, limit * 3))
        if kinds:
            allowed = set(kinds)
            rows = [r for r in rows if r.kind in allowed]
        # user feedback always surfaces first when relevant
        rows.sort(key=lambda r: (r.kind != "feedback",))
        rows = rows[:limit]
        if rows:
            self.conn.execute(f"UPDATE entries SET uses = uses + 1 WHERE id IN ({','.join('?' * len(rows))})",
                              tuple(r.id for r in rows))
            self.conn.commit()
        return rows

    def recall_text(self, query: str, *, domains: Iterable[str] = (), limit: int = 6, kinds=None) -> str:
        rows = self.recall(query, domains=domains, limit=limit, kinds=kinds)
        return "\n".join(f"- {r.render()[:700]}" for r in rows) if rows else "(no relevant prior experience)"

    def recent(self, limit: int = 20, kind: str | None = None) -> list[Entry]:
        if kind:
            return self._rows("SELECT * FROM entries WHERE kind=? ORDER BY created DESC LIMIT ?", (kind, limit))
        return self._rows("SELECT * FROM entries ORDER BY created DESC LIMIT ?", (limit,))

    def stats(self) -> dict[str, int]:
        out = {k: 0 for k in KINDS}
        for kind, n in self.conn.execute("SELECT kind, COUNT(*) FROM entries GROUP BY kind"):
            out[kind] = n
        out["total"] = sum(out.values())
        return out

    def close(self) -> None:
        self.conn.close()


def experience_tools(exp: Experience, session: str) -> list[Tool]:
    def recall(query: str) -> ToolResult:
        rows = exp.recall(str(query), limit=8)
        if not rows:
            return ToolResult("No relevant prior experience.")
        return ToolResult("\n".join(f"- {r.render()[:800]}" for r in rows))

    def skill_search(query: str) -> ToolResult:
        rows = exp.recall(str(query), kinds=("skill",), limit=5)
        if not rows:
            return ToolResult("No saved skill matches. Write the procedure with the python tool and save it with skill_save.")
        return ToolResult("\n\n".join(f"### {r.title}\n{r.content}" for r in rows))

    def skill_save(name: str, description: str, code: str) -> ToolResult:
        eid = exp.add_skill(str(name), str(description), str(code), session=session)
        return ToolResult(f"Saved skill '{name}' (id {eid}). Future runs can find it with skill_search.")

    def lesson_save(title: str, content: str, domains: str = "") -> ToolResult:
        eid = exp.add("lesson", str(title), str(content), domains=str(domains or ""), session=session)
        return ToolResult(f"Saved lesson (id {eid}).")

    return [
        Tool("experience_recall", "Recall lessons, facts, saved skills and user feedback from earlier work that relate to a query.",
             {"query": "string"}, recall, required=("query",)),
        Tool("skill_search", "Find a reusable, previously saved Python procedure (skill) for a kind of computation or analysis.",
             {"query": "string - what the procedure should do"}, skill_search, required=("query",)),
        Tool("skill_save", "Save a working Python procedure as a reusable skill for future tasks.",
             {"name": "string snake_case", "description": "string", "code": "string - python defining the function"},
             skill_save, required=("name", "description", "code")),
        Tool("lesson_save", "Record a lesson about method (what worked / what to avoid) for future tasks.",
             {"title": "string", "content": "string", "domains": "string, optional comma-separated domain ids"},
             lesson_save, required=("title", "content")),
    ]


class Reflector:
    """Looks back on a finished run and writes lessons, facts and skills into Experience."""

    def __init__(self, backend: LLMBackend, experience: Experience) -> None:
        self.backend = backend
        self.experience = experience

    def reflect(self, task: str, trace: str, answer: str, *, session: str = "", domains: str = "") -> dict:
        prompt = REFLECT_PROMPT.format(task=task[:2000], trace=trace[:9000], answer=answer[:5000])
        try:
            reply = self.backend.chat(REFLECT_SYSTEM, [ChatMessage("user", prompt)], temperature=0.1)
        except LLMError as exc:
            return {"error": str(exc), "lessons": [], "facts": [], "skills": []}
        data = parse_json_object(reply) or {}
        saved = {"lessons": 0, "facts": 0, "skills": 0, "self_assessment": str(data.get("self_assessment", ""))}
        for item in data.get("lessons") or []:
            if isinstance(item, dict) and self.experience.add("lesson", item.get("title", ""), item.get("content", ""),
                                                              domains=str(item.get("domains", "") or domains),
                                                              session=session):
                saved["lessons"] += 1
        for item in data.get("facts") or []:
            if isinstance(item, dict) and self.experience.add("fact", item.get("title", ""), item.get("content", ""),
                                                              domains=str(item.get("domains", "") or domains),
                                                              session=session):
                saved["facts"] += 1
        for item in data.get("skills") or []:
            if isinstance(item, dict) and item.get("code") and self.experience.add_skill(
                    str(item.get("name", "skill")), str(item.get("description", "")), str(item["code"]),
                    session=session, domains=domains):
                saved["skills"] += 1
        return saved


def trace_for_reflection(findings: list, max_chars: int = 9000) -> str:
    """Compact a list of AgentResult into a trace the Reflector can read."""
    parts = []
    for r in findings:
        calls = "; ".join(f"{c['tool']}({json.dumps(c['input'])[:80]}){' ERROR' if c.get('error') else ''}"
                          for c in r.tool_calls[:12])
        ver = r.verification.get("problems") if r.verification else None
        parts.append(f"- {r.agent}: {r.steps} steps, {r.attempts} attempt(s), confidence={r.confidence}; "
                     f"tools: {calls or 'none'}" + (f"; verifier problems: {ver}" if ver else "")
                     + f"\n  answer excerpt: {r.content[:300]!r}")
    return "\n".join(parts)[:max_chars]
