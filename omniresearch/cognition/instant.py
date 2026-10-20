"""Instant learning: tell the agent something once and it knows it from then on.

No training step.  Everything is stored immediately in searchable local stores:

- ``learn_text`` / ``learn_file`` / ``learn_url``: any text, document or page goes into the
  offline knowledge base (chunked, full-text indexed) and, when short, into the experience
  store as a fact.  The next question that touches it finds it.
- ``remember`` / ``recall``: every verified answer the system produces is kept as a
  question -> answer pair.  A sufficiently similar later question is answered instantly
  from storage instead of being worked out again.
"""
from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path

from ..tools.base import Tool, ToolResult
from ..tools.knowledge import KnowledgeBase
from ..tools.textutil import fts_query
from .experience import Experience

_WORD = re.compile(r"[a-z0-9][a-z0-9\-]{1,}")
_STOP = {"the", "and", "for", "that", "with", "this", "from", "are", "was", "were", "has", "have", "which", "into",
         "than", "then", "its", "also", "can", "but", "not", "more", "most", "about", "what", "how", "why", "does",
         "do", "is", "of", "to", "in", "on", "a", "an", "it", "be", "by", "or", "as", "at", "we", "you", "i", "my"}


_PLACEHOLDER = re.compile(r"^\s*(\[mock backend\]|\(model error:|No scripted response)", re.I)


def looks_placeholder(text: str) -> bool:
    """True for mock/error output that must never be learned as an answer."""
    return bool(_PLACEHOLDER.search(text or ""))


def _words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


def similarity(a: str, b: str) -> float:
    """Jaccard similarity of content words, used to decide whether a stored answer applies."""
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


class InstantLearner:
    def __init__(self, kb: KnowledgeBase, experience: Experience, db_path: Path) -> None:
        self.kb = kb
        self.experience = experience
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS answers (
                id INTEGER PRIMARY KEY, question TEXT, answer TEXT, sources TEXT, confidence REAL,
                session TEXT, created REAL, uses INTEGER DEFAULT 0
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS answers_fts USING fts5(
                question, answer, content='answers', content_rowid='id', tokenize='porter unicode61'
            );
            CREATE TRIGGER IF NOT EXISTS answers_ai AFTER INSERT ON answers BEGIN
                INSERT INTO answers_fts(rowid, question, answer) VALUES (new.id, new.question, new.answer);
            END;
            CREATE TRIGGER IF NOT EXISTS answers_ad AFTER DELETE ON answers BEGIN
                INSERT INTO answers_fts(answers_fts, rowid, question, answer)
                VALUES ('delete', old.id, old.question, old.answer);
            END;
            """
        )
        self.conn.commit()

    # -- teach it anything ------------------------------------------------------
    def learn_text(self, text: str, *, title: str | None = None, source: str = "user", session: str = "") -> dict:
        text = str(text).strip()
        if not text:
            return {"stored": False, "reason": "empty"}
        title = (title or text.splitlines()[0][:80]).strip()
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60] or "note"
        key = f"learned://{source}/{slug}-{int(time.time())}"
        doc_id = self.kb.add_text(title, text, path=key)
        fact_id = 0
        if len(text) <= 1200:
            fact_id = self.experience.add("fact", title, text, session=session, score=2.0, tags="learned")
        return {"stored": True, "title": title, "knowledge_doc": doc_id, "fact": fact_id, "chars": len(text)}

    def learn_file(self, path: str | Path) -> dict:
        p = Path(path).expanduser()
        if not p.exists():
            return {"stored": False, "reason": f"{p} does not exist"}
        indexed, skipped = self.kb.index_path(p)
        return {"stored": indexed > 0, "files_indexed": indexed, "files_skipped": skipped, "path": str(p)}

    def learn_url(self, url: str, fetch_tool: Tool | None) -> dict:
        if fetch_tool is None:
            return {"stored": False, "reason": "web_fetch is not available (offline)"}
        res = fetch_tool.run(url=url, max_chars=40000)
        if res.error:
            return {"stored": False, "reason": res.output[:200]}
        # web_fetch already caches the page into the knowledge base under its URL
        return {"stored": True, "url": url, "chars": len(res.output)}

    # -- remember what it answered --------------------------------------------------
    def remember(self, question: str, answer: str, *, confidence: float | None = None, sources: list[str] | None = None,
                 session: str = "") -> int:
        question, answer = question.strip(), answer.strip()
        if not question or not answer or looks_placeholder(answer):
            return 0
        row = self.conn.execute("SELECT id FROM answers WHERE question=?", (question,)).fetchone()
        if row:
            self.conn.execute("UPDATE answers SET answer=?, sources=?, confidence=?, session=?, created=? WHERE id=?",
                              (answer, "\n".join(sources or []), confidence, session, time.time(), row[0]))
            self.conn.execute("INSERT INTO answers_fts(answers_fts, rowid, question, answer) VALUES ('delete', ?, ?, ?)",
                              (row[0], question, answer))
            self.conn.execute("INSERT INTO answers_fts(rowid, question, answer) VALUES (?, ?, ?)", (row[0], question, answer))
            self.conn.commit()
            return int(row[0])
        cur = self.conn.execute(
            "INSERT INTO answers(question, answer, sources, confidence, session, created, uses) VALUES (?,?,?,?,?,?,0)",
            (question, answer, "\n".join(sources or []), confidence, session, time.time()))
        self.conn.commit()
        return int(cur.lastrowid or 0)

    def recall(self, question: str, *, min_similarity: float = 0.6) -> dict | None:
        rows = self.conn.execute(
            """SELECT a.id, a.question, a.answer, a.sources, a.confidence, a.created FROM answers_fts f
               JOIN answers a ON a.id = f.rowid WHERE answers_fts MATCH ? ORDER BY bm25(answers_fts) LIMIT 8""",
            (fts_query(question),)).fetchall()
        best, best_sim = None, 0.0
        for r in rows:
            sim = similarity(question, r[1])
            if sim > best_sim:
                best, best_sim = r, sim
        if best is None or best_sim < min_similarity:
            return None
        self.conn.execute("UPDATE answers SET uses = uses + 1 WHERE id=?", (best[0],))
        self.conn.commit()
        return {"id": best[0], "question": best[1], "answer": best[2], "sources": [s for s in best[3].split("\n") if s],
                "confidence": best[4], "created": best[5], "similarity": round(best_sim, 3)}

    def forget(self, answer_id: int) -> None:
        self.conn.execute("DELETE FROM answers WHERE id=?", (answer_id,))
        self.conn.commit()

    def recent(self, limit: int = 20) -> list[dict]:
        rows = self.conn.execute("SELECT id, question, answer, confidence, created, uses FROM answers "
                                 "ORDER BY created DESC LIMIT ?", (limit,)).fetchall()
        return [{"id": r[0], "question": r[1], "answer": r[2], "confidence": r[3], "created": r[4], "uses": r[5]}
                for r in rows]

    def stats(self) -> dict:
        n = self.conn.execute("SELECT COUNT(*) FROM answers").fetchone()[0]
        uses = self.conn.execute("SELECT COALESCE(SUM(uses),0) FROM answers").fetchone()[0]
        return {"answers": n, "instant_hits": uses, **self.kb.stats(), "experience": self.experience.stats()}


def instant_tools(learner: InstantLearner, session: str) -> list[Tool]:
    def learn(text: str, title: str = "") -> ToolResult:
        info = learner.learn_text(str(text), title=str(title) or None, source="agent", session=session)
        if not info.get("stored"):
            return ToolResult(f"Nothing stored: {info.get('reason')}", error=True)
        return ToolResult(f"Learned '{info['title']}' ({info['chars']} chars). It is now searchable with knowledge_search.")

    def recall_answer(question: str) -> ToolResult:
        hit = learner.recall(str(question))
        if not hit:
            return ToolResult("No previously learned answer matches closely enough.")
        when = time.strftime("%Y-%m-%d", time.localtime(hit["created"]))
        return ToolResult(f"Learned answer from {when} (similarity {hit['similarity']}, confidence {hit['confidence']}):\n"
                          f"Q: {hit['question']}\nA: {hit['answer'][:4000]}", sources=hit["sources"])

    return [
        Tool("learn", "Store any useful text you found or worked out (a fact, a table, a passage, a definition) so it is "
                      "known instantly in all future tasks. Use it whenever you discover something worth keeping.",
             {"text": "string - the content to learn", "title": "string, optional"}, learn, required=("text",)),
        Tool("recall_answer", "Check whether a closely similar question was already answered and learned earlier.",
             {"question": "string"}, recall_answer, required=("question",)),
    ]
