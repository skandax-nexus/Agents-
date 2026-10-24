"""Everything the engine hears and says, kept on disk so it can learn from it later."""
from __future__ import annotations

import json
import logging
import struct
import time
import uuid
import wave
from pathlib import Path

import numpy as np

from ..audio.io import read_wav, write_wav
from ..fsutil import atomic_write_text

log = logging.getLogger("voicecore.buffer")
UNREADABLE = (OSError, ValueError, EOFError, wave.Error, struct.error)


class ExperienceBuffer:
    """Paths in the index are stored relative to the buffer root, so the engine directory can move."""

    def __init__(self, root: Path, sr: int) -> None:
        self.root = Path(root).expanduser().resolve()
        self.sr = sr
        (self.root / "heard").mkdir(parents=True, exist_ok=True)
        (self.root / "said").mkdir(parents=True, exist_ok=True)
        self.index = self.root / "index.jsonl"
        self.rows: list[dict] = []
        if self.index.exists():
            for line in self.index.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    self.rows.append(json.loads(line))
                except ValueError:
                    continue   # a half-written line from a crash; skip it

    def _append(self, row: dict) -> dict:
        self.rows.append(row)
        with self.index.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        return row

    def _abs(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else self.root / p     # absolute entries from older indexes still work

    def add_heard(self, wav: np.ndarray, *, cluster: int | None = None, source: str = "mic") -> dict:
        uid = uuid.uuid4().hex[:10]
        rel = f"heard/{uid}.wav"
        write_wav(self.root / rel, wav, self.sr)
        return self._append({"id": uid, "kind": "heard", "path": rel, "seconds": len(wav) / self.sr,
                             "cluster": cluster, "source": source, "time": time.time()})

    def add_exchange(self, heard: np.ndarray, said: np.ndarray, *, score: float | None = None,
                     cluster: int | None = None, origin: str = "live") -> dict:
        uid = uuid.uuid4().hex[:10]
        hrel, srel = f"heard/{uid}.wav", f"said/{uid}.wav"
        write_wav(self.root / hrel, heard, self.sr)
        write_wav(self.root / srel, said, self.sr)
        return self._append({"id": uid, "kind": "exchange", "heard": hrel, "said": srel,
                             "seconds": (len(heard) + len(said)) / self.sr, "score": score, "feedback": None,
                             "cluster": cluster, "origin": origin, "time": time.time()})

    def feedback(self, exchange_id: str, value: int) -> bool:
        """+1 good reply, -1 bad reply (from the user). Rewrites the index."""
        hit = False
        for r in self.rows:
            if r.get("id") == exchange_id and r.get("kind") == "exchange":
                r["feedback"] = int(value)
                hit = True
        if hit:
            atomic_write_text(self.index, "".join(json.dumps(r) + "\n" for r in self.rows))
        return hit

    def _heard_rows(self) -> list[dict]:
        """Audio that was genuinely heard: 'heard' rows and the heard side of live exchanges.  Self-play
        exchanges re-use heard clips, so they are excluded to keep the corpus from filling with copies."""
        rows = [r for r in self.rows if r["kind"] == "heard" or (r["kind"] == "exchange" and r.get("origin", "live") == "live")]
        return sorted(rows, key=lambda r: r.get("time", 0.0))

    def heard_seconds(self) -> float:
        total = 0.0
        for r in self._heard_rows():
            if r["kind"] == "heard":
                total += r["seconds"]
            else:
                try:
                    total += len(read_wav(self._abs(r["heard"]))[0]) / self.sr
                except UNREADABLE:
                    continue
        return total

    def heard_clips(self, limit: int | None = None) -> list[np.ndarray]:
        rows = self._heard_rows()
        if limit:
            rows = rows[-limit:]          # most recent by time
        out = []
        for r in rows:
            path = self._abs(r["path"] if r["kind"] == "heard" else r["heard"])
            try:
                out.append(read_wav(path)[0])
            except UNREADABLE as exc:
                log.warning("skipping unreadable clip %s: %s", path, exc)
                continue
        return out

    def exchanges(self, *, min_score: float | None = None, exclude_negative: bool = True) -> list[dict]:
        rows = [r for r in self.rows if r["kind"] == "exchange"]
        if exclude_negative:
            rows = [r for r in rows if (r.get("feedback") or 0) >= 0]
        if min_score is not None:
            rows = [r for r in rows if r.get("score") is None or r["score"] >= min_score or (r.get("feedback") or 0) > 0]
        return rows

    def load_exchange(self, row: dict) -> tuple[np.ndarray, np.ndarray] | None:
        """Both clips of an exchange, or None if either file is unreadable (logged, never fatal)."""
        try:
            return read_wav(self._abs(row["heard"]))[0], read_wav(self._abs(row["said"]))[0]
        except UNREADABLE as exc:
            log.warning("skipping unreadable exchange %s: %s", row.get("id"), exc)
            return None

    def stats(self) -> dict:
        ex = [r for r in self.rows if r["kind"] == "exchange"]
        return {"heard_utterances": sum(1 for r in self.rows if r["kind"] == "heard"), "exchanges": len(ex),
                "positive_feedback": sum(1 for r in ex if (r.get("feedback") or 0) > 0),
                "negative_feedback": sum(1 for r in ex if (r.get("feedback") or 0) < 0),
                "seconds_heard": round(self.heard_seconds(), 1)}
