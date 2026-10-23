"""Everything the engine hears and says, kept on disk so it can learn from it later."""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import numpy as np

from ..audio.io import read_wav, write_wav


class ExperienceBuffer:
    def __init__(self, root: Path, sr: int) -> None:
        self.root = Path(root)
        self.sr = sr
        (self.root / "heard").mkdir(parents=True, exist_ok=True)
        (self.root / "said").mkdir(parents=True, exist_ok=True)
        self.index = self.root / "index.jsonl"
        self.rows: list[dict] = []
        if self.index.exists():
            for line in self.index.read_text().splitlines():
                if line.strip():
                    self.rows.append(json.loads(line))

    def _append(self, row: dict) -> dict:
        self.rows.append(row)
        with self.index.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        return row

    def add_heard(self, wav: np.ndarray, *, cluster: int | None = None, source: str = "mic") -> dict:
        uid = uuid.uuid4().hex[:10]
        path = self.root / "heard" / f"{uid}.wav"
        write_wav(path, wav, self.sr)
        return self._append({"id": uid, "kind": "heard", "path": str(path), "seconds": len(wav) / self.sr,
                             "cluster": cluster, "source": source, "time": time.time()})

    def add_exchange(self, heard: np.ndarray, said: np.ndarray, *, score: float | None = None,
                     cluster: int | None = None, origin: str = "live") -> dict:
        uid = uuid.uuid4().hex[:10]
        hp, sp = self.root / "heard" / f"{uid}.wav", self.root / "said" / f"{uid}.wav"
        write_wav(hp, heard, self.sr)
        write_wav(sp, said, self.sr)
        return self._append({"id": uid, "kind": "exchange", "heard": str(hp), "said": str(sp),
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
            self.index.write_text("".join(json.dumps(r) + "\n" for r in self.rows), encoding="utf-8")
        return hit

    def heard_seconds(self) -> float:
        return sum(r["seconds"] for r in self.rows if r["kind"] == "heard") + \
            sum(r["seconds"] for r in self.rows if r["kind"] == "exchange")

    def heard_clips(self, limit: int | None = None) -> list[np.ndarray]:
        paths = [r["path"] for r in self.rows if r["kind"] == "heard"] + \
                [r["heard"] for r in self.rows if r["kind"] == "exchange"]
        if limit:
            paths = paths[-limit:]
        out = []
        for p in paths:
            try:
                out.append(read_wav(p)[0])
            except (OSError, ValueError):
                continue
        return out

    def exchanges(self, *, min_score: float | None = None, exclude_negative: bool = True) -> list[dict]:
        rows = [r for r in self.rows if r["kind"] == "exchange"]
        if exclude_negative:
            rows = [r for r in rows if (r.get("feedback") or 0) >= 0]
        if min_score is not None:
            rows = [r for r in rows if r.get("score") is None or r["score"] >= min_score or (r.get("feedback") or 0) > 0]
        return rows

    def load_exchange(self, row: dict) -> tuple[np.ndarray, np.ndarray]:
        return read_wav(row["heard"])[0], read_wav(row["said"])[0]

    def stats(self) -> dict:
        ex = [r for r in self.rows if r["kind"] == "exchange"]
        return {"heard_utterances": sum(1 for r in self.rows if r["kind"] == "heard"), "exchanges": len(ex),
                "positive_feedback": sum(1 for r in ex if (r.get("feedback") or 0) > 0),
                "negative_feedback": sum(1 for r in ex if (r.get("feedback") or 0) < 0),
                "seconds_heard": round(self.heard_seconds(), 1)}
