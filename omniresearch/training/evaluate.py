"""Measure a model on held-out examples so training is judged, not assumed.

Scores are deterministic and need no judge model:
- numeric match: fraction of the reference's numbers that appear in the prediction (2% tolerance)
- token F1: overlap of content words
- json validity: for reference answers that are JSON objects
The combined score weights numeric accuracy highest, because that is what verification cares about.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ..llm import ChatMessage, LLMBackend, LLMError

_NUM = re.compile(r"(?<![\w.])[-+]?\d+(?:[.,]\d+)?(?:e[-+]?\d+)?(?![\w.])", re.I)
_WORD = re.compile(r"[a-z][a-z0-9\-]{2,}")
_STOP = {"the", "and", "for", "that", "with", "this", "from", "are", "was", "were", "has", "have", "which", "into",
         "than", "then", "its", "also", "can", "but", "not", "more", "most", "about", "over", "under", "per"}


def numbers(text: str) -> list[float]:
    out = []
    for m in _NUM.findall(text):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            continue
    return out


def numeric_match(pred: str, ref: str, tol: float = 0.02) -> float | None:
    ref_nums = numbers(ref)
    if not ref_nums:
        return None
    pred_nums = numbers(pred)
    hit = 0
    for r in ref_nums:
        if any(abs(p - r) <= tol * max(abs(r), 1e-9) or (r == 0 and p == 0) for p in pred_nums):
            hit += 1
    return hit / len(ref_nums)


def token_f1(pred: str, ref: str) -> float | None:
    """Content-word overlap; None when the reference has no content words to compare."""
    p = {w for w in _WORD.findall(pred.lower()) if w not in _STOP}
    r = {w for w in _WORD.findall(ref.lower()) if w not in _STOP}
    if not r:
        return None
    if not p:
        return 0.0
    overlap = len(p & r)
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(r)
    return 2 * precision * recall / (precision + recall)


def json_valid(pred: str, ref: str) -> float | None:
    try:
        json.loads(ref)
    except (ValueError, TypeError):
        return None
    try:
        json.loads(pred)
        return 1.0
    except (ValueError, TypeError):
        m = re.search(r"\{.*\}", pred, re.S)
        if m:
            try:
                json.loads(m.group(0))
                return 1.0
            except ValueError:
                pass
        return 0.0


def score(pred: str, ref: str) -> dict:
    nm = numeric_match(pred, ref)
    f1 = token_f1(pred, ref)
    jv = json_valid(pred, ref)
    parts, weights = [], []
    if f1 is not None:
        parts.append(f1)
        weights.append(1.0)
    if nm is not None:
        parts.append(nm)
        weights.append(2.0)
    if jv is not None:
        parts.append(jv)
        weights.append(1.0)
    combined = sum(p * w for p, w in zip(parts, weights)) / sum(weights) if weights else 0.0
    return {"numeric": nm, "token_f1": round(f1, 4) if f1 is not None else None, "json": jv,
            "combined": round(combined, 4)}


@dataclass
class EvalReport:
    backend: str
    n: int = 0
    combined: float = 0.0
    token_f1: float = 0.0
    numeric: float | None = None
    json_validity: float | None = None
    errors: int = 0
    per_example: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"backend": self.backend, "n": self.n, "combined": self.combined, "token_f1": self.token_f1,
                "numeric": self.numeric, "json_validity": self.json_validity, "errors": self.errors}


def evaluate_backend(backend: LLMBackend, rows: list[dict], *, limit: int | None = None, max_tokens: int = 1024,
                     on_event=None) -> EvalReport:
    rows = rows[:limit] if limit else rows
    report = EvalReport(backend=backend.describe())
    num_scores, json_scores = [], []
    for i, row in enumerate(rows):
        msgs = row["messages"]
        system = next((m["content"] for m in msgs if m["role"] == "system"), "")
        user = next((m["content"] for m in msgs if m["role"] == "user"), "")
        ref = msgs[-1]["content"]
        try:
            pred = backend.chat(system, [ChatMessage("user", user)], temperature=0.0, max_tokens=max_tokens)
        except LLMError as exc:
            report.errors += 1
            report.per_example.append({"i": i, "error": str(exc)})
            continue
        s = score(pred, ref)
        report.per_example.append({"i": i, **s})
        report.n += 1
        report.combined += s["combined"]
        report.token_f1 += s["token_f1"] or 0.0
        if s["numeric"] is not None:
            num_scores.append(s["numeric"])
        if s["json"] is not None:
            json_scores.append(s["json"])
        if on_event:
            on_event("eval_example", {"i": i, "of": len(rows), **s})
    if report.n:
        report.combined = round(report.combined / report.n, 4)
        report.token_f1 = round(report.token_f1 / report.n, 4)
    report.numeric = round(sum(num_scores) / len(num_scores), 4) if num_scores else None
    report.json_validity = round(sum(json_scores) / len(json_scores), 4) if json_scores else None
    return report
