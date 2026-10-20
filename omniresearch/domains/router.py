"""Keyword-based domain routing.

This is the deterministic fallback used when the planner model returns an
unusable plan, and as a sanity check on the planner's choices.  It works fully
offline and needs no model.
"""
from __future__ import annotations

import re

from .registry import DOMAINS, Domain, get_domain

_WORD = re.compile(r"[a-z0-9][a-z0-9\-\.]*")


def _score(text: str, domain: Domain) -> float:
    score = 0.0
    for kw in domain.keywords:
        if kw in text:
            # longer, more specific keywords weigh more
            score += 1.0 + 0.15 * len(kw.split())
    return score


def route_domains(question: str, max_domains: int = 4, min_score: float = 1.0) -> list[Domain]:
    """Return the most relevant domains for a question, always at least one."""
    text = " " + question.lower() + " "
    scored = [(_score(text, d), d) for d in DOMAINS if d.id != "general"]
    scored.sort(key=lambda t: t[0], reverse=True)
    chosen = [d for s, d in scored[:max_domains] if s >= min_score]
    if not chosen:
        return [get_domain("general")]
    return chosen


def validate_domain_ids(ids: list[str], question: str, max_domains: int) -> list[Domain]:
    """Keep known ids from the planner; fill any gap with keyword routing."""
    known = {d.id for d in DOMAINS}
    picked: list[Domain] = []
    for raw in ids:
        did = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
        if did in known and get_domain(did) not in picked:
            picked.append(get_domain(did))
    if not picked:
        picked = route_domains(question, max_domains=max_domains)
    return picked[:max_domains]
