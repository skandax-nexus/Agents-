"""Text extraction helpers shared by the knowledge index and web tools."""
from __future__ import annotations

import csv
import html
import io
import json
import re
from pathlib import Path

_TAG = re.compile(r"<[^>]+>")
_SCRIPT = re.compile(r"<(script|style|noscript)[^>]*>.*?</\1>", re.S | re.I)
_WS = re.compile(r"[ \t\r\f\v]+")
_NL = re.compile(r"\n{3,}")

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".tex", ".org", ".log", ".py", ".r", ".jl", ".m",
                 ".js", ".ts", ".java", ".c", ".cpp", ".h", ".go", ".rs", ".sh", ".yaml", ".yml", ".toml",
                 ".ini", ".cfg", ".bib", ".tsv", ".xml", ".svg"}


def html_to_text(raw: str) -> str:
    raw = _SCRIPT.sub(" ", raw)
    raw = re.sub(r"<br\s*/?>|</p>|</div>|</h\d>|</li>|</tr>", "\n", raw, flags=re.I)
    text = _TAG.sub(" ", raw)
    text = html.unescape(text)
    return normalise(text)


def normalise(text: str) -> str:
    text = _WS.sub(" ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return _NL.sub("\n\n", text).strip()


def extract_text(path: Path) -> str:
    """Best-effort text extraction for common document types."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader  # optional dependency
        except ImportError:
            return ""
        reader = PdfReader(str(path))
        return normalise("\n\n".join((page.extract_text() or "") for page in reader.pages))
    raw = path.read_text(encoding="utf-8", errors="replace")
    if suffix in {".html", ".htm", ".xhtml"}:
        return html_to_text(raw)
    if suffix == ".json":
        try:
            return normalise(json.dumps(json.loads(raw), indent=1, ensure_ascii=False))
        except json.JSONDecodeError:
            return normalise(raw)
    if suffix == ".csv":
        rows = list(csv.reader(io.StringIO(raw)))
        return normalise("\n".join(" | ".join(r) for r in rows[:5000]))
    if suffix in TEXT_SUFFIXES or suffix == "":
        return normalise(raw)
    return ""


def chunk_text(text: str, size: int = 1500, overlap: int = 200) -> list[str]:
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + size)
        # try to end on a paragraph or sentence boundary
        if end < n:
            cut = max(text.rfind("\n\n", start, end), text.rfind(". ", start, end))
            if cut > start + size // 2:
                end = cut + 1
        chunks.append(text[start:end].strip())
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


def fts_query(query: str) -> str:
    """Turn free text into a safe FTS5 query: quoted tokens joined by OR."""
    tokens = re.findall(r"[\w\-]+", query.lower())
    tokens = [t for t in tokens if len(t) > 1][:24]
    if not tokens:
        return '""'
    return " OR ".join(f'"{t}"' for t in tokens)
