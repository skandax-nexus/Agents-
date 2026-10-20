"""Online tools (used only when connectivity is available and not disabled).

- web_search: DuckDuckGo HTML endpoint, no API key.
- web_fetch:  fetch a URL and return readable text; the page is also added to the
              local knowledge base so it stays available offline afterwards.
- wikipedia:  Wikipedia REST summary + extract, or an offline kiwix-serve mirror.
"""
from __future__ import annotations

import json
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from typing import TYPE_CHECKING

from .base import Tool, ToolError, ToolResult
from .textutil import html_to_text

if TYPE_CHECKING:
    from .knowledge import KnowledgeBase

_UA = "Mozilla/5.0 (compatible; OmniResearch/0.1; +https://github.com/)"
_RESULT = re.compile(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.S)
_SNIPPET = re.compile(r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', re.S)


def _get(url: str, timeout: float = 20.0, data: bytes | None = None) -> str:
    req = urllib.request.Request(url, data=data, headers={"User-Agent": _UA, "Accept": "text/html,application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            charset = resp.headers.get_content_charset() or "utf-8"
            return resp.read(2_000_000).decode(charset, "replace")
    except urllib.error.HTTPError as exc:
        raise ToolError(f"HTTP {exc.code} for {url}") from exc
    except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
        raise ToolError(f"Network error for {url}: {exc}") from exc


def has_connectivity(timeout: float = 3.0) -> bool:
    for host in ("duckduckgo.com", "en.wikipedia.org", "1.1.1.1"):
        try:
            socket.create_connection((host, 443), timeout=timeout).close()
            return True
        except OSError:
            continue
    return False


def _unwrap_ddg(href: str) -> str:
    if href.startswith("//"):
        href = "https:" + href
    parsed = urllib.parse.urlparse(href)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        target = urllib.parse.parse_qs(parsed.query).get("uddg", [""])[0]
        return urllib.parse.unquote(target) or href
    return href


def web_tools(kb: "KnowledgeBase | None" = None, kiwix_url: str | None = None, online: bool = True) -> list[Tool]:
    tools: list[Tool] = []

    def search(query: str, limit: int = 6) -> ToolResult:
        try:
            limit = max(1, min(int(limit), 10))
        except (TypeError, ValueError):
            limit = 6
        data = urllib.parse.urlencode({"q": str(query)}).encode()
        page = _get("https://html.duckduckgo.com/html/", data=data)
        links = _RESULT.findall(page)
        snippets = _SNIPPET.findall(page)
        if not links:
            return ToolResult("No web results (search engine returned nothing or blocked the request).")
        lines, sources = [], []
        for i, (href, title) in enumerate(links[:limit]):
            url = _unwrap_ddg(href)
            snippet = html_to_text(snippets[i]) if i < len(snippets) else ""
            sources.append(url)
            lines.append(f"[{url}]\n{html_to_text(title)}\n{snippet}")
        return ToolResult("\n\n".join(lines), sources=sources)

    def fetch(url: str, max_chars: int = 8000) -> ToolResult:
        url = str(url).strip()
        if not url.startswith(("http://", "https://")):
            raise ToolError("URL must start with http:// or https://")
        try:
            limit = max(1000, min(int(max_chars), 40000))
        except (TypeError, ValueError):
            limit = 8000
        raw = _get(url)
        text = html_to_text(raw) if "<" in raw[:2000] else raw
        title_match = re.search(r"<title[^>]*>(.*?)</title>", raw, re.S | re.I)
        title = html_to_text(title_match.group(1))[:200] if title_match else url
        if kb is not None and text.strip():
            try:
                kb.add_text(title, text, path=url)   # cache for offline use
            except Exception:  # noqa: BLE001
                pass
        suffix = "" if len(text) <= limit else f"\n... [truncated; full page cached in knowledge base as {url}]"
        return ToolResult(f"# {title}\n{text[:limit]}{suffix}", sources=[url])

    def wikipedia(topic: str) -> ToolResult:
        topic = str(topic).strip()
        if kiwix_url:
            q = urllib.parse.quote(topic)
            page = _get(f"{kiwix_url.rstrip('/')}/search?pattern={q}&books.name=wikipedia")
            text = html_to_text(page)
            return ToolResult(text[:8000], sources=[f"{kiwix_url} (offline Wikipedia): {topic}"])
        if not online:
            raise ToolError("Wikipedia is unavailable offline (set kiwix_url for an offline mirror).")
        q = urllib.parse.quote(topic.replace(" ", "_"))
        try:
            summary = json.loads(_get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{q}"))
        except ToolError:
            # fall back to search
            params = urllib.parse.urlencode({"action": "query", "list": "search", "srsearch": topic, "format": "json"})
            res = json.loads(_get(f"https://en.wikipedia.org/w/api.php?{params}"))
            hits = res.get("query", {}).get("search", [])
            if not hits:
                return ToolResult("No Wikipedia article found.")
            titles = ", ".join(h["title"] for h in hits[:8])
            return ToolResult(f"No exact article. Closest titles: {titles}")
        title = summary.get("title", topic)
        params = urllib.parse.urlencode({"action": "query", "prop": "extracts", "explaintext": 1,
                                         "titles": title, "format": "json", "exchars": 6000})
        extract = json.loads(_get(f"https://en.wikipedia.org/w/api.php?{params}"))
        pages = extract.get("query", {}).get("pages", {})
        body = next(iter(pages.values())).get("extract", "") if pages else summary.get("extract", "")
        url = summary.get("content_urls", {}).get("desktop", {}).get("page", f"https://en.wikipedia.org/wiki/{q}")
        if kb is not None and body:
            try:
                kb.add_text(f"Wikipedia: {title}", body, path=url)
            except Exception:  # noqa: BLE001
                pass
        return ToolResult(f"# {title}\n{body}", sources=[url])

    if online:
        tools.append(Tool("web_search", "Search the web (DuckDuckGo). Returns URLs, titles and snippets.",
                          {"query": "string", "limit": "int, optional"}, search, offline=False, required=("query",)))
        tools.append(Tool("web_fetch", "Fetch a web page or online document and return its readable text. Pages are cached offline.",
                          {"url": "string - http(s) URL", "max_chars": "int, optional"}, fetch, offline=False, required=("url",)))
    if online or kiwix_url:
        tools.append(Tool("wikipedia", "Look up an encyclopaedia article (Wikipedia online, or an offline kiwix mirror if configured).",
                          {"topic": "string - article title or topic"}, wikipedia, offline=bool(kiwix_url), required=("topic",)))
    return tools
