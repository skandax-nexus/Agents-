"""Tiny stdlib HTTP helper so the offline backends need no third-party packages."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from .base import LLMError


def post_json(url: str, payload: dict[str, Any], *, headers: dict[str, str] | None = None,
              timeout: float = 600.0) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - local URLs by default
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise LLMError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"Cannot reach {url}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise LLMError(f"Timed out talking to {url}") from exc


def get_json(url: str, *, timeout: float = 3.0) -> dict[str, Any] | list[Any]:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))
