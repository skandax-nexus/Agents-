"""Run Python for calculations, simulations, unit conversions and data analysis.

Runs in a subprocess with a timeout and an isolated working directory.  It is a
convenience sandbox, not a security boundary: only run the system with models
and documents you trust.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .base import Tool, ToolResult

_PRELUDE = """
import math, json, statistics, itertools, collections, re, datetime, fractions, decimal, random
try:
    import numpy as np
except Exception:
    np = None
"""


def python_tool(workdir: Path, timeout: int = 60) -> Tool:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    def run(code: str) -> ToolResult:
        code = str(code)
        if "```" in code:  # tolerate fenced code from small models
            inner = code.split("```")
            code = inner[1] if len(inner) > 1 else code
            if code.startswith("python"):
                code = code[len("python"):]
        script = _PRELUDE + "\n" + code
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", script],
                capture_output=True, text=True, timeout=timeout, cwd=str(workdir),
            )
        except subprocess.TimeoutExpired:
            return ToolResult(f"Execution timed out after {timeout}s.", error=True)
        out = proc.stdout[-6000:]
        err = proc.stderr[-2000:]
        if proc.returncode != 0:
            return ToolResult(f"stdout:\n{out}\nstderr:\n{err}", error=True)
        return ToolResult(out if out.strip() else "(no output - use print() to show results)" + ("\n" + err if err else ""))

    return Tool(
        "python",
        "Execute Python code for calculations, statistics, simulations, unit conversions or data processing. "
        "math, statistics, json, re, datetime and numpy (if installed) are pre-imported. Print what you want to see.",
        {"code": "string - python source code"},
        run, required=("code",),
    )
