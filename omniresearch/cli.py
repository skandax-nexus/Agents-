"""Command-line interface.

    omniresearch ask "How do ocean currents regulate regional climate?"
    omniresearch index ~/papers ~/notes
    omniresearch search "coral bleaching thresholds"
    omniresearch domains
    omniresearch doctor
    omniresearch chat
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import __version__
from .config import DEPTH_PRESETS, load_config
from .domains import categories, list_domains
from .llm import LLMError, build_backend, probe_backends


def _print_event(kind: str, data: dict) -> None:
    agent = data.get("agent", "")
    if kind == "start":
        print(f"[omniresearch] backend={data['backend']} online={data['online']} depth={data['depth']}")
    elif kind == "phase":
        print(f"\n== {data['name']} ==")
    elif kind == "plan":
        print(data["plan"])
    elif kind == "tool_call":
        args = ", ".join(f"{k}={str(v)[:60]!r}" for k, v in data.get("input", {}).items())
        print(f"  [{agent}] -> {data['tool']}({args})")
    elif kind == "tool_result" and data.get("error"):
        print(f"  [{agent}]    error: {data.get('preview', '')[:120]}")
    elif kind == "agent_final":
        print(f"  [{agent}] done in {data.get('steps')} step(s)" + (" (forced)" if data.get("forced") else ""))
    elif kind == "critique":
        print(f"  verdict: {data.get('verdict')} | follow-ups: {data.get('followups')}")
    elif kind == "done":
        print(f"\n[omniresearch] finished in {data['elapsed']:.0f}s -> {data['path']}")


def _config_from_args(args: argparse.Namespace):
    online = None
    if getattr(args, "offline", False):
        online = False
    elif getattr(args, "online", False):
        online = True
    return load_config(
        getattr(args, "config", None),
        backend=getattr(args, "backend", None),
        model=getattr(args, "model", None),
        depth=getattr(args, "depth", None),
        online=online,
        output_dir=getattr(args, "out", None),
        parallel_agents=getattr(args, "parallel", None),
        language=getattr(args, "language", None),
    )


# ---------------------------------------------------------------- commands
def cmd_ask(args: argparse.Namespace) -> int:
    from .agents.orchestrator import ResearchOrchestrator

    cfg = _config_from_args(args)
    question = " ".join(args.question).strip()
    if not question:
        print("Give a research question.", file=sys.stderr)
        return 2
    try:
        orchestrator = ResearchOrchestrator(cfg, on_event=None if args.quiet else _print_event)
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    report = orchestrator.research(question)
    print("\n" + "=" * 78 + "\n")
    print(report.report)
    if report.output_path:
        print(f"\nSaved: {report.output_path}")
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    from .agents.orchestrator import ResearchOrchestrator

    cfg = _config_from_args(args)
    try:
        orchestrator = ResearchOrchestrator(cfg, on_event=None if args.quiet else _print_event)
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"OmniResearch chat ({orchestrator.backend.describe()}, online={orchestrator.online}).")
    print("Type a question for a quick answer, '/research <question>' for a full multi-agent run, '/quit' to exit.")
    last_report = None
    while True:
        try:
            line = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line in {"/quit", "/exit", "/q"}:
            return 0
        if line.startswith("/research "):
            last_report = orchestrator.research(line[len("/research "):])
            print("\n" + last_report.report)
            continue
        print("\n" + orchestrator.ask(line, last_report))


def cmd_index(args: argparse.Namespace) -> int:
    from .tools.knowledge import KnowledgeBase

    cfg = _config_from_args(args)
    cfg.ensure_dirs()
    kb = KnowledgeBase(cfg.knowledge_db)
    total_i = total_s = 0
    for raw in args.paths:
        path = Path(raw).expanduser()
        if not path.exists():
            print(f"skip: {path} does not exist")
            continue
        started = time.time()
        indexed, skipped = kb.index_path(path, verbose=args.verbose)
        total_i += indexed
        total_s += skipped
        print(f"{path}: indexed {indexed}, skipped {skipped} (unchanged/unsupported) in {time.time() - started:.1f}s")
    stats = kb.stats()
    print(f"knowledge base: {stats['documents']} documents, {stats['chunks']} chunks at {stats['path']}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    from .tools.knowledge import KnowledgeBase

    cfg = _config_from_args(args)
    kb = KnowledgeBase(cfg.knowledge_db)
    hits = kb.search(" ".join(args.query), limit=args.limit)
    if not hits:
        print("no matches")
        return 1
    for h in hits:
        print(f"--- {h['path']} #chunk{h['chunk']} ({h['title']})\n{h['snippet']}\n")
    return 0


def cmd_domains(args: argparse.Namespace) -> int:
    for cat in categories():
        if args.category and not cat.lower().startswith(args.category.lower()):
            continue
        print(f"\n{cat}")
        for d in list_domains(cat):
            print(f"  {d.id:<26} {d.name}")
    print(f"\n{len(list_domains())} domains in {len(categories())} categories. Anything else routes to 'general'.")
    return 0


def cmd_memory(args: argparse.Namespace) -> int:
    from .tools.memory import Memory

    cfg = _config_from_args(args)
    mem = Memory(cfg.memory_db)
    rows = mem.search(" ".join(args.query), limit=20) if args.query else mem.recent(20)
    if not rows:
        print("memory is empty")
        return 0
    for r in rows:
        print(f"--- {time.strftime('%Y-%m-%d %H:%M', time.localtime(r['created']))} | {r['topic']} [{r['tags']}]\n{r['content'][:500]}\n")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    from .tools.web import has_connectivity

    cfg = _config_from_args(args)
    print(f"OmniResearch {__version__}")
    print(f"home: {cfg.home}   output: {cfg.output_dir}   depth: {cfg.depth}")
    print("\nModel backends:")
    for name, backend, ok in probe_backends(cfg):
        extra = ""
        if ok and hasattr(backend, "installed_models"):
            models = backend.installed_models()
            extra = f"  models: {', '.join(models[:6]) or '(none pulled)'}"
        print(f"  {'OK ' if ok else '-- '} {name:<10} {backend.describe()}{extra}")
    try:
        chosen = build_backend(cfg)
        print(f"\nAuto-selected backend: {chosen.describe()}")
    except LLMError as exc:
        print(f"\nNo usable backend: {exc}")
    net = has_connectivity()
    mode = "forced offline" if cfg.online is False else ("forced online" if cfg.online else "auto")
    print(f"\nConnectivity: {'online' if net else 'offline'} (mode: {mode})")
    try:
        from .tools.knowledge import KnowledgeBase
        stats = KnowledgeBase(cfg.knowledge_db).stats()
        print(f"Knowledge base: {stats['documents']} documents / {stats['chunks']} chunks")
    except Exception as exc:  # noqa: BLE001
        print(f"Knowledge base: unavailable ({exc})")
    try:
        import pypdf  # noqa: F401
        print("PDF support: yes (pypdf)")
    except ImportError:
        print("PDF support: no  (pip install pypdf)")
    try:
        import anthropic  # noqa: F401
        print("Claude API SDK: installed")
    except ImportError:
        print("Claude API SDK: not installed (pip install anthropic) - only needed for the online backend")
    print(f"\nDepth presets: " + "; ".join(f"{k}={v}" for k, v in DEPTH_PRESETS.items()))
    return 0


# ---------------------------------------------------------------- parser
def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="path to a config.toml")
    p.add_argument("--backend", choices=["auto", "ollama", "openai", "anthropic", "mock"], help="model backend")
    p.add_argument("--model", help="model name for the backend (e.g. llama3.1:8b, qwen2.5:14b)")
    p.add_argument("--offline", action="store_true", help="never use network tools")
    p.add_argument("--online", action="store_true", help="force network tools on")
    p.add_argument("--out", help="directory for reports (default research_output/)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="omniresearch",
                                     description="Offline-first multi-agent research across every domain.")
    parser.add_argument("--version", action="version", version=f"omniresearch {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ask", help="run a full multi-agent research session on a question")
    p.add_argument("question", nargs="+")
    p.add_argument("--depth", choices=list(DEPTH_PRESETS), help="quick | standard | deep")
    p.add_argument("--parallel", type=int, help="specialists to run concurrently")
    p.add_argument("--language", help="language for the report (default English)")
    p.add_argument("--quiet", action="store_true", help="only print the final report")
    _add_common(p)
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("chat", help="interactive session with the research agent")
    p.add_argument("--depth", choices=list(DEPTH_PRESETS))
    p.add_argument("--parallel", type=int)
    p.add_argument("--language")
    p.add_argument("--quiet", action="store_true")
    _add_common(p)
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("index", help="index local documents into the offline knowledge base")
    p.add_argument("paths", nargs="+")
    p.add_argument("-v", "--verbose", action="store_true")
    _add_common(p)
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("search", help="search the offline knowledge base")
    p.add_argument("query", nargs="+")
    p.add_argument("--limit", type=int, default=8)
    _add_common(p)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("domains", help="list the research domains the system covers")
    p.add_argument("--category")
    p.set_defaults(func=cmd_domains)

    p = sub.add_parser("memory", help="show or search what earlier runs learned")
    p.add_argument("query", nargs="*")
    _add_common(p)
    p.set_defaults(func=cmd_memory)

    p = sub.add_parser("doctor", help="check backends, connectivity and the knowledge base")
    _add_common(p)
    p.set_defaults(func=cmd_doctor)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
