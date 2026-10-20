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
    elif kind == "deliberation":
        print(f"  [{agent}] thinking: {data.get('difficulty') or '?'} problem, {data.get('hypotheses', 0)} hypotheses; "
              f"strategy: {str(data.get('strategy', ''))[:110]}")
    elif kind == "verification":
        print(f"  [{agent}] self-check: {'pass' if data.get('passes') else 'FAIL'} "
              f"(confidence {data.get('confidence', 0):.2f}, {data.get('problems', 0)} problem(s))")
    elif kind == "retry":
        print(f"  [{agent}] retrying after attempt {data.get('attempt')}: {data.get('reason', '')[:120]}")
    elif kind == "classified":
        print(f"  type: {data.get('type')} | difficulty: {data.get('difficulty')} | strategy: {data.get('strategy')} - {data.get('why', '')}")
    elif kind == "instant_recall":
        print(f"  answered instantly from memory (learned {data.get('learned_at')}, similarity {data.get('similarity'):.2f})")
    elif kind == "practice_generated":
        print(f"\n== practice: {data['count']} problem(s) in {data['domain']} ==")
    elif kind == "practice_solved":
        print(f"  [{data['domain']}] {'KEPT' if data['passed'] else 'discarded'} (confidence {data.get('confidence')}, {data.get('attempts')} attempt(s))")
    elif kind == "dataset":
        print(f"\n== dataset: {data['train']} train / {data['eval']} eval examples, {data['pairs']} preference pairs ==")
    elif kind == "model_loaded":
        print(f"  model {data['base']} on {data['device']}: {data['trainable_params']:,} trainable of {data['total_params']:,} params")
    elif kind == "train_start":
        print(f"  training on {data['examples']} examples, {data['updates']} updates (eval loss before: {data.get('eval_loss_before')})")
    elif kind == "train_step":
        print(f"  step {data['update']}/{data['of']} loss {data['loss']} ({data['elapsed']}s)")
    elif kind == "train_done":
        print(f"  trained: loss {data.get('train_loss_first')} -> {data.get('train_loss_last')}, eval {data.get('eval_loss_before')} -> {data.get('eval_loss_after')} in {data.get('elapsed_seconds')}s")
    elif kind == "exported":
        print(f"  export: {'created in Ollama as ' + data['ollama_model'] if data.get('created') else data.get('instructions')}")
    elif kind == "round_done":
        print(f"\n== round {data['round']}: {data['note']} ==")
    elif kind == "learned":
        print(f"  learned: {data.get('lessons', 0)} lesson(s), {data.get('facts', 0)} fact(s), {data.get('skills', 0)} skill(s)"
              + (f" | {data.get('self_assessment')}" if data.get("self_assessment") else ""))
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


def cmd_solve(args: argparse.Namespace) -> int:
    from .cognition.solver import ProblemSolver

    cfg = _config_from_args(args)
    problem = " ".join(args.problem).strip()
    if not problem:
        print("Describe the problem to solve.", file=sys.stderr)
        return 2
    try:
        solver = ProblemSolver(cfg, on_event=None if args.quiet else _print_event)
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    sol = solver.solve(problem, fresh=args.fresh)
    print("\n" + "=" * 78 + "\n")
    print(f"# {sol.restated}\n")
    print(sol.answer)
    conf = f"{sol.confidence:.2f}" if sol.confidence is not None else "n/a"
    print(f"\n[{sol.ptype} | {sol.strategy} | confidence {conf} | {sol.attempts} attempt(s)]")
    if sol.output_path:
        print(f"Saved: {sol.output_path}")
    return 0


def _learner(cfg):
    from .cognition.experience import Experience
    from .cognition.instant import InstantLearner
    from .tools.knowledge import KnowledgeBase

    cfg.ensure_dirs()
    kb = KnowledgeBase(cfg.knowledge_db)
    return InstantLearner(kb, Experience(cfg.experience_db), cfg.answers_db)


def cmd_learn(args: argparse.Namespace) -> int:
    """Teach the agent instantly: text, files/directories, or URLs."""
    cfg = _config_from_args(args)
    learner = _learner(cfg)
    done = 0
    for path in args.file or []:
        info = learner.learn_file(path)
        print(f"file {path}: " + (f"indexed {info['files_indexed']} file(s)" if info.get("stored") else f"not stored ({info.get('reason') or 'nothing new'})"))
        done += int(bool(info.get("stored")))
    if args.url:
        from .tools.web import web_tools
        from .tools.registry import ToolRegistry
        fetch = ToolRegistry(web_tools(learner.kb, kiwix_url=cfg.kiwix_url, online=True)).get("web_fetch")
        for url in args.url:
            info = learner.learn_url(url, fetch)
            print(f"url {url}: " + ("stored" if info.get("stored") else f"failed ({info.get('reason')})"))
            done += int(bool(info.get("stored")))
    text = " ".join(args.text).strip()
    if text == "-":
        text = sys.stdin.read().strip()
    if text:
        info = learner.learn_text(text, title=args.title)
        print(f"learned '{info['title']}' ({info['chars']} chars)" if info.get("stored") else f"not stored ({info.get('reason')})")
        done += int(bool(info.get("stored")))
    if not done and not (args.file or args.url or text):
        print("Give something to learn: text, --file PATH, or --url URL", file=sys.stderr)
        return 2
    st = learner.stats()
    print(f"knowledge: {st['documents']} documents / {st['chunks']} chunks; learned answers: {st['answers']}")
    return 0


def cmd_answer(args: argparse.Namespace) -> int:
    """Quick answer: instantly from what was learned, otherwise one specialist (and then it is learned)."""
    from .agents.orchestrator import ResearchOrchestrator

    cfg = _config_from_args(args)
    question = " ".join(args.question).strip()
    if not question:
        print("Ask a question.", file=sys.stderr)
        return 2
    if cfg.instant_recall and not args.fresh:
        hit = _learner(cfg).recall(question, min_similarity=cfg.recall_similarity)
        if hit:   # no model needed at all
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(hit["created"]))
            print(hit["answer"])
            print(f"\n(answered instantly from what I learned on {when}; similarity {hit['similarity']:.2f}. --fresh to redo)")
            return 0
    try:
        orchestrator = ResearchOrchestrator(cfg, on_event=None if args.quiet else _print_event)
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(orchestrator.ask(question, fresh=args.fresh))
    return 0


def cmd_learned(args: argparse.Namespace) -> int:
    cfg = _config_from_args(args)
    learner = _learner(cfg)
    if args.forget:
        learner.forget(args.forget)
        print(f"forgot answer {args.forget}")
        return 0
    st = learner.stats()
    print(f"learned answers: {st['answers']} (used instantly {st['instant_hits']} times); "
          f"knowledge: {st['documents']} documents; experience: {st['experience']['total']} entries")
    query = " ".join(args.query)
    if query:
        hit = learner.recall(query, min_similarity=0.0)
        if hit:
            print(f"\nclosest: #{hit['id']} (similarity {hit['similarity']}) {hit['question']}\n{hit['answer'][:800]}")
        return 0
    for r in learner.recent(20):
        print(f"\n#{r['id']} [{time.strftime('%Y-%m-%d', time.localtime(r['created']))}, used {r['uses']}x] {r['question']}\n  {r['answer'][:300].replace(chr(10), ' ')}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Optional: fine-tune a local model on everything the system has verified."""
    from .training.loop import SelfTrainingLoop, current_student
    from .training.dataset import DatasetBuilder

    cfg = _config_from_args(args)
    cfg.ensure_dirs()
    ev = None if args.quiet else _print_event
    if args.train_cmd == "status":
        s = current_student(cfg)
        print("student model: " + (f"{s['ollama_model']} (round {s['round']}, base {s['base_model']}, in Ollama: {s['created_in_ollama']})" if s else "none yet"))
        summary = cfg.home / "training" / "data" / "summary.json"
        print("dataset: " + (summary.read_text().strip() if summary.exists() else "not built"))
        return 0
    if args.train_cmd == "dataset":
        from .cognition.experience import Experience
        b = DatasetBuilder(min_confidence=max(0.7, cfg.min_confidence))
        b.add_traces(cfg.output_dir)
        b.add_experience(Experience(cfg.experience_db))
        print(b.write(cfg.home / "training" / "data"))
        return 0
    from .cognition.solver import ProblemSolver
    try:
        solver = ProblemSolver(cfg, on_event=ev)
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    def student_factory(merged_dir):
        return None  # evaluated by eval-loss unless the student is imported into Ollama

    loop = SelfTrainingLoop(cfg, solver, base_model=args.base, on_event=ev, train=args.train_cmd != "practice")
    if args.train_cmd == "practice":
        print(loop.practice(args.domains, args.per_domain, args.difficulty, args.max_problems))
        return 0
    if args.train_cmd == "eval":
        print(loop.evaluate(solver.backend, args.limit))
        return 0
    if args.train_cmd == "run":
        if not args.base:
            print("--base <hf-model-id-or-dir> is required (e.g. Qwen/Qwen2.5-0.5B-Instruct)", file=sys.stderr)
            return 2
        try:
            stats, trainer = loop.train(args.base, epochs=args.epochs, max_examples=args.max_examples)
        except Exception as exc:  # noqa: BLE001
            print(f"error: {exc}", file=sys.stderr)
            return 1
        merged = trainer.merge_and_save()
        print(trainer.export_ollama(args.name, merged))
        return 0
    if args.train_cmd == "auto":
        for r in loop.run(args.rounds, domains=args.domains, per_domain=args.per_domain, epochs=args.epochs,
                          student_name=args.name):
            print(f"round {r.round}: {r.note} | practice kept {r.practice.get('kept')} | dataset {r.dataset.get('train')} | "
                  f"eval base {r.eval_base.get('combined')} student {r.eval_student.get('combined')}")
        return 0
    return 2


def cmd_feedback(args: argparse.Namespace) -> int:
    from .cognition.experience import Experience

    cfg = _config_from_args(args)
    cfg.ensure_dirs()
    text = " ".join(args.text).strip()
    if not text:
        print("Give the feedback text, e.g. omniresearch feedback \"lead with numbers; shorter reports\"", file=sys.stderr)
        return 2
    exp = Experience(cfg.experience_db)
    eid = exp.add_feedback(text, session=args.session or "")
    print(f"Stored feedback #{eid}. It will be shown to the agents on future runs.")
    return 0


def cmd_experience(args: argparse.Namespace) -> int:
    from .cognition.experience import Experience, KINDS

    cfg = _config_from_args(args)
    exp = Experience(cfg.experience_db)
    if args.forget:
        exp.delete(args.forget)
        print(f"Deleted entry {args.forget}.")
        return 0
    stats = exp.stats()
    print("Experience store: " + ", ".join(f"{k}={stats[k]}" for k in KINDS) + f" (total {stats['total']})")
    query = " ".join(args.query)
    rows = exp.recall(query, limit=20, kinds=[args.kind] if args.kind else None) if query else exp.recent(20, args.kind)
    for r in rows:
        print(f"\n#{r.id} [{r.kind}] {r.title}  (domains: {r.domains or '-'}, uses: {r.uses}, score: {r.score:g})\n"
              f"  {r.content[:600]}")
    if not rows:
        print("(nothing learned yet - run `omniresearch ask` or `omniresearch solve`, or add `omniresearch feedback`)")
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
    print("Type a question for a quick answer, '/research <q>' for a full multi-agent run, '/solve <problem>' for the "
          "problem solver, '/feedback <text>' to teach it, '/quit' to exit.")
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
        if line.startswith("/solve "):
            from .cognition.solver import ProblemSolver
            sol = ProblemSolver(cfg, orchestrator, on_event=orchestrator.on_event).solve(line[len("/solve "):])
            print("\n" + sol.answer)
            continue
        if line.startswith("/feedback "):
            orchestrator.experience.add_feedback(line[len("/feedback "):], session=orchestrator.session_id)
            print("noted - I will apply that from now on.")
            continue
        if line.startswith("/learn "):
            info = orchestrator.learner.learn_text(line[len("/learn "):], session=orchestrator.session_id)
            print(f"learned '{info.get('title')}' - I know this from now on." if info.get("stored") else "nothing to learn")
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
        from .cognition.experience import Experience
        es = Experience(cfg.experience_db).stats()
        print(f"Experience: {es['lesson']} lessons, {es['fact']} facts, {es['skill']} skills, {es['feedback']} feedback")
    except Exception as exc:  # noqa: BLE001
        print(f"Experience: unavailable ({exc})")
    try:
        st = _learner(cfg).stats()
        print(f"Learned answers: {st['answers']} (instant hits so far: {st['instant_hits']})")
    except Exception as exc:  # noqa: BLE001
        print(f"Learned answers: unavailable ({exc})")
    try:
        import torch  # noqa: F401
        import peft  # noqa: F401
        print("Training (optional): available")
    except ImportError:
        print("Training (optional): not installed (pip install 'omniresearch[train]') - instant learning works without it")
    preset = cfg.preset()
    print(f"Cognition: deliberate={bool(preset['deliberate'])} verify={bool(preset['verify'])} "
          f"max_attempts={preset['max_attempts']} reflect={cfg.reflect}")
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

    p = sub.add_parser("solve", help="solve any problem: calculation, design, decision, troubleshooting, plan, puzzle ...")
    p.add_argument("problem", nargs="+")
    p.add_argument("--fresh", action="store_true", help="ignore learned answers and solve again")
    p.add_argument("--depth", choices=list(DEPTH_PRESETS))
    p.add_argument("--language")
    p.add_argument("--quiet", action="store_true")
    _add_common(p)
    p.set_defaults(func=cmd_solve)

    p = sub.add_parser("learn", help="teach it instantly: text, --file paths/dirs, --url pages (no training needed)")
    p.add_argument("text", nargs="*", help="text to learn ('-' reads stdin)")
    p.add_argument("--file", action="append", help="file or directory to learn (repeatable)")
    p.add_argument("--url", action="append", help="web page to learn (repeatable, needs network)")
    p.add_argument("--title")
    _add_common(p)
    p.set_defaults(func=cmd_learn)

    p = sub.add_parser("answer", help="quick answer: instant if already learned, else one specialist (then learned)")
    p.add_argument("question", nargs="+")
    p.add_argument("--fresh", action="store_true", help="ignore learned answers and work it out again")
    p.add_argument("--depth", choices=list(DEPTH_PRESETS))
    p.add_argument("--language")
    p.add_argument("--quiet", action="store_true")
    _add_common(p)
    p.set_defaults(func=cmd_answer)

    p = sub.add_parser("learned", help="show learned answers (or find the closest to a query); --forget ID")
    p.add_argument("query", nargs="*")
    p.add_argument("--forget", type=int, metavar="ID")
    _add_common(p)
    p.set_defaults(func=cmd_learned)

    p = sub.add_parser("train", help="optional: fine-tune a local model on the system's own verified work")
    tsub = p.add_subparsers(dest="train_cmd", required=True)
    for name, helptext in [("status", "show the current student model and dataset"),
                           ("dataset", "build SFT / preference datasets from traces and experience"),
                           ("practice", "self-set and solve practice problems; verified ones become training data"),
                           ("run", "LoRA-train a base model on the dataset and export it to Ollama"),
                           ("eval", "score the current backend on the held-out set"),
                           ("auto", "full cycle: practice -> dataset -> train -> eval -> adopt if better")]:
        tp = tsub.add_parser(name, help=helptext)
        tp.add_argument("--base", help="base model: Hugging Face id or local directory")
        tp.add_argument("--name", default="omniresearch-student", help="Ollama name for the trained model")
        tp.add_argument("--domains", nargs="*", help="domain ids to practise (default: all)")
        tp.add_argument("--per-domain", type=int, default=2)
        tp.add_argument("--difficulty", default="moderate")
        tp.add_argument("--max-problems", type=int)
        tp.add_argument("--epochs", type=float, default=2.0)
        tp.add_argument("--max-examples", type=int)
        tp.add_argument("--rounds", type=int, default=1)
        tp.add_argument("--limit", type=int, default=40)
        tp.add_argument("--depth", choices=list(DEPTH_PRESETS))
        tp.add_argument("--quiet", action="store_true")
        _add_common(tp)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("feedback", help="guide the agent: feedback is recalled on every future run")
    p.add_argument("text", nargs="+")
    p.add_argument("--session", help="session id the feedback refers to")
    _add_common(p)
    p.set_defaults(func=cmd_feedback)

    p = sub.add_parser("experience", help="show what the agent has learned (lessons, facts, skills, feedback)")
    p.add_argument("query", nargs="*")
    p.add_argument("--kind", choices=["lesson", "fact", "skill", "feedback"])
    p.add_argument("--forget", type=int, metavar="ID", help="delete an entry")
    _add_common(p)
    p.set_defaults(func=cmd_experience)

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
