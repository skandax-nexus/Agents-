# OmniResearch

An **offline-first, multi-agent research system** that can investigate *any* domain:
every natural and formal science, every environmental field (climate, atmosphere, oceans,
hydrology, ecology, conservation, forestry, soils, agriculture, energy, waste, pollution,
policy, economics, health, hazards, remote sensing, the cryosphere, cities, transport ...),
engineering, medicine, social sciences, humanities, business, law, the arts, and practical
everyday questions. Nothing is out of scope: unknown topics route to a general
interdisciplinary specialist instead of being refused.

It runs entirely on your machine with a local model (Ollama or any llama.cpp / LM Studio /
vLLM server), and can optionally use the Claude API and web tools when you are online.

```
question
  └─ Main agent (ResearchOrchestrator)
       ├─ Planner        picks domains, writes focused sub-questions
       ├─ Source scout   finds evidence in your local knowledge base (and the web if online)
       ├─ Specialists    one per sub-question, each primed with its field's methods & sources (parallel)
       ├─ Analyst        runs Python for numbers, statistics, simulations
       ├─ Critic         flags unsupported claims and gaps, proposes follow-ups  (loops)
       └─ Synthesizer    writes the final cited report  ->  research_output/*.md + trace.json
```

## Quick start (offline)

```bash
git clone <this repo> && cd Agents-
./scripts/setup_offline.sh                  # pip install + pulls llama3.1:8b into Ollama
omniresearch index ~/papers ~/notes         # build the offline knowledge base (txt, md, pdf, html, csv, json, code)
omniresearch ask --offline "How will permafrost thaw change the Arctic carbon budget by 2100?"
```

The report is printed and saved to `research_output/<date>-<slug>.md`, with a full trace of
every sub-agent, tool call and source in the matching `.trace.json`.

No installer? Run it in place: `python3 -m omniresearch ask "..."` (no third-party packages
are needed for offline use; `pypdf` adds PDF indexing).

## Commands

| Command | What it does |
|---|---|
| `omniresearch ask "<question>"` | full multi-agent research run (`--depth quick\|standard\|deep`) |
| `omniresearch chat` | interactive session; `/research <q>` runs the full pipeline, plain text gets a quick specialist answer grounded in the last report |
| `omniresearch index <paths...>` | index documents into the offline knowledge base (incremental) |
| `omniresearch search "<terms>"` | query the knowledge base directly |
| `omniresearch domains` | list the 70+ domains and categories |
| `omniresearch memory [query]` | what earlier runs learned (persisted across sessions) |
| `omniresearch doctor` | check backends, models, connectivity, knowledge base |

Useful flags: `--backend ollama|openai|anthropic|mock`, `--model <name>`, `--offline`,
`--online`, `--parallel N`, `--language <lang>`, `--out <dir>`, `--quiet`.

## Model backends

| Backend | Offline | How |
|---|---|---|
| **Ollama** (default) | yes | `ollama pull qwen2.5:14b` then `--backend ollama --model qwen2.5:14b` (auto-detected on `localhost:11434`) |
| **OpenAI-compatible** | yes | llama.cpp `llama-server`, LM Studio, vLLM, LocalAI, Jan on `localhost:8080/v1` (`--backend openai`) |
| **Claude API** | no | `pip install anthropic`, set `ANTHROPIC_API_KEY`, `--backend anthropic` (default model `claude-opus-5-5`, adaptive thinking, effort configurable, server-side refusal fallbacks on) |
| **mock** | yes | `--backend mock` dry-run of the whole pipeline with no model |

`auto` prefers offline engines and only falls back to the Claude API when a key is present.
Tool use is done with a plain JSON protocol, so the same agents work on small local models
that have no native function-calling.

Recommended local models: `qwen2.5:14b` or `llama3.1:8b` for general use; `qwen2.5:32b`,
`llama3.3:70b` or `deepseek-r1` for deep runs if you have the hardware. Raise `num_ctx` in the
config for larger models.

## Tools the agents can use

Offline (always):
- `knowledge_search` / `knowledge_read` - SQLite FTS5 full-text index over your documents
- `read_file` / `list_dir` - local files (txt, md, pdf, html, csv, json, code)
- `python` - calculations, statistics, simulations, unit conversions (numpy if installed)
- `memory_save` / `memory_recall` - persistent cross-session research memory

Online (only when connected and not `--offline`):
- `web_search` (DuckDuckGo, no key), `web_fetch`, `wikipedia`. Fetched pages are cached into the
  knowledge base so they remain searchable offline later.
- Point `kiwix_url` at a local [kiwix-serve](https://kiwix.org) mirror to get Wikipedia offline.

## Configuration

Copy `config.example.toml` to `./config.toml` or `~/.omniresearch/config.toml`, or use
environment variables (`OMNIRESEARCH_BACKEND`, `OMNIRESEARCH_MODEL`, `OMNIRESEARCH_DEPTH`,
`OMNIRESEARCH_ONLINE`, `OMNIRESEARCH_OUTPUT_DIR`, `OMNIRESEARCH_KIWIX_URL`, ...).

Depth presets:

| depth | sub-questions | tool steps / agent | critique rounds | domains |
|---|---|---|---|---|
| quick | 3 | 4 | 0 | 2 |
| standard | 5 | 8 | 1 | 4 |
| deep | 8 | 14 | 2 | 6 |

## Using it from Python

```python
from omniresearch import ResearchOrchestrator, load_config

cfg = load_config(backend="ollama", model="qwen2.5:14b", depth="deep", online=False)
agent = ResearchOrchestrator(cfg, on_event=lambda kind, data: print(kind, data))
report = agent.research("Compare the life-cycle emissions of green hydrogen and battery storage for grid balancing")
print(report.report)          # final Markdown report
print(report.plan.describe()) # domains + sub-questions
print(report.sources)         # every path/URL the sub-agents used
```

Add your own tool in a few lines:

```python
from omniresearch.tools import Tool, ToolResult
agent.tools.add(Tool("tide_table", "Look up tide heights for a port and date",
                     {"port": "string", "date": "YYYY-MM-DD"},
                     lambda port, date: ToolResult(f"..."), required=("port", "date")))
```

## Claude Code sub-agents

`.claude/agents/` contains the same team (orchestrator, domain specialist, source scout,
analyst, critic, synthesizer) as Claude Code sub-agent definitions, so you can run
`@research-orchestrator <question>` from Claude Code when you are online. The Python
package is the offline path.

## Development

```bash
pip install -e ".[dev]"
pytest -q           # 22 tests, no model or network needed
```

Layout: `omniresearch/llm` (backends), `omniresearch/domains` (registry + router),
`omniresearch/tools` (offline/online tools), `omniresearch/agents` (protocol, base loop,
sub-agents, orchestrator), `omniresearch/cli.py`.

## Notes

- The `python` tool runs code from the model in a subprocess with a timeout; it is a
  convenience, not a security sandbox. Use trusted models and documents.
- Reports are research syntheses, not professional medical, legal or financial advice.
