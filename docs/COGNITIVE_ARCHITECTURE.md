# Cognitive architecture

OmniResearch agents follow the cycle a careful human uses on an unfamiliar problem.
Each stage is a concrete component in `omniresearch/cognition/`.

```
            ┌──────────────── LEARN (Reflector -> Experience store) ◄─────────────┐
            ▼                                                                     │
 problem ─► UNDERSTAND ─► RECALL ─► DELIBERATE ─► PLAN ─► ACT ─► VERIFY ─► REFLECT ┘
            (restate,     (lessons,  (hypotheses, (strategy, (tools,   (check   (what went
             knowns,       facts,     alternatives, steps)     evidence, against   right / wrong,
             unknowns)     skills)    risks)                   compute)  criteria)  retry?)
```

| Stage | Human analogue | Component |
|---|---|---|
| Understand | Restate the problem in your own words; list what is known, unknown, assumed | `ThinkingAgent.deliberate()` |
| Recall | "Have I seen something like this before?" | `Experience.recall()` injected as *Relevant experience* |
| Deliberate | Generate several hypotheses / approaches, pick one, define what success looks like | `ThinkingAgent.deliberate()` -> `Thought` |
| Plan & Act | Work the plan with tools: search, read, compute, note | `Agent.run()` tool loop (JSON protocol) |
| Verify | Check the answer against the success criteria; estimate confidence | `ThinkingAgent.verify()` -> `Verification` |
| Retry | If the check fails, change strategy and try again (bounded) | `ThinkingAgent.run()` attempts loop |
| Reflect & Learn | Extract lessons, facts and reusable procedures for next time | `Reflector` -> `Experience` (lessons, facts, skills, feedback) |

## Problem-solving strategies

`ProblemSolver` (`omniresearch solve`) classifies a problem (research, calculation, design,
decision, troubleshooting, planning, creative, logic, how-to, coding) and picks a strategy:

- **decomposition** - split into sub-problems, solve, recombine
- **first principles** - derive from fundamental laws / definitions
- **analogy** - map to a solved problem in another field
- **working backwards** - start from the goal state
- **constraint analysis** - list hard constraints, prune the space
- **simplify / special case** - solve an easier version first
- **trial and verification** - propose, test with computation, iterate
- **inversion** - ask what would make it fail
- **simulation** - model it in code

Research-heavy problems call the multi-agent `ResearchOrchestrator` as a tool.

## Learning

The `Experience` store (SQLite FTS5, `~/.omniresearch/experience.db`) holds four kinds of entries:

| kind | written by | example |
|---|---|---|
| `lesson` | Reflector after each run | "For hydrology questions, search the knowledge base with basin names, not generic terms." |
| `fact` | Reflector | "Global surface ocean pH fell ~0.1 since pre-industrial (source: ...)" |
| `skill` | Reflector / analyst via `skill_save` | a named, reusable Python procedure (e.g. aragonite saturation from pH & DIC) |
| `feedback` | the user via `omniresearch feedback` | "Reports are too long; lead with numbers." |

Before each agent acts, the most relevant entries are recalled and shown to it; after each run the
Reflector writes new ones. Use counts and user feedback weight what gets recalled. This is
episodic + procedural learning: the model's weights never change, but the system's behaviour
improves with every run, offline.
