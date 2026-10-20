# Learning: instant, and (optionally) into the weights

OmniResearch learns in two ways. The first is what you want almost all of the time.

## 1. Instant learning (default, no training, works offline)

Tell it something once and it knows it from then on. Nothing is trained; everything is
stored immediately in local, searchable stores and recalled when relevant.

| You do | What is stored | Where it is used |
|---|---|---|
| `omniresearch learn "The Rhine is 1,233 km long"` | fact in the experience store + searchable note in the knowledge base | every agent recalls it when a task touches it |
| `omniresearch learn --file ~/papers --file notes.md` | documents chunked and full-text indexed | `knowledge_search` in every run |
| `omniresearch learn --url https://...` | page text cached offline | same |
| `omniresearch feedback "lead with numbers"` | guidance with top priority | shown to the planner and every agent |
| any question it answers (`answer`, `solve`, `ask`) | the verified answer, keyed by the question | a closely similar later question is answered **instantly, with no model call** |
| an agent finds something useful mid-task | it calls the `learn` tool itself | available to all later tasks |
| after every run | the Reflector stores lessons, facts and reusable Python skills | recalled by relevance and domain |

`omniresearch answer "<question>"` is the fast path: if a learned answer matches closely
enough (`recall_similarity`, default 0.6), it is returned at once; otherwise one specialist
works it out and the result is learned. `--fresh` forces a redo. `omniresearch learned`
lists what it knows; `--forget ID` removes an answer.

Storage: `~/.omniresearch/knowledge.db` (documents), `experience.db` (lessons, facts,
skills, feedback), `answers.db` (question -> answer), `memory.db` (run summaries).

## 2. Self-training into a local model (optional)

If you also want the *model itself* to improve, the training module turns the system's own
verified work into a fine-tuned student:

```
traces + experience + feedback + self-practice  ->  dataset (SFT + preference pairs)
   -> LoRA fine-tune of a local base model -> evaluate vs. the current model -> adopt if better
   -> export to Ollama -> becomes the default offline model
```

- **Only verified work is used.** An answer enters the dataset only if the agent's own
  self-check passed at or above the confidence bar. Unverified output is never trained on.
- **Preference pairs come free.** When an answer failed verification and the retry passed,
  the pair (rejected, chosen) is kept for DPO-style training (`pip install trl`).
- **Distillation.** Runs made with the online Claude backend become training targets for the
  local student, so the student inherits the teacher's behaviour.
- **Self-practice (STaR-style).** `omniresearch train practice` has the system set itself
  problems per domain, solve them with verification, and keep the verified solutions.
- **Measured, not assumed.** Held-out examples score the base and the student (numeric match,
  token overlap, JSON validity); the student is adopted only if it is not worse.

```bash
pip install "omniresearch[train]"                       # torch, transformers, peft
omniresearch train practice --domains hydrology ecology --per-domain 5
omniresearch train dataset
omniresearch train run --base Qwen/Qwen2.5-1.5B-Instruct --epochs 2
omniresearch train status
omniresearch train auto --base Qwen/Qwen2.5-1.5B-Instruct --rounds 3   # the whole cycle
```

Hardware: 0.5B-1.5B models train on CPU (slowly) or any GPU; 7B+ needs a GPU with 16 GB+.
The exported model is created in Ollama as `omniresearch-student` when `ollama` is on the
path; otherwise a Modelfile and conversion instructions are written.

### What self-training can and cannot do

It cannot make a small model know things it never saw: knowledge still comes from the
knowledge base, the web, and instant learning. What it does is make the local model better
at *the way this system works*: following the JSON tool protocol, structuring answers,
computing with code, citing, and the domain habits captured from verified runs and from a
stronger teacher. Instant learning is the right tool for facts; training is the right tool for
behaviour.
