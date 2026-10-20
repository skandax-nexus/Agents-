"""LoRA fine-tuning of a local base model on the system's own verified work.

Optional dependencies: ``pip install "omniresearch[train]"`` (torch, transformers, peft).
Runs on CPU for small models (0.5B-1.5B) and on any GPU torch can see.  The result is a
merged Hugging Face model directory plus an Ollama Modelfile, so the trained student plugs
straight back into the offline backend.
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

from .dataset import load_jsonl


class TrainingUnavailable(RuntimeError):
    pass


def _require():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import peft  # noqa: F401
    except ImportError as exc:
        raise TrainingUnavailable(
            "Training needs torch, transformers and peft: pip install 'omniresearch[train]' "
            "(or: pip install torch transformers peft)") from exc


SIMPLE_TEMPLATE_START = {"system": "<|system|>\n", "user": "<|user|>\n", "assistant": "<|assistant|>\n"}


@dataclass
class TrainConfig:
    base_model: str                       # HF hub id or local directory
    output_dir: Path
    epochs: float = 2.0
    learning_rate: float = 2e-4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] | None = None   # None = auto (all linear attention/MLP projections found)
    max_seq_len: int = 2048
    batch_size: int = 1
    grad_accum: int = 8
    max_examples: int | None = None
    device: str = "auto"                  # auto | cpu | cuda | mps
    seed: int = 7
    warmup_steps: int = 10
    log_every: int = 10

    def to_dict(self) -> dict:
        d = asdict(self)
        d["output_dir"] = str(self.output_dir)
        return d


def format_chat(tokenizer, messages: list[dict], add_generation_prompt: bool = False) -> str:
    """Render a chat with the model's own template, or a plain fallback template."""
    template = getattr(tokenizer, "chat_template", None)
    if template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    out = ""
    for m in messages:
        out += SIMPLE_TEMPLATE_START.get(m["role"], f"<|{m['role']}|>\n") + m["content"].strip() + "\n"
    if add_generation_prompt:
        out += SIMPLE_TEMPLATE_START["assistant"]
    return out


class LoRATrainer:
    def __init__(self, cfg: TrainConfig, *, on_event: Callable[[str, dict], None] | None = None) -> None:
        _require()
        import torch

        self.cfg = cfg
        self.on_event = on_event or (lambda kind, data: None)
        if cfg.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None)
                                                                     and torch.backends.mps.is_available() else "cpu")
        else:
            self.device = cfg.device
        self.cfg.output_dir = Path(cfg.output_dir)
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = None
        self.model = None

    def _emit(self, kind: str, **data) -> None:
        try:
            self.on_event(kind, data)
        except Exception:  # noqa: BLE001
            pass

    # -- model ------------------------------------------------------------------
    def load(self) -> None:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.manual_seed(self.cfg.seed)
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
        dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        base = AutoModelForCausalLM.from_pretrained(self.cfg.base_model, dtype=dtype)
        targets = self.cfg.target_modules or self._auto_targets(base)
        lora = LoraConfig(r=self.cfg.lora_r, lora_alpha=self.cfg.lora_alpha, lora_dropout=self.cfg.lora_dropout,
                          target_modules=targets, task_type="CAUSAL_LM")
        self.model = get_peft_model(base, lora).to(self.device)
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        self._emit("model_loaded", base=self.cfg.base_model, device=self.device, targets=targets,
                   trainable_params=trainable, total_params=total)

    @staticmethod
    def _auto_targets(model) -> list[str]:
        import torch

        names = set()
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                leaf = name.split(".")[-1]
                if leaf in {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                            "c_attn", "c_proj", "c_fc", "query_key_value", "dense", "fc1", "fc2", "wq", "wk", "wv", "wo"}:
                    names.add(leaf)
        return sorted(names) or ["q_proj", "v_proj"]

    # -- data ------------------------------------------------------------------
    def _encode(self, row: dict) -> dict | None:
        msgs = row["messages"]
        prompt_msgs, answer = msgs[:-1], msgs[-1]["content"]
        prompt_text = format_chat(self.tokenizer, prompt_msgs, add_generation_prompt=True)
        full_text = format_chat(self.tokenizer, msgs)
        if not full_text.startswith(prompt_text):   # template quirk: fall back to concatenation
            full_text = prompt_text + answer + (self.tokenizer.eos_token or "")
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]
        if len(full_ids) > self.cfg.max_seq_len:
            full_ids = full_ids[: self.cfg.max_seq_len]
        if len(prompt_ids) >= len(full_ids):
            return None   # nothing to learn from (answer truncated away)
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        return {"input_ids": full_ids, "labels": labels}

    def _batches(self, rows: list[dict], shuffle: bool, seed: int):
        import random
        import torch

        encoded = [e for e in (self._encode(r) for r in rows) if e]
        idx = list(range(len(encoded)))
        if shuffle:
            random.Random(seed).shuffle(idx)
        pad = self.tokenizer.pad_token_id
        for i in range(0, len(idx), self.cfg.batch_size):
            chunk = [encoded[j] for j in idx[i:i + self.cfg.batch_size]]
            width = max(len(c["input_ids"]) for c in chunk)
            ids = torch.full((len(chunk), width), pad, dtype=torch.long)
            labels = torch.full((len(chunk), width), -100, dtype=torch.long)
            mask = torch.zeros((len(chunk), width), dtype=torch.long)
            for r, c in enumerate(chunk):
                n = len(c["input_ids"])
                ids[r, :n] = torch.tensor(c["input_ids"])
                labels[r, :n] = torch.tensor(c["labels"])
                mask[r, :n] = 1
            yield {"input_ids": ids.to(self.device), "labels": labels.to(self.device), "attention_mask": mask.to(self.device)}

    # -- training ----------------------------------------------------------------
    def evaluate_loss(self, rows: list[dict]) -> float | None:
        import torch

        if not rows or self.model is None:
            return None
        self.model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for batch in self._batches(rows, shuffle=False, seed=0):
                total += float(self.model(**batch).loss)
                n += 1
        self.model.train()
        return total / n if n else None

    def train_sft(self, sft_path: Path, eval_path: Path | None = None) -> dict:
        import torch

        if self.model is None:
            self.load()
        rows = load_jsonl(sft_path)
        if self.cfg.max_examples:
            rows = rows[: self.cfg.max_examples]
        if not rows:
            raise TrainingUnavailable("The SFT dataset is empty. Run `omniresearch train practice` or more research first.")
        eval_rows = load_jsonl(eval_path) if eval_path and Path(eval_path).exists() else []
        steps_per_epoch = max(1, math.ceil(len(rows) / self.cfg.batch_size))
        total_updates = max(1, math.ceil(steps_per_epoch * self.cfg.epochs / self.cfg.grad_accum))
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=self.cfg.learning_rate, weight_decay=0.0)

        def lr_at(update: int) -> float:
            if update < self.cfg.warmup_steps:
                return self.cfg.learning_rate * (update + 1) / self.cfg.warmup_steps
            progress = (update - self.cfg.warmup_steps) / max(1, total_updates - self.cfg.warmup_steps)
            return self.cfg.learning_rate * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

        eval_before = self.evaluate_loss(eval_rows)
        self._emit("train_start", examples=len(rows), eval_examples=len(eval_rows), updates=total_updates,
                   eval_loss_before=eval_before)
        self.model.train()
        started = time.time()
        update = micro = 0
        losses: list[float] = []
        epochs_full = int(math.ceil(self.cfg.epochs))
        for epoch in range(epochs_full):
            frac = min(1.0, self.cfg.epochs - epoch)
            for b, batch in enumerate(self._batches(rows, shuffle=True, seed=self.cfg.seed + epoch)):
                if b >= steps_per_epoch * frac:
                    break
                loss = self.model(**batch).loss / self.cfg.grad_accum
                loss.backward()
                losses.append(float(loss.detach()) * self.cfg.grad_accum)
                micro += 1
                if micro % self.cfg.grad_accum == 0:
                    for g in opt.param_groups:
                        g["lr"] = lr_at(update)
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    update += 1
                    if update % self.cfg.log_every == 0 or update == total_updates:
                        recent = sum(losses[-self.cfg.grad_accum * self.cfg.log_every:]) / max(1, len(losses[-self.cfg.grad_accum * self.cfg.log_every:]))
                        self._emit("train_step", update=update, of=total_updates, loss=round(recent, 4),
                                   elapsed=round(time.time() - started))
        if micro % self.cfg.grad_accum:
            opt.step()
            opt.zero_grad(set_to_none=True)
            update += 1
        eval_after = self.evaluate_loss(eval_rows)
        adapter_dir = self.cfg.output_dir / "adapter"
        self.model.save_pretrained(str(adapter_dir))
        self.tokenizer.save_pretrained(str(adapter_dir))
        window = max(1, min(20, len(losses) // 3))   # non-overlapping first/last windows even on tiny runs
        stats = {"examples": len(rows), "updates": update, "epochs": self.cfg.epochs,
                 "train_loss_first": round(sum(losses[:window]) / window, 4) if losses else None,
                 "train_loss_last": round(sum(losses[-window:]) / window, 4) if losses else None,
                 "eval_loss_before": eval_before, "eval_loss_after": eval_after,
                 "adapter_dir": str(adapter_dir), "elapsed_seconds": round(time.time() - started, 1),
                 "device": self.device, "config": self.cfg.to_dict()}
        (self.cfg.output_dir / "train_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
        self._emit("train_done", **{k: v for k, v in stats.items() if k != "config"})
        return stats

    def train_dpo(self, dpo_path: Path) -> dict | None:
        """Preference training on (rejected, chosen) pairs. Needs the optional `trl` package."""
        try:
            from trl import DPOConfig, DPOTrainer  # type: ignore
        except ImportError:
            self._emit("dpo_skipped", reason="pip install trl to enable preference training")
            return None
        from datasets import Dataset

        pairs = load_jsonl(dpo_path)
        if not pairs:
            return None
        if self.model is None:
            self.load()
        rows = [{"prompt": format_chat(self.tokenizer, [{"role": "system", "content": p["system"]},
                                                       {"role": "user", "content": p["prompt"]}], True),
                 "chosen": p["chosen"], "rejected": p["rejected"]} for p in pairs]
        out = self.cfg.output_dir / "dpo"
        args = DPOConfig(output_dir=str(out), per_device_train_batch_size=self.cfg.batch_size,
                         gradient_accumulation_steps=self.cfg.grad_accum, num_train_epochs=1,
                         learning_rate=self.cfg.learning_rate / 4, logging_steps=self.cfg.log_every,
                         max_length=self.cfg.max_seq_len, report_to=[], save_strategy="no", beta=0.1)
        trainer = DPOTrainer(model=self.model, ref_model=None, args=args, train_dataset=Dataset.from_list(rows),
                             processing_class=self.tokenizer)
        result = trainer.train()
        self.model.save_pretrained(str(self.cfg.output_dir / "adapter"))
        return {"pairs": len(rows), "train_loss": getattr(result, "training_loss", None)}

    # -- export ----------------------------------------------------------------
    def merge_and_save(self) -> Path:
        merged_dir = self.cfg.output_dir / "merged"
        merged = self.model.merge_and_unload()
        merged.save_pretrained(str(merged_dir), safe_serialization=True)
        self.tokenizer.save_pretrained(str(merged_dir))
        self._emit("merged", path=str(merged_dir))
        return merged_dir

    def export_ollama(self, name: str, merged_dir: Path | None = None, *, system_prompt: str | None = None) -> dict:
        merged_dir = Path(merged_dir or (self.cfg.output_dir / "merged"))
        modelfile = self.cfg.output_dir / "Modelfile"
        lines = [f"FROM {merged_dir.resolve()}", "PARAMETER temperature 0.2", "PARAMETER num_ctx 8192"]
        if system_prompt:
            lines.append(f'SYSTEM """{system_prompt}"""')
        modelfile.write_text("\n".join(lines) + "\n", encoding="utf-8")
        info = {"modelfile": str(modelfile), "ollama_model": name, "created": False,
                "instructions": f"ollama create {name} -f {modelfile}   # then: omniresearch ask --model {name} ..."}
        if shutil.which("ollama"):
            try:
                proc = subprocess.run(["ollama", "create", name, "-f", str(modelfile)], capture_output=True, text=True,
                                      timeout=3600)
                info["created"] = proc.returncode == 0
                info["output"] = (proc.stdout + proc.stderr)[-800:]
            except (subprocess.TimeoutExpired, OSError) as exc:
                info["output"] = str(exc)
        if not info["created"]:
            info["gguf_hint"] = ("If `ollama create` cannot import this architecture directly, convert with llama.cpp: "
                                 f"python convert_hf_to_gguf.py {merged_dir} --outfile student.gguf && "
                                 f"sed -i 's|^FROM .*|FROM ./student.gguf|' {modelfile}")
        self._emit("exported", **info)
        return info
