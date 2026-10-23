from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import SpeechLM


@torch.no_grad()
def generate(model: SpeechLM, prompt: torch.Tensor, style: torch.Tensor, *, max_new_frames: int,
             temperature: float = 0.9, top_k: int = 64, stop_at_eos: bool = True,
             chunk_callback=None, chunk_frames: int = 25) -> torch.Tensor:
    """Autoregressively continue `prompt` (B, n_q, F). Returns only the new frames (B, n_q, F_new).

    chunk_callback(new_frames_tensor) is called every `chunk_frames` frames so audio can be streamed.
    For batched prompts, rows that hit EOS early emit code 0 until the whole batch stops; their real
    lengths are available afterwards in ``generate.last_lengths``.
    """
    model.eval()
    sp = model.special
    seq = prompt
    produced = []
    since_cb = 0
    B = prompt.shape[0]
    done = torch.zeros(B, dtype=torch.bool, device=prompt.device)
    lengths = torch.zeros(B, dtype=torch.long, device=prompt.device)
    for _ in range(max_new_frames):
        if seq.shape[-1] >= model.cfg.max_frames:
            seq = seq[:, :, -model.cfg.max_frames + 1:]
        logits = model(seq, style)[:, :, -1, :]                       # (B, n_q, V)
        logits[..., sp.pad] = -float("inf")
        logits[..., sp.bos] = -float("inf")
        logits[..., sp.sep] = -float("inf")
        if temperature <= 0:
            nxt = logits.argmax(dim=-1)
        else:
            logits = logits / temperature
            if top_k and top_k < logits.shape[-1]:
                kth = torch.topk(logits, top_k, dim=-1).values[..., -1:]
                logits = logits.masked_fill(logits < kth, -float("inf"))
            probs = F.softmax(logits, dim=-1)
            B, n_q, V = probs.shape
            nxt = torch.multinomial(probs.view(-1, V), 1).view(B, n_q)
        if stop_at_eos:
            done = done | (nxt[:, 0] == sp.eos)
            if bool(done.all()):
                break
        nxt = nxt.masked_fill(nxt >= sp.base, 0)   # a stray special id in a lower codebook: treat as code 0
        nxt = nxt.masked_fill(done.unsqueeze(1), 0)   # finished rows emit silence code until the batch stops
        lengths += (~done).long()
        produced.append(nxt.unsqueeze(-1))
        seq = torch.cat([seq, nxt.unsqueeze(-1)], dim=-1)
        since_cb += 1
        if chunk_callback is not None and since_cb >= chunk_frames:
            chunk_callback(torch.cat(produced[-since_cb:], dim=-1))
            since_cb = 0
    if chunk_callback is not None and since_cb:
        chunk_callback(torch.cat(produced[-since_cb:], dim=-1))
    if not produced:
        out = prompt[:, :, :0]
    else:
        out = torch.cat(produced, dim=-1)
    generate.last_lengths = lengths.tolist()   # per-row number of real frames (rows may finish early)
    return out
