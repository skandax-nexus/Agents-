"""Training the speech LM on (a) continuous audio and (b) heard -> reply exchanges. No text anywhere."""
from __future__ import annotations

import random
import time

import torch

from ..codec.model import AudioCodec
from ..codec.train import pick_device
from ..config import VoiceCoreConfig
from ..data import AudioCorpus
from .model import SpeechLM


def tokenize(codec: AudioCodec, wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """wav (B, T) -> (codes (B, n_q, F), style (B, D)) with the codec frozen."""
    codec.eval()
    with torch.no_grad():
        codes = codec.encode(wav)
        style = codec.embed(wav)
    return codes, style


def build_exchange(lm: SpeechLM, heard: torch.Tensor, reply: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[BOS] heard [SEP] reply [EOS] -> (codes (1, n_q, F), loss mask selecting SEP..EOS targets)."""
    sp = lm.special
    n_q = heard.shape[1]
    dev = heard.device

    def special(tok: int) -> torch.Tensor:
        return torch.full((1, n_q, 1), tok, dtype=torch.long, device=dev)

    seq = torch.cat([special(sp.bos), heard, special(sp.sep), reply, special(sp.eos)], dim=-1)
    mask = torch.zeros(1, seq.shape[-1], dtype=torch.bool, device=dev)
    mask[:, heard.shape[-1] + 2:] = True        # learn to produce the reply frames and the EOS
    return seq, mask


def pad_batch(lm: SpeechLM, items: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    sp = lm.special
    n_q = items[0][0].shape[1]
    L = min(max(s.shape[-1] for s, _ in items), lm.cfg.max_frames)
    codes = torch.full((len(items), n_q, L), sp.pad, dtype=torch.long, device=items[0][0].device)
    mask = torch.zeros(len(items), L, dtype=torch.bool, device=codes.device)
    for i, (s, m) in enumerate(items):
        n = min(s.shape[-1], L)
        codes[i, :, :n] = s[0, :, -n:] if s.shape[-1] > L else s[0, :, :n]
        mask[i, :n] = m[0, -n:] if m.shape[-1] > L else m[0, :n]
    return codes, mask


class LMTrainer:
    def __init__(self, lm: SpeechLM, codec: AudioCodec, cfg: VoiceCoreConfig, *, on_event=None) -> None:
        self.lm, self.codec, self.cfg = lm, codec, cfg
        self.device = pick_device(cfg.train.device)
        self.lm.to(self.device)
        self.codec.to(self.device)
        self.opt = torch.optim.AdamW(self.lm.parameters(), lr=cfg.train.lr_lm, betas=(0.9, 0.95), weight_decay=0.01)
        self.on_event = on_event or (lambda k, d: None)
        self.step_no = 0

    def _update(self, loss: torch.Tensor) -> float:
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.lm.parameters(), self.cfg.train.grad_clip)
        self.opt.step()
        self.step_no += 1
        return loss.item()

    # -- continuous audio: learn the structure of sound ---------------------------------
    def step_continuation(self, wav: torch.Tensor) -> float:
        self.lm.train()
        codes, style = tokenize(self.codec, wav.to(self.device))
        sp = self.lm.special
        bos = torch.full((codes.shape[0], codes.shape[1], 1), sp.bos, dtype=torch.long, device=self.device)
        seq = torch.cat([bos, codes], dim=-1)[:, :, : self.lm.cfg.max_frames]
        return self._update(self.lm.loss(seq, style))

    def train_continuation(self, corpus: AudioCorpus, steps: int, *, log_every: int = 25, seed: int | None = None) -> float:
        rng = random.Random(self.cfg.train.seed if seed is None else seed)
        started = time.time()
        loss = float("nan")
        for i in range(steps):
            batch = corpus.sample_batch(self.cfg.train.batch_size, self.cfg.train.segment_seconds, rng)
            loss = self.step_continuation(batch)
            if (i + 1) % log_every == 0 or i + 1 == steps:
                self.on_event("lm_step", {"step": self.step_no, "loss": round(loss, 4), "elapsed": round(time.time() - started, 1)})
        return loss

    # -- exchanges: learn to reply -----------------------------------------------------------
    def step_exchanges(self, pairs: list[tuple[torch.Tensor, torch.Tensor]], weights: list[float] | None = None) -> float:
        """pairs of (heard wav (T,), reply wav (T',)) numpy/tensors at the codec sample rate."""
        self.lm.train()
        items = []
        styles = []
        kept_weights = []
        hop = self.codec.hop
        for i, (heard, reply) in enumerate(pairs):
            if len(reply) < hop or len(heard) < hop:
                continue   # nothing to learn from an empty side
            h = torch.as_tensor(heard, dtype=torch.float32, device=self.device).unsqueeze(0)
            r = torch.as_tensor(reply, dtype=torch.float32, device=self.device).unsqueeze(0)
            hc, style = tokenize(self.codec, h)
            rc, _ = tokenize(self.codec, r)
            hc = self._fit_heard(hc, rc.shape[-1])
            items.append(build_exchange(self.lm, hc, rc))
            styles.append(style)
            if weights:
                kept_weights.append(weights[i])
        if not items:
            return float("nan")
        weights = kept_weights if weights else None
        codes, mask = pad_batch(self.lm, items)
        style = torch.cat(styles, dim=0)
        if weights:
            w = torch.tensor(weights, dtype=torch.float32, device=self.device).view(-1, 1)
            mask_f = mask.float() * w
            logits = self.lm(codes, style)[:, :, :-1]
            B, n_q, Fr, V = logits.shape
            per = torch.nn.functional.cross_entropy(logits.reshape(-1, V), codes.reshape(-1), reduction="none").view(B, n_q, Fr)
            m = mask_f.unsqueeze(1).expand_as(per)
            loss = (per * m).sum() / m.sum().clamp(min=1.0)
        else:
            loss = self.lm.loss(codes, style, loss_mask=mask)
        return self._update(loss)

    def _fit_heard(self, hc: torch.Tensor, reply_frames: int) -> torch.Tensor:
        """Keep the tail of the heard tokens so [BOS] heard [SEP] reply [EOS] fits in max_frames."""
        room = self.lm.cfg.max_frames - reply_frames - 3
        if room < 1:
            room = 1
        return hc[:, :, -room:] if hc.shape[-1] > room else hc

    @torch.no_grad()
    def exchange_logprob(self, heard: torch.Tensor, reply: torch.Tensor) -> float:
        """How plausible the model finds `reply` as an answer to `heard` (mean log-prob per reply code).

        Scores exactly the span training supervises (reply frames and EOS), keeping the tail of a long
        heard utterance so the reply is always inside the context."""
        self.lm.eval()
        hop = self.codec.hop
        if len(reply) < hop or len(heard) < hop:
            return -1e9
        h = torch.as_tensor(heard, dtype=torch.float32, device=self.device).unsqueeze(0)
        r = torch.as_tensor(reply, dtype=torch.float32, device=self.device).unsqueeze(0)
        hc, style = tokenize(self.codec, h)
        rc, _ = tokenize(self.codec, r)
        max_reply = self.lm.cfg.max_frames - 4
        if rc.shape[-1] > max_reply:
            rc = rc[:, :, :max_reply]
        hc = self._fit_heard(hc, rc.shape[-1])
        seq, _ = build_exchange(self.lm, hc, rc)
        return float(self.lm.sequence_logprob(seq, style, from_frame=hc.shape[-1] + 2)[0])
