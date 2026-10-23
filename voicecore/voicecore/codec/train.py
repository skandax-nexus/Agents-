from __future__ import annotations

import random
import time
from pathlib import Path

import torch

from ..audio.features import snr_db
from ..config import VoiceCoreConfig
from ..data import AudioCorpus
from .losses import codec_loss, mel_distance
from .model import AudioCodec


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


class CodecTrainer:
    def __init__(self, codec: AudioCodec, cfg: VoiceCoreConfig, *, on_event=None) -> None:
        self.codec = codec
        self.cfg = cfg
        self.device = pick_device(cfg.train.device)
        self.codec.to(self.device)
        self.opt = torch.optim.AdamW(self.codec.parameters(), lr=cfg.train.lr_codec, betas=(0.8, 0.99))
        self.on_event = on_event or (lambda k, d: None)
        self.step_no = 0

    def step(self, batch: torch.Tensor) -> dict:
        self.codec.train()
        batch = batch.to(self.device)
        recon, _, commit = self.codec(batch)
        loss, parts = codec_loss(batch, recon, commit, self.cfg.codec.sample_rate)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.codec.parameters(), self.cfg.train.grad_clip)
        self.opt.step()
        self.step_no += 1
        return parts

    def train(self, corpus: AudioCorpus, steps: int, *, log_every: int = 25, seed: int | None = None) -> dict:
        rng = random.Random(self.cfg.train.seed if seed is None else seed)
        started = time.time()
        last = {}
        for i in range(steps):
            batch = corpus.sample_batch(self.cfg.train.batch_size, self.cfg.train.segment_seconds, rng)
            last = self.step(batch)
            if (i + 1) % log_every == 0 or i + 1 == steps:
                self.on_event("codec_step", {"step": self.step_no, **last, "elapsed": round(time.time() - started, 1)})
        return last

    @torch.no_grad()
    def evaluate(self, corpus: AudioCorpus, n: int = 8, seed: int = 123) -> dict:
        """Reconstruction quality on a fixed sample: SNR (phase-aligned), log-mel distance, codebook usage."""
        self.codec.eval()
        rng = random.Random(seed)
        batch = corpus.sample_batch(n, self.cfg.train.segment_seconds, rng).to(self.device)
        recon, codes, _ = self.codec(batch)
        used = [int(torch.unique(codes[:, q]).numel()) for q in range(codes.shape[1])]
        return {"snr_db": round(snr_db(batch, recon), 2), "mel_distance": round(mel_distance(batch, recon, self.cfg.codec.sample_rate), 3),
                "codes_used": used, "codebook_size": self.cfg.codec.codebook_size}

    def evaluate_snr(self, corpus: AudioCorpus, n: int = 8, seed: int = 123) -> float:
        return self.evaluate(corpus, n, seed)["snr_db"]

    def save(self, path: Path) -> None:
        torch.save({"state": self.codec.state_dict(), "step": self.step_no}, path)
