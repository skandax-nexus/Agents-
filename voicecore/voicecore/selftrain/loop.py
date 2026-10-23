"""Self-training: the engine improves from what it hears, what it says, and how that is received.

Round structure (``SelfTrainer.round()``):
  1. codec update     - self-supervised reconstruction on everything heard (any language)
  2. LM update        - next-frame prediction on everything heard: learns the structure of the speech around it
  3. self-play        - for recent utterances, generate several candidate replies, score each with the model's
                        own judgement (reply likelihood + codec round-trip consistency), keep the best as
                        practice exchanges
  4. dialogue update  - train on kept self-play exchanges, real exchanges, and user-approved replies
                        (positive feedback up-weighted, negative excluded)
  5. voice clusters   - report the unsupervised language/voice map (adapted as audio is heard)

A curriculum gate keeps the LM from training on garbage: the codec must reach a reconstruction SNR
target first.  All of it is offline and needs nothing pretrained.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from ..codec.train import CodecTrainer
from ..config import VoiceCoreConfig
from ..data import AudioCorpus
from ..lm.train import LMTrainer
from .buffer import ExperienceBuffer


class SelfTrainer:
    def __init__(self, engine, *, on_event=None) -> None:
        from ..engine import VoiceCore  # noqa: F401 - type only

        self.engine = engine
        self.cfg: VoiceCoreConfig = engine.cfg
        self.sc = self.cfg.selftrain
        self.buffer: ExperienceBuffer = engine.buffer
        self.on_event = on_event or (lambda k, d: None)
        self.codec_trainer = CodecTrainer(engine.codec, self.cfg, on_event=self.on_event)
        self.lm_trainer = LMTrainer(engine.lm, engine.codec, self.cfg, on_event=self.on_event)
        self.state_path = engine.root / "selftrain_state.json"
        self.state = {"rounds": 0, "stage": "codec", "codec_snr": None, "lm_loss": None, "dialogue_loss": None}
        if self.state_path.exists():
            try:
                self.state.update(json.loads(self.state_path.read_text()))
            except (OSError, ValueError):
                pass

    def _emit(self, kind: str, **data) -> None:
        try:
            self.on_event(kind, data)
        except Exception:  # noqa: BLE001
            pass

    # -- scoring a candidate reply with the model's own judgement ------------------------------
    def score_reply(self, heard: np.ndarray, reply: np.ndarray) -> float:
        if len(reply) < self.engine.codec.hop * 5:
            return -1e9
        lp = self.lm_trainer.exchange_logprob(heard, reply)
        # codec round-trip consistency: a reply the codec itself re-encodes to the same codes is 'clean'
        dev = self.lm_trainer.device
        r = torch.as_tensor(reply, dtype=torch.float32, device=dev).unsqueeze(0)
        codes = self.engine.codec.encode(r)
        again = self.engine.codec.encode(self.engine.codec.decode(codes))
        n = min(codes.shape[-1], again.shape[-1])
        consistency = float((codes[..., :n] == again[..., :n]).float().mean()) if n else 0.0
        # penalise silence / near-silence replies
        energy = float(np.sqrt(np.mean(reply ** 2) + 1e-12))
        energy_term = -2.0 if energy < 0.01 else 0.0
        return lp + consistency + energy_term

    # -- one round ----------------------------------------------------------------------------------
    def round(self, *, corpus_extra: AudioCorpus | None = None) -> dict:
        started = time.time()
        sr = self.cfg.codec.sample_rate
        report = {"round": self.state["rounds"] + 1, "stage_before": self.state["stage"]}
        corpus = AudioCorpus(sr)
        for clip in self.buffer.heard_clips():
            corpus.add_array(clip)
        if corpus_extra is not None:
            for clip, name in zip(corpus_extra.clips, corpus_extra.names):
                corpus.add_array(clip, name)
        report["corpus_seconds"] = round(corpus.seconds, 1)
        if corpus.seconds < self.sc.min_buffer_seconds:
            report["note"] = f"need {self.sc.min_buffer_seconds}s of heard audio, have {corpus.seconds:.1f}s"
            self._emit("selftrain_skipped", **report)
            return report

        # 1. codec
        self.codec_trainer.train(corpus, self.sc.codec_steps_per_round, seed=self.state["rounds"])
        self.engine.mark_codec_trained(self.sc.codec_steps_per_round)
        quality = self.codec_trainer.evaluate(corpus)
        snr = quality["snr_db"]
        self.state["codec_snr"] = snr
        self.state["codec_mel_distance"] = quality["mel_distance"]
        report["codec_snr_db"] = snr
        report["codec_mel_distance"] = quality["mel_distance"]
        if snr < self.sc.codec_snr_target_db and self.state["stage"] == "codec":
            self.state["rounds"] += 1
            report["note"] = f"codec stage: SNR {snr:.1f} dB < target {self.sc.codec_snr_target_db} dB"
            return self._finish(report, started)
        if self.state["stage"] == "codec":
            self.state["stage"] = "lm"

        # 2. LM on everything heard
        loss = self.lm_trainer.train_continuation(corpus, self.sc.lm_steps_per_round, seed=self.state["rounds"])
        self.state["lm_loss"] = round(loss, 4)
        report["lm_loss"] = self.state["lm_loss"]

        # 3. self-play
        rng = random.Random(self.state["rounds"])
        clips = self.buffer.heard_clips(limit=50)
        rng.shuffle(clips)
        kept = 0
        for heard in clips[: self.sc.self_play_samples]:
            cands = []
            for _ in range(self.sc.candidates_per_sample):
                reply = self.engine.respond(heard, sr, learn=False)
                cands.append((self.score_reply(heard, reply), reply))
            cands.sort(key=lambda t: t[0], reverse=True)
            n_keep = max(1, int(round(len(cands) * self.sc.keep_fraction)))
            for s, reply in cands[:n_keep]:
                if s > -1e8 and len(reply) >= self.engine.codec.hop:
                    self.buffer.add_exchange(heard, reply, score=s, origin="self-play")
                    kept += 1
        report["self_play_kept"] = kept

        # 4. dialogue update on exchanges (real + self-play + approved)
        rows = self.buffer.exchanges()
        if rows:
            self.state["stage"] = "dialogue"
            d_loss = float("nan")
            for _ in range(self.sc.dialogue_steps_per_round):
                batch_rows = rng.sample(rows, k=min(self.cfg.train.batch_size, len(rows)))
                pairs, weights = [], []
                for r in batch_rows:
                    h, s = self.buffer.load_exchange(r)
                    if len(h) < self.engine.codec.hop or len(s) < self.engine.codec.hop:
                        continue
                    pairs.append((h[: int(self.cfg.max_response_seconds * sr)], s[: int(self.cfg.max_response_seconds * sr)]))
                    fb = r.get("feedback") or 0
                    weights.append(3.0 if fb > 0 else (1.0 if r.get("origin") == "live" else 0.7))
                if not pairs:
                    continue
                d_loss = self.lm_trainer.step_exchanges(pairs, weights)
            self.state["dialogue_loss"] = round(d_loss, 4)
            report["dialogue_loss"] = self.state["dialogue_loss"]
            report["exchanges_used"] = len(rows)

        # 5. voice/language map: re-cluster all stored fingerprints (keeps the current number of clusters)
        if len(self.engine.clusters.fingerprints) >= 4:
            self.engine.clusters.refit()
        report["voice_clusters"] = len(self.engine.clusters.counts)
        self.state["rounds"] += 1
        return self._finish(report, started)

    def _finish(self, report: dict, started: float) -> dict:
        report["stage"] = self.state["stage"]
        report["elapsed"] = round(time.time() - started, 1)
        self.state_path.write_text(json.dumps(self.state, indent=1))
        self.engine.save()
        self._emit("selftrain_round", **report)
        return report
