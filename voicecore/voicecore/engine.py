"""The voice-to-voice engine.

    hear(wav)     -> remember it (and its voice/language cluster)
    respond(wav)  -> codec tokens -> speech LM reply tokens -> codec waveform    (no STT, no TTS)
    stream        -> replies are decoded chunk by chunk as they are generated

Every exchange is stored so the SelfTrainer can learn from it.
"""
from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import torch

from .audio.features import log_mel
from .audio.io import load_audio, resample, to_mono, write_wav
from .codec.model import AudioCodec
from .codec.train import pick_device
from .config import VoiceCoreConfig
from .lang.cluster import VoiceClusters
from .lm.generate import generate
from .lm.model import SpeechLM
from .selftrain.buffer import ExperienceBuffer

FINGERPRINT_MELS = 40
FINGERPRINT_DIM = 3 * FINGERPRINT_MELS            # acoustic (DSP) fingerprint, used before the codec has trained


class VoiceCore:
    def __init__(self, root: str | Path, cfg: VoiceCoreConfig | None = None) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        cfg_path = self.root / "config.json"
        if cfg is None:
            cfg = VoiceCoreConfig.load(cfg_path) if cfg_path.exists() else VoiceCoreConfig()
        self.cfg = cfg
        self.device = pick_device(cfg.train.device)
        torch.manual_seed(cfg.train.seed)
        self.codec = AudioCodec(cfg.codec).to(self.device)
        self.lm = SpeechLM(cfg.lm, cfg.codec).to(self.device)
        self.meta = {"created": time.time(), "codec_steps": 0, "lm_steps": 0, "exchanges": 0}
        self.clusters = VoiceClusters(self.fingerprint_dim)
        self.buffer = ExperienceBuffer(self.root / "experience", cfg.codec.sample_rate)
        self._load_weights()          # raises before anything on disk is touched if cfg and weights disagree
        cfg.save(cfg_path)

    # -- persistence ------------------------------------------------------------------------------
    def mark_codec_trained(self, steps: int) -> None:
        """Record codec training; the first time, switch the cluster space to the codec fingerprint."""
        was = self.codec_trained
        self.meta["codec_steps"] = int(self.meta.get("codec_steps", 0)) + int(steps)
        if not was and self.codec_trained:
            self.clusters = VoiceClusters(self.fingerprint_dim)

    def _load_weights(self) -> None:
        cp = self.root / "codec.pt"
        lp = self.root / "lm.pt"
        if cp.exists():
            self.codec.load_state_dict(torch.load(cp, map_location=self.device)["state"])
        if lp.exists():
            self.lm.load_state_dict(torch.load(lp, map_location=self.device)["state"])
        mp = self.root / "meta.json"
        if mp.exists():
            try:
                self.meta.update(json.loads(mp.read_text()))
            except (OSError, ValueError):
                pass
        vc = self.root / "clusters.json"
        if vc.exists():
            loaded = VoiceClusters.load(vc)
            if loaded.dim == self.fingerprint_dim:
                self.clusters = loaded
            else:
                self.clusters = VoiceClusters(self.fingerprint_dim)   # fingerprint space changed; start over

    @staticmethod
    def _atomic_torch_save(obj, path: Path) -> None:
        """Write to a temporary file and rename, so a crash mid-write never leaves a corrupt checkpoint."""
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(obj, tmp)
        os.replace(tmp, path)

    @staticmethod
    def _atomic_write_text(path: Path, text: str) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)

    def save(self) -> None:
        self._atomic_torch_save({"state": self.codec.state_dict(), "version": 1}, self.root / "codec.pt")
        self._atomic_torch_save({"state": self.lm.state_dict(), "version": 1}, self.root / "lm.pt")
        self.save_light()

    def save_light(self) -> None:
        """Cheap state (clusters, counters) - written after every hear/respond so it never lags the buffer."""
        self.clusters.save(self.root / "clusters.json")
        self._atomic_write_text(self.root / "meta.json", json.dumps(self.meta, indent=1))

    @property
    def sr(self) -> int:
        return self.cfg.codec.sample_rate

    @property
    def codec_trained(self) -> bool:
        return int(self.meta.get("codec_steps", 0)) > 0

    @property
    def fingerprint_dim(self) -> int:
        return 2 * self.cfg.codec.latent_dim if self.codec_trained else FINGERPRINT_DIM

    # -- perception --------------------------------------------------------------------------------
    def _prepare(self, wav: np.ndarray, sr: int | None = None) -> np.ndarray:
        """Mono float32 at the engine sample rate, finite, clipped to [-1, 1]."""
        wav = to_mono(np.asarray(wav, dtype=np.float32))
        if wav.ndim != 1:
            raise ValueError(f"expected a 1-D (mono) or 2-D (channels) array, got shape {wav.shape}")
        if sr is not None and sr != self.sr:
            wav = resample(wav, sr, self.sr)
        wav = np.clip(np.nan_to_num(wav, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
        return np.ascontiguousarray(wav, dtype=np.float32)

    def _tensor(self, wav: np.ndarray, sr: int | None = None) -> torch.Tensor:
        return torch.from_numpy(self._prepare(wav, sr)).to(self.device).unsqueeze(0)

    def fingerprint(self, wav: np.ndarray, sr: int | None = None) -> np.ndarray:
        """Fingerprint of an utterance for voice/language discovery.

        Once the codec has trained, its latent (mean and spread over time) is used: measured on real
        recordings it separates speakers better than any hand-built feature.  Before that, an acoustic
        fingerprint (level-normalised spectral shape, per-band spread and flux over voiced frames) is used
        so discovery works from the first utterance."""
        if self.codec_trained:
            return self.codec.fingerprint(self._tensor(wav, sr))[0].cpu().numpy()
        m = log_mel(self._tensor(wav, sr), self.sr, 1024, 256, FINGERPRINT_MELS)[0]       # (mels, frames)
        level = m.mean(0)
        active = level > level.median()                                                    # drop silent frames
        ma = m[:, active] if int(active.sum()) > 10 else m
        mean = ma.mean(1)
        shape = mean - mean.mean()
        spread = ma.std(1) if ma.shape[1] > 1 else torch.zeros_like(mean)
        flux = (ma[:, 1:] - ma[:, :-1]).abs().mean(1) if ma.shape[1] > 1 else torch.zeros_like(mean)
        return torch.cat([shape, spread, flux]).cpu().numpy()

    def hear(self, wav: np.ndarray, sr: int | None = None, *, source: str = "mic") -> dict:
        """Store an utterance for learning (without replying) and report its voice/language cluster.
        Do not also call respond(learn=True) on the same audio: respond() already hears what it answers."""
        wav = self._prepare(wav, sr)
        cluster = self.clusters.observe(self.fingerprint(wav))
        row = self.buffer.add_heard(wav, cluster=cluster, source=source)
        self.save_light()
        return {"id": row["id"], "cluster": cluster, "cluster_name": self.clusters.label(cluster),
                "seconds": row["seconds"]}

    # -- speaking ----------------------------------------------------------------------------------
    def respond(self, wav: np.ndarray, sr: int | None = None, *, learn: bool = True,
                on_chunk: Callable[[np.ndarray], None] | None = None, max_seconds: float | None = None) -> np.ndarray:
        """Reply in audio to audio. Returns the reply waveform at the engine's sample rate.
        With learn=True the utterance is heard (clustered, stored with the reply) exactly once."""
        return self.converse(wav, sr, learn=learn, on_chunk=on_chunk, max_seconds=max_seconds)["reply"]

    def converse(self, wav: np.ndarray, sr: int | None = None, *, learn: bool = True,
                 on_chunk: Callable[[np.ndarray], None] | None = None, max_seconds: float | None = None) -> dict:
        """Like respond() but also returns the cluster and the stored exchange id."""
        self.codec.eval()
        self.lm.eval()
        wav_t = self._tensor(wav, sr)
        heard_np = wav_t[0].cpu().numpy()
        self.last_cluster = None
        self.last_exchange_id = None
        codes = self.codec.encode(wav_t)
        style = self.codec.embed(wav_t)
        sp = self.lm.special
        n_q = codes.shape[1]
        bos = torch.full((1, n_q, 1), sp.bos, dtype=torch.long, device=self.device)
        sep = torch.full((1, n_q, 1), sp.sep, dtype=torch.long, device=self.device)
        prompt = torch.cat([bos, codes, sep], dim=-1)
        if prompt.shape[-1] >= self.lm.cfg.max_frames - 2:
            prompt = torch.cat([bos, codes[:, :, -(self.lm.cfg.max_frames - 4):], sep], dim=-1)
        limit = self.cfg.max_response_seconds if max_seconds is None else max_seconds
        max_frames = int(limit * self.cfg.codec.frame_rate)

        def cb(frames: torch.Tensor) -> None:
            if on_chunk is not None:
                on_chunk(self.codec.decode(frames)[0].cpu().numpy())

        new = generate(self.lm, prompt, style, max_new_frames=max_frames, temperature=self.cfg.temperature,
                       top_k=self.cfg.top_k, chunk_callback=cb if on_chunk else None)
        reply = self.codec.decode(new)[0].cpu().numpy() if new.shape[-1] else np.zeros(0, dtype=np.float32)
        info = {"reply": reply, "cluster": None, "cluster_name": "unknown", "exchange_id": None,
                "reply_seconds": len(reply) / self.sr}
        if learn:
            cluster = self.clusters.observe(self.fingerprint(heard_np))
            info["cluster"], info["cluster_name"] = cluster, self.clusters.label(cluster)
            if len(reply) >= self.codec.hop:                      # an empty reply teaches nothing
                row = self.buffer.add_exchange(heard_np, reply, cluster=cluster, origin="live")
                info["exchange_id"] = row["id"]
                self.meta["exchanges"] = self.meta.get("exchanges", 0) + 1
            else:
                self.buffer.add_heard(heard_np, cluster=cluster, source="live")
            self.save_light()
        return info

    def respond_file(self, in_path: str | Path, out_path: str | Path, **kw) -> np.ndarray:
        wav = load_audio(in_path, self.sr)
        reply = self.respond(wav, **kw)
        write_wav(out_path, reply, self.sr)
        return reply

    def stream(self, wav: np.ndarray, sr: int | None = None, **kw) -> Iterator[np.ndarray]:
        """Yield reply audio chunk by chunk while it is being generated."""
        chunks: list[np.ndarray] = []
        self.respond(wav, sr, on_chunk=chunks.append, **kw)
        yield from chunks

    def round_trip(self, wav: np.ndarray, sr: int | None = None) -> np.ndarray:
        """Codec only: encode and decode (how well the engine can 'hear' and 're-say' sound)."""
        self.codec.eval()
        t = self._tensor(wav, sr)
        return self.codec.decode(self.codec.encode(t))[0].cpu().numpy()[: t.shape[-1]]

    def feedback(self, exchange_id: str, value: int) -> bool:
        return self.buffer.feedback(exchange_id, value)

    def status(self) -> dict:
        return {"root": str(self.root), "device": self.device, "sample_rate": self.sr,
                "frame_rate": self.cfg.codec.frame_rate, "codec_params": sum(p.numel() for p in self.codec.parameters()),
                "lm_params": sum(p.numel() for p in self.lm.parameters()), "experience": self.buffer.stats(),
                "voice_clusters": self.clusters.summary(), **self.meta}
