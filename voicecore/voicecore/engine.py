"""The voice-to-voice engine.

    hear(wav)     -> remember it (and its voice/language cluster)
    respond(wav)  -> codec tokens -> speech LM reply tokens -> codec waveform    (no STT, no TTS)
    stream        -> replies are decoded chunk by chunk as they are generated

Every exchange is stored so the SelfTrainer can learn from it.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import torch

from .audio.features import log_mel
from .audio.io import load_audio, resample, write_wav
from .codec.model import AudioCodec
from .codec.train import pick_device
from .config import VoiceCoreConfig
from .lang.cluster import VoiceClusters
from .lm.generate import generate
from .lm.model import SpeechLM
from .selftrain.buffer import ExperienceBuffer


FINGERPRINT_MELS = 40
FINGERPRINT_DIM = 3 * FINGERPRINT_MELS


class VoiceCore:
    def __init__(self, root: str | Path, cfg: VoiceCoreConfig | None = None) -> None:
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        cfg_path = self.root / "config.json"
        if cfg is None:
            cfg = VoiceCoreConfig.load(cfg_path) if cfg_path.exists() else VoiceCoreConfig()
        self.cfg = cfg
        cfg.save(cfg_path)
        self.device = pick_device(cfg.train.device)
        torch.manual_seed(cfg.train.seed)
        self.codec = AudioCodec(cfg.codec).to(self.device)
        self.lm = SpeechLM(cfg.lm, cfg.codec).to(self.device)
        self.clusters = VoiceClusters(FINGERPRINT_DIM)
        self.buffer = ExperienceBuffer(self.root / "experience", cfg.codec.sample_rate)
        self.meta = {"created": time.time(), "codec_steps": 0, "lm_steps": 0, "exchanges": 0}
        self._load_weights()

    # -- persistence ------------------------------------------------------------------------------
    def _load_weights(self) -> None:
        cp = self.root / "codec.pt"
        lp = self.root / "lm.pt"
        if cp.exists():
            self.codec.load_state_dict(torch.load(cp, map_location=self.device)["state"])
        if lp.exists():
            self.lm.load_state_dict(torch.load(lp, map_location=self.device)["state"])
        vc = self.root / "clusters.json"
        if vc.exists():
            self.clusters = VoiceClusters.load(vc)
        mp = self.root / "meta.json"
        if mp.exists():
            try:
                self.meta.update(json.loads(mp.read_text()))
            except (OSError, ValueError):
                pass

    def save(self) -> None:
        torch.save({"state": self.codec.state_dict()}, self.root / "codec.pt")
        torch.save({"state": self.lm.state_dict()}, self.root / "lm.pt")
        self.clusters.save(self.root / "clusters.json")
        (self.root / "meta.json").write_text(json.dumps(self.meta, indent=1))

    @property
    def sr(self) -> int:
        return self.cfg.codec.sample_rate

    # -- perception --------------------------------------------------------------------------------
    def _tensor(self, wav: np.ndarray, sr: int | None = None) -> torch.Tensor:
        wav = np.asarray(wav, dtype=np.float32)
        if sr is not None and sr != self.sr:
            wav = resample(wav, sr, self.sr)
        return torch.from_numpy(wav).to(self.device).unsqueeze(0)

    def fingerprint(self, wav: np.ndarray, sr: int | None = None) -> np.ndarray:
        """Acoustic fingerprint of an utterance: level-normalised spectral shape, per-band spread and per-band
        flux over the voiced frames (pitch, timbre, sound inventory, articulation rate).  Pure signal
        processing, so voice/language discovery works before anything is trained."""
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
        """Store an utterance for learning and report which voice/language cluster it belongs to."""
        wav = resample(np.asarray(wav, dtype=np.float32), sr, self.sr) if sr and sr != self.sr else np.asarray(wav, dtype=np.float32)
        cluster = self.clusters.observe(self.fingerprint(wav))
        row = self.buffer.add_heard(wav, cluster=cluster, source=source)
        return {"id": row["id"], "cluster": cluster, "cluster_name": self.clusters.label(cluster),
                "seconds": row["seconds"]}

    # -- speaking ----------------------------------------------------------------------------------
    def respond(self, wav: np.ndarray, sr: int | None = None, *, learn: bool = True,
                on_chunk: Callable[[np.ndarray], None] | None = None, max_seconds: float | None = None) -> np.ndarray:
        """Reply in audio to audio. Returns the reply waveform at the engine's sample rate."""
        self.codec.eval()
        self.lm.eval()
        wav_t = self._tensor(wav, sr)
        heard_np = wav_t[0].cpu().numpy()
        codes = self.codec.encode(wav_t)
        style = self.codec.embed(wav_t)
        sp = self.lm.special
        n_q = codes.shape[1]
        bos = torch.full((1, n_q, 1), sp.bos, dtype=torch.long, device=self.device)
        sep = torch.full((1, n_q, 1), sp.sep, dtype=torch.long, device=self.device)
        prompt = torch.cat([bos, codes, sep], dim=-1)
        if prompt.shape[-1] >= self.lm.cfg.max_frames - 2:
            prompt = torch.cat([bos, codes[:, :, -(self.lm.cfg.max_frames - 4):], sep], dim=-1)
        max_frames = int((max_seconds or self.cfg.max_response_seconds) * self.cfg.codec.frame_rate)

        def cb(frames: torch.Tensor) -> None:
            if on_chunk is not None:
                on_chunk(self.codec.decode(frames)[0].cpu().numpy())

        new = generate(self.lm, prompt, style, max_new_frames=max_frames, temperature=self.cfg.temperature,
                       top_k=self.cfg.top_k, chunk_callback=cb if on_chunk else None)
        reply = self.codec.decode(new)[0].cpu().numpy() if new.shape[-1] else np.zeros(0, dtype=np.float32)
        if learn:
            cluster = self.clusters.observe(self.fingerprint(heard_np))
            self.buffer.add_exchange(heard_np, reply, cluster=cluster, origin="live")
            self.meta["exchanges"] = self.meta.get("exchanges", 0) + 1
        return reply

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
