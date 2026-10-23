from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class CodecConfig:
    sample_rate: int = 16000
    channels: int = 32                 # base conv width (x2 at each downsampling stage)
    strides: tuple[int, ...] = (2, 4, 5, 8)   # product = frame hop (320 samples = 20 ms at 16 kHz)
    latent_dim: int = 128
    n_codebooks: int = 4               # residual VQ depth
    codebook_size: int = 1024
    ema_decay: float = 0.99
    commitment: float = 0.25
    dead_code_threshold: int = 2       # usage count below which a code is re-seeded

    @property
    def hop(self) -> int:
        h = 1
        for s in self.strides:
            h *= s
        return h

    @property
    def frame_rate(self) -> float:
        return self.sample_rate / self.hop


@dataclass
class LMConfig:
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 4
    d_ff: int = 1024
    max_frames: int = 1500             # context in codec frames (30 s at 50 fps)
    dropout: float = 0.1
    style_dim: int = 64                # global voice/language/prosody embedding


@dataclass
class TrainConfig:
    batch_size: int = 8
    segment_seconds: float = 2.0
    lr_codec: float = 3e-4
    lr_lm: float = 3e-4
    grad_clip: float = 1.0
    device: str = "auto"
    seed: int = 0


@dataclass
class SelfTrainConfig:
    codec_steps_per_round: int = 50
    lm_steps_per_round: int = 100
    dialogue_steps_per_round: int = 50
    self_play_samples: int = 8         # heard utterances to practise on per round
    candidates_per_sample: int = 4     # responses generated per utterance; best kept
    keep_fraction: float = 0.5
    min_buffer_seconds: float = 30.0   # do nothing until this much audio has been heard
    codec_snr_target_db: float = 12.0  # curriculum: codec must reach this before LM training starts


@dataclass
class VoiceCoreConfig:
    codec: CodecConfig = field(default_factory=CodecConfig)
    lm: LMConfig = field(default_factory=LMConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    selftrain: SelfTrainConfig = field(default_factory=SelfTrainConfig)
    max_response_seconds: float = 8.0
    temperature: float = 0.9
    top_k: int = 64

    # -- presets --------------------------------------------------------------
    @classmethod
    def tiny(cls) -> "VoiceCoreConfig":
        """Runs in seconds on CPU: for tests and the synthetic demo."""
        return cls(codec=CodecConfig(channels=8, latent_dim=32, n_codebooks=2, codebook_size=64),
                   lm=LMConfig(d_model=64, n_layers=2, n_heads=2, d_ff=128, max_frames=256, style_dim=16),
                   train=TrainConfig(batch_size=4, segment_seconds=1.0))

    @classmethod
    def small(cls) -> "VoiceCoreConfig":
        """A laptop-trainable size for real recordings (hours of audio, a GPU helps)."""
        return cls(codec=CodecConfig(channels=32, latent_dim=128, n_codebooks=4, codebook_size=1024),
                   lm=LMConfig(d_model=384, n_layers=8, n_heads=6, d_ff=1536))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "VoiceCoreConfig":
        codec = CodecConfig(**{**d.get("codec", {}), "strides": tuple(d.get("codec", {}).get("strides", (2, 4, 5, 8)))})
        return cls(codec=codec, lm=LMConfig(**d.get("lm", {})), train=TrainConfig(**d.get("train", {})),
                   selftrain=SelfTrainConfig(**d.get("selftrain", {})),
                   max_response_seconds=d.get("max_response_seconds", 8.0), temperature=d.get("temperature", 0.9),
                   top_k=d.get("top_k", 64))

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def load(cls, path: Path) -> "VoiceCoreConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))
