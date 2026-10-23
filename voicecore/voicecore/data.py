"""Audio corpora, segment sampling, and a synthetic 'proto-speech' generator for tests/demos."""
from __future__ import annotations

import math
import random
import sys
import wave
from pathlib import Path

import numpy as np
import torch

from .audio.io import load_audio

AUDIO_SUFFIXES = {".wav"}


def list_audio(paths: list[str | Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        p = Path(p).expanduser()
        if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES:
            out.append(p)
        elif p.is_dir():
            out += sorted(q for q in p.rglob("*") if q.suffix.lower() in AUDIO_SUFFIXES)
    return out


class AudioCorpus:
    """Keeps decoded mono audio in memory (fine for hours of 16 kHz speech) and samples segments."""

    def __init__(self, sr: int) -> None:
        self.sr = sr
        self.clips: list[np.ndarray] = []
        self.names: list[str] = []

    def add_array(self, x: np.ndarray, name: str = "") -> None:
        x = np.asarray(x, dtype=np.float32)
        if len(x) > 0:
            self.clips.append(x)
            self.names.append(name)

    def add_files(self, paths: list[str | Path], *, verbose: bool = True) -> int:
        n = 0
        for p in list_audio(paths):
            try:
                self.add_array(load_audio(p, self.sr), str(p))
                n += 1
            except (OSError, ValueError, wave.Error, EOFError) as exc:
                if verbose:
                    print(f"  skipped {p}: {exc}", file=sys.stderr)
        return n

    @property
    def seconds(self) -> float:
        return sum(len(c) for c in self.clips) / self.sr

    def sample_batch(self, batch: int, seconds: float, rng: random.Random) -> torch.Tensor:
        n = int(seconds * self.sr)
        if not self.clips:
            raise ValueError("corpus is empty")
        weights = [len(c) for c in self.clips]
        out = np.zeros((batch, n), dtype=np.float32)
        for i in range(batch):
            clip = rng.choices(self.clips, weights=weights, k=1)[0]
            if len(clip) > n:
                s = rng.randrange(0, len(clip) - n)
                out[i] = clip[s : s + n]
                continue
            # short clip: fill the segment with further random clips instead of digital silence
            pos = 0
            while pos < n:
                take = min(len(clip), n - pos)
                out[i, pos : pos + take] = clip[:take]
                pos += take
                clip = rng.choices(self.clips, weights=weights, k=1)[0]
        return torch.from_numpy(out)


# ---------------------------------------------------------------- synthetic proto-speech
def proto_speech(seconds: float, sr: int = 16000, *, seed: int = 0, n_phones: int = 12, speaker: float = 1.0,
                 language: int = 0) -> np.ndarray:
    """A deterministic, speech-like signal with no real language: a sequence of 'phones' (harmonic
    vowels with distinct formants, noise-burst consonants, pauses) drawn from a 'language' with its own
    phone inventory and transition rules.  Good enough to show that the codec learns to encode it and
    that the LM learns its structure - entirely from scratch."""
    rng = np.random.default_rng(seed * 1000 + language)
    f0_base = 110.0 * speaker
    # each language has its own formant inventory and bigram preferences
    formants = rng.uniform(300, 2800, size=(n_phones, 2))
    trans = rng.dirichlet(np.ones(n_phones) * 0.3, size=n_phones)
    total = int(seconds * sr)
    out = np.zeros(total, dtype=np.float32)
    t_cursor = 0
    phone = int(rng.integers(n_phones))
    while t_cursor < total:
        kind = rng.random()
        dur = int(sr * rng.uniform(0.06, 0.18))
        n = min(dur, total - t_cursor)
        t = np.arange(n) / sr
        if kind < 0.15:                                   # pause
            seg = np.zeros(n, dtype=np.float32)
        elif kind < 0.35:                                 # consonant: shaped noise burst
            noise = rng.standard_normal(n).astype(np.float32)
            env = np.exp(-t * 25)
            seg = noise * env * 0.3
        else:                                             # vowel-like: harmonic series with formant envelope
            f0 = f0_base * (1 + 0.1 * math.sin(2 * math.pi * 3 * t_cursor / sr))
            seg = np.zeros(n, dtype=np.float32)
            for h in range(1, 20):
                fh = f0 * h
                if fh > sr / 2:
                    break
                amp = sum(math.exp(-((fh - F) / 250.0) ** 2) for F in formants[phone]) + 0.05
                seg += (amp / h) * np.sin(2 * math.pi * fh * t + rng.uniform(0, 2 * math.pi))
            seg *= np.hanning(n).astype(np.float32) * 0.4
        out[t_cursor : t_cursor + n] = seg[:n]
        t_cursor += n
        phone = int(rng.choice(n_phones, p=trans[phone]))
    peak = np.abs(out).max() or 1.0
    return (out / peak * 0.8).astype(np.float32)
