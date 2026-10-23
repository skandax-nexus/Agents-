"""Energy-based voice activity detection: split a recording into utterances (turns)."""
from __future__ import annotations

import numpy as np


def rms_energy(x: np.ndarray, frame: int, hop: int) -> np.ndarray:
    if len(x) < frame:
        return np.array([np.sqrt(np.mean(x ** 2) + 1e-12)], dtype=np.float32)
    n = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    frames = x[idx]
    return np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12).astype(np.float32)


def split_on_silence(x: np.ndarray, sr: int, *, min_silence: float = 0.4, min_utterance: float = 0.3,
                     threshold_ratio: float = 0.08, pad: float = 0.05) -> list[tuple[int, int]]:
    """Return (start, end) sample spans of speech separated by silences of >= min_silence seconds.

    The threshold adapts to the recording: a fraction of the median-to-max energy range, so quiet
    rooms and loud rooms both work.
    """
    frame, hop = int(0.025 * sr), int(0.010 * sr)
    e = rms_energy(x, frame, hop)
    if len(e) == 0:
        return []
    floor, peak = float(np.percentile(e, 10)), float(np.percentile(e, 95))
    thr = floor + threshold_ratio * (peak - floor)
    active = e > thr
    spans: list[tuple[int, int]] = []
    min_sil_frames = max(1, int(min_silence * sr / hop))
    start = None
    silence_run = 0
    for i, a in enumerate(active):
        if a:
            if start is None:
                start = i
            silence_run = 0
        elif start is not None:
            silence_run += 1
            if silence_run >= min_sil_frames:
                spans.append((start, i - silence_run + 1))
                start, silence_run = None, 0
    if start is not None:
        spans.append((start, len(active)))
    out = []
    pad_s = int(pad * sr)
    for s, e_ in spans:
        raw_a, raw_b = s * hop, (e_ - 1) * hop + frame          # unpadded speech span
        if (raw_b - raw_a) / sr < min_utterance:
            continue
        out.append((max(0, raw_a - pad_s), min(len(x), raw_b + pad_s)))
    return out
