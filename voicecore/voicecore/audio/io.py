"""WAV in/out and resampling with only the standard library and numpy."""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Return (float32 samples in [-1, 1] shaped (n,) or (n, ch), sample_rate)."""
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        width = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported sample width {width}")
    if ch > 1:
        data = data.reshape(-1, ch)
    return data, sr


def write_wav(path: str | Path, samples: np.ndarray, sr: int) -> None:
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 2:
        ch = samples.shape[1]
    else:
        ch = 1
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(ch)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def to_mono(samples: np.ndarray) -> np.ndarray:
    return samples.mean(axis=1) if samples.ndim == 2 else samples


def resample(samples: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Linear-interpolation resampler (adequate for speech; swap for polyphase if needed)."""
    if sr_in == sr_out:
        return samples.astype(np.float32)
    samples = to_mono(samples)
    n_out = int(round(len(samples) * sr_out / sr_in))
    if n_out <= 1:
        return np.zeros(max(n_out, 0), dtype=np.float32)
    if sr_out < sr_in:   # crude anti-alias: moving average over the decimation ratio
        k = max(1, int(sr_in / sr_out))
        kernel = np.ones(k, dtype=np.float32) / k
        samples = np.convolve(samples, kernel, mode="same")
    x_old = np.linspace(0.0, 1.0, num=len(samples), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, samples).astype(np.float32)


def load_audio(path: str | Path, sr: int) -> np.ndarray:
    data, native = read_wav(path)
    return resample(to_mono(data), native, sr)
