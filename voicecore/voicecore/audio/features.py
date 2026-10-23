"""Spectral features (torch) used for codec losses and the unsupervised language/voice clustering."""
from __future__ import annotations

import math

import torch


def hz_to_mel(f: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + f / 700.0)


def mel_to_hz(m: torch.Tensor) -> torch.Tensor:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float = 20.0, fmax: float | None = None) -> torch.Tensor:
    fmax = fmax or sr / 2
    n_bins = n_fft // 2 + 1
    freqs = torch.linspace(0, sr / 2, n_bins)
    mels = torch.linspace(hz_to_mel(torch.tensor(fmin)), hz_to_mel(torch.tensor(fmax)), n_mels + 2)
    edges = mel_to_hz(mels)
    fb = torch.zeros(n_mels, n_bins)
    for i in range(n_mels):
        lo, mid, hi = edges[i], edges[i + 1], edges[i + 2]
        up = (freqs - lo) / max(float(mid - lo), 1e-6)
        down = (hi - freqs) / max(float(hi - mid), 1e-6)
        fb[i] = torch.clamp(torch.minimum(up, down), min=0.0)
    return fb


def stft_mag(x: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    """|STFT| of (B, T) -> (B, n_fft//2+1, frames)."""
    window = torch.hann_window(n_fft, device=x.device)
    spec = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, return_complex=True,
                      center=True, pad_mode="reflect")
    return spec.abs()


_FB_CACHE: dict[tuple, torch.Tensor] = {}


def log_mel(x: torch.Tensor, sr: int, n_fft: int = 1024, hop: int = 256, n_mels: int = 64) -> torch.Tensor:
    key = (sr, n_fft, n_mels, str(x.device))
    if key not in _FB_CACHE:
        _FB_CACHE[key] = mel_filterbank(sr, n_fft, n_mels).to(x.device)
    mag = stft_mag(x, n_fft, hop)
    mel = torch.matmul(_FB_CACHE[key], mag)
    return torch.log(mel + 1e-5)


def snr_db(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    n = min(reference.shape[-1], estimate.shape[-1])
    ref, est = reference[..., :n], estimate[..., :n]
    noise = ref - est
    return float(10 * math.log10((ref.pow(2).mean() + 1e-9) / (noise.pow(2).mean() + 1e-9)))
