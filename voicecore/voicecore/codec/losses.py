"""Codec training loss.

Waveform-dominant: the encoder/decoder frames are time-aligned, so a plain MSE on the waveform
is a strong, well-conditioned signal from step one.  A small spectral term (linear-magnitude L1,
scale-free, plus clamped log-mel L1) sharpens timbre.  Measured on this architecture: a large
log-spectral weight drives early training toward near-silent output and codebook collapse, so
the spectral weight is deliberately small and its log terms are clamped (bounded gradients).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..audio.features import mel_filterbank, stft_mag

_FB: dict[tuple, torch.Tensor] = {}


def safe_log_mel(x: torch.Tensor, sr: int, n_fft: int, hop: int, n_mels: int, eps: float = 1e-3) -> torch.Tensor:
    key = (sr, n_fft, n_mels, str(x.device))
    if key not in _FB:
        _FB[key] = mel_filterbank(sr, n_fft, n_mels).to(x.device)
    return torch.log(torch.clamp(_FB[key] @ stft_mag(x, n_fft, hop), min=eps))


def multi_scale_spectral_loss(ref: torch.Tensor, est: torch.Tensor, sr: int,
                              scales: tuple[int, ...] = (256, 512, 1024)) -> torch.Tensor:
    loss = torch.zeros((), device=ref.device)
    n = 0
    for n_fft in scales:
        if ref.shape[-1] < n_fft:
            continue
        hop = n_fft // 4
        m_ref, m_est = stft_mag(ref, n_fft, hop), stft_mag(est, n_fft, hop)
        loss = loss + F.l1_loss(m_est, m_ref) / (m_ref.abs().mean() + 1e-6)
        n_mels = min(64, n_fft // 8)
        loss = loss + F.l1_loss(safe_log_mel(est, sr, n_fft, hop, n_mels), safe_log_mel(ref, sr, n_fft, hop, n_mels))
        n += 1
    return loss / max(n, 1)


def mel_distance(ref: torch.Tensor, est: torch.Tensor, sr: int) -> float:
    """Log-mel L1 distance: the codec's main quality metric early in training (phase-free)."""
    n = min(ref.shape[-1], est.shape[-1])
    with torch.no_grad():
        return float(F.l1_loss(safe_log_mel(est[..., :n], sr, 1024, 256, 64), safe_log_mel(ref[..., :n], sr, 1024, 256, 64)))


def codec_loss(ref: torch.Tensor, est: torch.Tensor, commit: torch.Tensor, sr: int, *, mse_weight: float = 100.0,
               spec_weight: float = 0.1) -> tuple[torch.Tensor, dict]:
    n = min(ref.shape[-1], est.shape[-1])
    ref, est = ref[..., :n], est[..., :n]
    wave = F.mse_loss(est, ref)
    spec = multi_scale_spectral_loss(ref, est, sr)
    total = mse_weight * wave + spec_weight * spec + commit
    return total, {"wave": wave.item(), "spec": spec.item(), "commit": commit.item(), "total": total.item()}
