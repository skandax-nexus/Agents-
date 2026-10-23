from __future__ import annotations

import torch
import torch.nn.functional as F

from ..audio.features import log_mel, stft_mag


def multi_scale_spectral_loss(ref: torch.Tensor, est: torch.Tensor, sr: int,
                              scales: tuple[int, ...] = (256, 512, 1024)) -> torch.Tensor:
    loss = torch.zeros((), device=ref.device)
    for n_fft in scales:
        if ref.shape[-1] < n_fft:
            continue
        hop = n_fft // 4
        m_ref, m_est = stft_mag(ref, n_fft, hop), stft_mag(est, n_fft, hop)
        loss = loss + F.l1_loss(torch.log(m_est + 1e-5), torch.log(m_ref + 1e-5))
        loss = loss + (m_ref - m_est).norm() / (m_ref.norm() + 1e-6)   # spectral convergence
        n_mels = min(64, n_fft // 8)
        loss = loss + F.l1_loss(log_mel(est, sr, n_fft, hop, n_mels), log_mel(ref, sr, n_fft, hop, n_mels))
    return loss / len(scales)


def codec_loss(ref: torch.Tensor, est: torch.Tensor, commit: torch.Tensor, sr: int) -> tuple[torch.Tensor, dict]:
    n = min(ref.shape[-1], est.shape[-1])
    ref, est = ref[..., :n], est[..., :n]
    wave = F.l1_loss(est, ref)
    spec = multi_scale_spectral_loss(ref, est, sr)
    total = 10.0 * wave + spec + commit
    return total, {"wave": wave.item(), "spec": spec.item(), "commit": commit.item(), "total": total.item()}
