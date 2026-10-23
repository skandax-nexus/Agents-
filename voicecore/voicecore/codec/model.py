"""Neural audio codec: waveform -> discrete tokens -> waveform.  Trained from scratch.

Encoder: strided 1-D convolutions (SoundStream/EnCodec-style, no pretrained weights).
Quantiser: residual vector quantisation with EMA codebooks and dead-code re-seeding.
Decoder: mirror of the encoder with transposed convolutions.
The tokens it produces are what the speech language model speaks in.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import CodecConfig


class ResBlock(nn.Module):
    def __init__(self, ch: int, dilation: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ELU(), nn.Conv1d(ch, ch, 3, dilation=dilation, padding=dilation),
            nn.ELU(), nn.Conv1d(ch, ch, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class Encoder(nn.Module):
    def __init__(self, cfg: CodecConfig) -> None:
        super().__init__()
        ch = cfg.channels
        layers: list[nn.Module] = [nn.Conv1d(1, ch, 7, padding=3)]
        for stride in cfg.strides:
            layers += [ResBlock(ch, 1), ResBlock(ch, 3), nn.ELU(),
                       nn.Conv1d(ch, ch * 2, 2 * stride, stride=stride, padding=stride // 2 + (stride % 2))]
            ch *= 2
        layers += [nn.ELU(), nn.Conv1d(ch, cfg.latent_dim, 3, padding=1)]
        self.net = nn.Sequential(*layers)
        self.out_channels = ch

    def forward(self, wav: torch.Tensor) -> torch.Tensor:   # (B, T) -> (B, D, frames)
        return self.net(wav.unsqueeze(1))


class Decoder(nn.Module):
    def __init__(self, cfg: CodecConfig, in_ch: int) -> None:
        super().__init__()
        ch = in_ch
        layers: list[nn.Module] = [nn.Conv1d(cfg.latent_dim, ch, 3, padding=1)]
        for stride in reversed(cfg.strides):
            pad = stride // 2 + (stride % 2)
            layers += [nn.ELU(), nn.ConvTranspose1d(ch, ch // 2, 2 * stride, stride=stride, padding=pad,
                                                    output_padding=2 * pad - stride),
                       ResBlock(ch // 2, 1), ResBlock(ch // 2, 3)]
            ch //= 2
        layers += [nn.ELU(), nn.Conv1d(ch, 1, 7, padding=3)]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:    # (B, D, frames) -> (B, T)
        return torch.tanh(self.net(z)).squeeze(1)


class VectorQuantizer(nn.Module):
    """One EMA-updated codebook (VQ-VAE v2 style)."""

    def __init__(self, dim: int, size: int, decay: float, dead_threshold: int) -> None:
        super().__init__()
        self.dim, self.size, self.decay, self.dead_threshold = dim, size, decay, dead_threshold
        embed = torch.randn(size, dim) * 0.1
        self.register_buffer("embed", embed)
        self.register_buffer("ema_count", torch.ones(size))
        self.register_buffer("ema_sum", embed.clone())
        self.register_buffer("usage", torch.zeros(size))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (N, dim) -> (quantised (N, dim) with straight-through grad, codes (N,))."""
        flat = x.detach()
        dist = flat.pow(2).sum(1, keepdim=True) - 2 * flat @ self.embed.t() + self.embed.pow(2).sum(1)[None, :]
        codes = dist.argmin(dim=1)
        quant = self.embed[codes]
        if self.training:
            onehot = F.one_hot(codes, self.size).float()
            counts = onehot.sum(0)
            self.ema_count.mul_(self.decay).add_(counts, alpha=1 - self.decay)
            self.ema_sum.mul_(self.decay).add_(onehot.t() @ flat, alpha=1 - self.decay)
            n = self.ema_count.sum()
            smoothed = (self.ema_count + 1e-5) / (n + self.size * 1e-5) * n
            self.embed.copy_(self.ema_sum / smoothed.unsqueeze(1))
            self.usage.mul_(self.decay).add_(counts, alpha=1 - self.decay)
            dead = self.usage < (self.dead_threshold * (1 - self.decay))
            if dead.any() and flat.shape[0] > 0:
                replacement = flat[torch.randint(0, flat.shape[0], (int(dead.sum()),), device=flat.device)]
                self.embed[dead] = replacement
                self.ema_sum[dead] = replacement
                self.ema_count[dead] = 1.0
                self.usage[dead] = 1.0
        quant_st = x + (quant - x).detach()
        return quant_st, codes

    def lookup(self, codes: torch.Tensor) -> torch.Tensor:
        return self.embed[codes]


class ResidualVQ(nn.Module):
    def __init__(self, cfg: CodecConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(VectorQuantizer(cfg.latent_dim, cfg.codebook_size, cfg.ema_decay,
                                                    cfg.dead_code_threshold) for _ in range(cfg.n_codebooks))
        self.commitment = cfg.commitment

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """z: (B, D, F) -> (quantised (B, D, F), codes (B, n_q, F), commitment loss)."""
        B, D, Fr = z.shape
        x = z.permute(0, 2, 1).reshape(-1, D)
        residual = x
        out = torch.zeros_like(x)
        all_codes = []
        commit = torch.zeros((), device=z.device)
        for vq in self.layers:
            q, codes = vq(residual)
            commit = commit + F.mse_loss(residual, q.detach())
            out = out + q
            residual = residual - q.detach()
            all_codes.append(codes.view(B, Fr))
        quant = out.view(B, Fr, D).permute(0, 2, 1)
        return quant, torch.stack(all_codes, dim=1), self.commitment * commit

    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """codes: (B, n_q, F) -> latent (B, D, F)."""
        B, n_q, Fr = codes.shape
        out = None
        for q, vq in enumerate(self.layers[:n_q]):
            emb = vq.lookup(codes[:, q].reshape(-1)).view(B, Fr, -1)
            out = emb if out is None else out + emb
        return out.permute(0, 2, 1)


class AudioCodec(nn.Module):
    def __init__(self, cfg: CodecConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.rvq = ResidualVQ(cfg)
        self.decoder = Decoder(cfg, self.encoder.out_channels)

    @property
    def hop(self) -> int:
        return self.cfg.hop

    def pad(self, wav: torch.Tensor) -> torch.Tensor:
        rem = wav.shape[-1] % self.hop
        return F.pad(wav, (0, self.hop - rem)) if rem else wav

    def forward(self, wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """wav (B, T) -> (reconstruction (B, T'), codes (B, n_q, F), commitment loss)."""
        wav = self.pad(wav)
        z = self.encoder(wav)
        quant, codes, commit = self.rvq(z)
        recon = self.decoder(quant)
        return recon[..., : wav.shape[-1]], codes, commit

    @torch.no_grad()
    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        """wav (B, T) -> codes (B, n_q, F)."""
        _, codes, _ = self.rvq(self.encoder(self.pad(wav)))
        return codes

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """codes (B, n_q, F) -> wav (B, F * hop)."""
        return self.decoder(self.rvq.decode_codes(codes))

    @torch.no_grad()
    def embed(self, wav: torch.Tensor) -> torch.Tensor:
        """Mean-pooled continuous latent (B, D): conditions the speech LM on voice/language/prosody."""
        return self.encoder(self.pad(wav)).mean(dim=-1)

    @torch.no_grad()
    def fingerprint(self, wav: torch.Tensor) -> torch.Tensor:
        """Mean and spread of the latent over time (B, 2D): richer than `embed`, used for voice/language clustering."""
        z = self.encoder(self.pad(wav))
        return torch.cat([z.mean(dim=-1), z.std(dim=-1)], dim=1)
