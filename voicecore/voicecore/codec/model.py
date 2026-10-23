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


def _wn(module: nn.Module, enabled: bool) -> nn.Module:
    return nn.utils.parametrizations.weight_norm(module) if enabled else module


class ResBlock(nn.Module):
    def __init__(self, ch: int, dilation: int, wn: bool = True) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ELU(), _wn(nn.Conv1d(ch, ch, 3, dilation=dilation, padding=dilation), wn),
            nn.ELU(), _wn(nn.Conv1d(ch, ch, 1), wn),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class Encoder(nn.Module):
    def __init__(self, cfg: CodecConfig) -> None:
        super().__init__()
        ch = cfg.channels
        wn = cfg.weight_norm
        layers: list[nn.Module] = [_wn(nn.Conv1d(1, ch, 7, padding=3), wn)]
        for stride in cfg.strides:
            layers += [ResBlock(ch, 1, wn), ResBlock(ch, 3, wn), nn.ELU(),
                       _wn(nn.Conv1d(ch, ch * 2, 2 * stride, stride=stride, padding=stride // 2 + (stride % 2)), wn)]
            ch *= 2
        layers += [nn.ELU(), _wn(nn.Conv1d(ch, cfg.latent_dim, 3, padding=1), wn)]
        self.net = nn.Sequential(*layers)
        self.out_channels = ch

    def forward(self, wav: torch.Tensor) -> torch.Tensor:   # (B, T) -> (B, D, frames)
        return self.net(wav.unsqueeze(1))


class Decoder(nn.Module):
    def __init__(self, cfg: CodecConfig, in_ch: int) -> None:
        super().__init__()
        ch = in_ch
        wn = cfg.weight_norm
        layers: list[nn.Module] = [_wn(nn.Conv1d(cfg.latent_dim, ch, 3, padding=1), wn)]
        for stride in reversed(cfg.strides):
            pad = stride // 2 + (stride % 2)
            layers += [nn.ELU(), _wn(nn.ConvTranspose1d(ch, ch // 2, 2 * stride, stride=stride, padding=pad,
                                                        output_padding=2 * pad - stride), wn),
                       ResBlock(ch // 2, 1, wn), ResBlock(ch // 2, 3, wn)]
            ch //= 2
        layers += [nn.ELU(), _wn(nn.Conv1d(ch, 1, 7, padding=3), wn)]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:    # (B, D, frames) -> (B, T)
        return torch.tanh(self.net(z)).squeeze(1)


class VectorQuantizer(nn.Module):
    """One EMA-updated codebook (VQ-VAE v2 style) with two anti-collapse measures:
    optional l2-normalised (cosine) matching, and re-seeding of codes unused for `dead_steps` steps."""

    def __init__(self, dim: int, size: int, decay: float, dead_steps: int, cosine: bool) -> None:
        super().__init__()
        self.dim, self.size, self.decay, self.dead_steps, self.cosine = dim, size, decay, dead_steps, cosine
        embed = torch.randn(size, dim)
        embed = F.normalize(embed, dim=1) if cosine else embed * 0.1
        self.register_buffer("embed", embed)
        self.register_buffer("ema_count", torch.ones(size))
        self.register_buffer("ema_sum", embed.clone())
        self.register_buffer("last_used", torch.zeros(size, dtype=torch.long))
        self.register_buffer("step", torch.zeros((), dtype=torch.long))
        self.register_buffer("initialised", torch.zeros((), dtype=torch.bool))

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=1) if self.cosine else x

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (N, dim) -> (quantised (N, dim) with straight-through grad, codes (N,))."""
        xn = self._prep(x)
        flat = xn.detach()
        if self.training and not bool(self.initialised):
            # Data-dependent init, decided once at the very first training forward: seed as many codes as the
            # batch has frames from real latents (the rest keep their random init and are re-seeded by the
            # dead-code rule).  Never re-run later: that would wipe a trained codebook and remap every token.
            n_seed = min(self.size, flat.shape[0])
            if n_seed > 0:
                idx = torch.randperm(flat.shape[0], device=flat.device)[:n_seed]
                self.embed[:n_seed] = flat[idx]
                self.ema_sum[:n_seed] = flat[idx]
                self.ema_count.fill_(1.0)
            self.initialised.fill_(True)
        dist = flat.pow(2).sum(1, keepdim=True) - 2 * flat @ self.embed.t() + self.embed.pow(2).sum(1)[None, :]
        codes = dist.argmin(dim=1)
        quant = self.embed[codes]
        if self.training:
            self.step += 1
            onehot = F.one_hot(codes, self.size).float()
            counts = onehot.sum(0)
            self.ema_count.mul_(self.decay).add_(counts, alpha=1 - self.decay)
            self.ema_sum.mul_(self.decay).add_(onehot.t() @ flat, alpha=1 - self.decay)
            n = self.ema_count.sum()
            smoothed = (self.ema_count + 1e-5) / (n + self.size * 1e-5) * n
            new_embed = self.ema_sum / smoothed.unsqueeze(1)
            self.embed.copy_(F.normalize(new_embed, dim=1) if self.cosine else new_embed)
            self.last_used[counts > 0] = self.step
            dead = (self.step - self.last_used) >= self.dead_steps
            if dead.any() and flat.shape[0] > 0:
                replacement = flat[torch.randint(0, flat.shape[0], (int(dead.sum()),), device=flat.device)]
                self.embed[dead] = replacement
                self.ema_sum[dead] = replacement
                self.ema_count[dead] = 1.0
                self.last_used[dead] = self.step
        return quant, codes

    def lookup(self, codes: torch.Tensor) -> torch.Tensor:
        return self.embed[codes]

    @property
    def usage(self) -> torch.Tensor:
        """Steps since each code was last selected (0 = used in the latest step)."""
        return (self.step - self.last_used).float()


class ResidualVQ(nn.Module):
    def __init__(self, cfg: CodecConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(VectorQuantizer(cfg.latent_dim, cfg.codebook_size, cfg.ema_decay,
                                                    cfg.dead_code_steps, cfg.vq_cosine) for _ in range(cfg.n_codebooks))
        self.commitment = cfg.commitment
        self.center = cfg.vq_center
        self.register_buffer("running_mean", torch.zeros(cfg.latent_dim))
        self.register_buffer("mean_steps", torch.zeros((), dtype=torch.long))

    def _centered(self, x: torch.Tensor) -> torch.Tensor:
        """Remove the shared offset of the latents (EMA mean) so codes describe deviations, not the DC level.
        Early in training every frame's latent is almost the same vector; without centering they all map to
        one code and the codebook stalls."""
        if not self.center:
            return x
        if self.training:
            batch_mean = x.detach().mean(0)
            momentum = 0.1 if int(self.mean_steps) > 0 else 1.0
            self.running_mean.mul_(1 - momentum).add_(batch_mean, alpha=momentum)
            self.mean_steps += 1
        return x - self.running_mean

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """z: (B, D, F) -> (quantised (B, D, F), codes (B, n_q, F), commitment loss).

        A straight-through path is kept per codebook, so the encoder's reconstruction gradient is scaled
        by n_codebooks.  This is deliberate: measured on this architecture, a single straight-through
        path (gradient x1) leaves the codec stuck in its early few-codes plateau far longer."""
        B, D, Fr = z.shape
        x = self._centered(z.permute(0, 2, 1).reshape(-1, D))
        residual = x
        out = torch.zeros_like(x)
        all_codes = []
        commit = torch.zeros((), device=z.device)
        for vq in self.layers:
            q, codes = vq(residual)
            xn = vq._prep(residual)
            commit = commit + F.mse_loss(xn, q.detach())
            out = out + xn + (q - xn).detach()
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

    class _inference:
        """Context manager: no grad and eval mode (so EMA codebooks are never touched), restoring the mode after."""

        def __init__(self, module: nn.Module) -> None:
            self.module = module

        def __enter__(self):
            self.was_training = self.module.training
            self.module.eval()
            self.ctx = torch.no_grad()
            self.ctx.__enter__()
            return self

        def __exit__(self, *exc):
            self.ctx.__exit__(*exc)
            self.module.train(self.was_training)
            return False

    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        """wav (B, T) -> codes (B, n_q, F).  Never updates the codebooks."""
        with self._inference(self):
            _, codes, _ = self.rvq(self.encoder(self.pad(wav)))
            return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """codes (B, n_q, F) -> wav (B, F * hop)."""
        with self._inference(self):
            return self.decoder(self.rvq.decode_codes(codes))

    def embed(self, wav: torch.Tensor) -> torch.Tensor:
        """Mean-pooled continuous latent (B, D): conditions the speech LM on voice/language/prosody."""
        with self._inference(self):
            return self.encoder(self.pad(wav)).mean(dim=-1)

    def fingerprint(self, wav: torch.Tensor) -> torch.Tensor:
        """Mean and spread of the latent over time (B, 2D)."""
        with self._inference(self):
            z = self.encoder(self.pad(wav))
            return torch.cat([z.mean(dim=-1), z.std(dim=-1, unbiased=False)], dim=1)
