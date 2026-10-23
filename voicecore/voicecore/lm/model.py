"""Speech language model: a decoder-only transformer over codec tokens.

It never sees text.  A frame is the n_q residual codes the codec produced for 20 ms of
sound; the model embeds all of them (summed), attends causally over frames, and predicts
the n_q codes of the next frame with one head per codebook.  A learned *style* prefix
(from the codec's pooled latent of the input utterance) carries the speaker, language and
prosody, so a reply is produced in the language and voice it was spoken to in.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import CodecConfig, LMConfig


@dataclass(frozen=True)
class SpecialTokens:
    """Codebook ids above the codec vocabulary, shared by every codebook."""
    base: int

    @property
    def bos(self) -> int:
        return self.base

    @property
    def sep(self) -> int:   # end of the heard utterance / start of the reply
        return self.base + 1

    @property
    def eos(self) -> int:
        return self.base + 2

    @property
    def pad(self) -> int:
        return self.base + 3

    @property
    def vocab(self) -> int:
        return self.base + 4


class Block(nn.Module):
    def __init__(self, cfg: LMConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(), nn.Linear(cfg.d_ff, cfg.d_model),
                                 nn.Dropout(cfg.dropout))

    def forward(self, x: torch.Tensor, mask: torch.Tensor, key_padding: torch.Tensor | None) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, key_padding_mask=key_padding, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class SpeechLM(nn.Module):
    def __init__(self, cfg: LMConfig, codec_cfg: CodecConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_q = codec_cfg.n_codebooks
        self.special = SpecialTokens(codec_cfg.codebook_size)
        V = self.special.vocab
        self.tok_embed = nn.ModuleList(nn.Embedding(V, cfg.d_model) for _ in range(self.n_q))
        self.pos_embed = nn.Embedding(cfg.max_frames + 1, cfg.d_model)
        self.style_proj = nn.Sequential(nn.Linear(codec_cfg.latent_dim, cfg.style_dim), nn.Tanh(),
                                        nn.Linear(cfg.style_dim, cfg.d_model))
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.heads = nn.ModuleList(nn.Linear(cfg.d_model, V) for _ in range(self.n_q))
        self.drop = nn.Dropout(cfg.dropout)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, codes: torch.Tensor, style: torch.Tensor, key_padding: torch.Tensor | None = None) -> torch.Tensor:
        """codes (B, n_q, F), style (B, latent_dim) -> logits (B, n_q, F, V) predicting frame t+1 at position t.

        The style vector is prepended as frame 0, so the output at position 0 predicts the first real frame.
        """
        B, n_q, Fr = codes.shape
        if Fr + 1 > self.cfg.max_frames + 1:
            raise ValueError(f"sequence of {Fr} frames exceeds max_frames={self.cfg.max_frames}")
        x = sum(self.tok_embed[q](codes[:, q]) for q in range(n_q))      # (B, F, d)
        x = torch.cat([self.style_proj(style).unsqueeze(1), x], dim=1)      # (B, F+1, d)
        pos = torch.arange(Fr + 1, device=codes.device)
        x = self.drop(x + self.pos_embed(pos)[None])
        L = Fr + 1
        mask = torch.triu(torch.ones(L, L, device=codes.device, dtype=torch.bool), diagonal=1)
        if key_padding is not None:
            key_padding = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=codes.device), key_padding], dim=1)
        for blk in self.blocks:
            x = blk(x, mask, key_padding)
        x = self.ln_f(x)
        return torch.stack([head(x) for head in self.heads], dim=1)        # (B, n_q, F+1, V)

    def loss(self, codes: torch.Tensor, style: torch.Tensor, loss_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Next-frame prediction over every codebook. loss_mask (B, F) selects which target frames count."""
        logits = self(codes, style, key_padding=None)[:, :, :-1]            # position t predicts frame t
        B, n_q, Fr, V = logits.shape
        targets = codes
        if loss_mask is None:
            loss_mask = targets[:, 0] != self.special.pad
        per = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), reduction="none").view(B, n_q, Fr)
        m = loss_mask.unsqueeze(1).float().expand_as(per)
        return (per * m).sum() / m.sum().clamp(min=1.0)

    @torch.no_grad()
    def sequence_logprob(self, codes: torch.Tensor, style: torch.Tensor, from_frame: int = 0) -> torch.Tensor:
        """Mean per-frame log-probability of codes[from_frame:] under the model (B,)."""
        logits = self(codes, style)[:, :, :-1]
        logp = F.log_softmax(logits, dim=-1).gather(-1, codes.unsqueeze(-1)).squeeze(-1)   # (B, n_q, F)
        logp = logp[:, :, from_frame:].mean(dim=1)                                           # (B, F')
        return logp.mean(dim=1)
