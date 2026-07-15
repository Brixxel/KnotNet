"""
KnotGraphNet V5 — Sequence Prediction Model.

Identische Architektur zum Training.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .image_encoder import ImageEncoder


def fourier_encode(x: torch.Tensor, num_bands: int = 8) -> torch.Tensor:
    """Fourier feature encoding für Koordinaten."""
    freqs = 2.0 ** torch.arange(num_bands, device=x.device).float() * np.pi
    x_proj = x.unsqueeze(-1) * freqs
    return torch.cat([x_proj.sin(), x_proj.cos()], dim=-1).flatten(-2)


def compute_internal_partners(
    token_cid: torch.Tensor,
    token_pair: torch.Tensor,
    token_mask: torch.Tensor,
) -> torch.Tensor:
    """Findet für jeden Token den 'Partner' im gleichen Crossing."""
    B, N = token_cid.shape
    partner = torch.full((B, N), -1, dtype=torch.long, device=token_cid.device)
    for b in range(B):
        for i in range(N):
            if not token_mask[b, i] or token_cid[b, i] < 0:
                continue
            ci = token_cid[b, i].item()
            pi = token_pair[b, i].item()
            partner_role = pi ^ 1  # 0↔1, 2↔3
            if partner_role >= 4:
                continue
            for j in range(N):
                if j == i or not token_mask[b, j]:
                    continue
                if token_cid[b, j] == ci and token_pair[b, j] == partner_role:
                    partner[b, i] = j
                    break
    return partner


def compute_skeleton_bias_batch(
    skel: torch.Tensor,
    token_xy: torch.Tensor,
    token_mask: torch.Tensor,
    n_samples: int = 15,
) -> torch.Tensor:
    """Sampelt das Skelett entlang der Linie zwischen Token-Paaren."""
    B, _, H, W = skel.shape
    N = token_xy.shape[1]
    device = skel.device

    t = torch.linspace(0, 1, n_samples, device=device)
    a_exp = token_xy.unsqueeze(2)
    b_exp = token_xy.unsqueeze(1)
    t_exp = t.view(1, 1, 1, -1)

    pts = a_exp.unsqueeze(3) + t_exp.unsqueeze(-1) * (
        b_exp.unsqueeze(3) - a_exp.unsqueeze(3)
    )

    xi = (pts[..., 0] * (W - 1)).long().clamp(0, W - 1)
    yi = (pts[..., 1] * (H - 1)).long().clamp(0, H - 1)

    skel_flat = skel[:, 0]
    xi_flat = xi.view(B * N * N, n_samples)
    yi_flat = yi.view(B * N * N, n_samples)
    skel_flat_exp = skel_flat.repeat_interleave(N * N, dim=0)

    values = skel_flat_exp[
        torch.arange(B * N * N, device=device).unsqueeze(-1), yi_flat, xi_flat
    ]
    values = values.view(B, N, N, n_samples)
    bias = (values > 0.3).float().mean(dim=-1)

    valid_mask = token_mask.unsqueeze(2) & token_mask.unsqueeze(1)
    return bias * valid_mask.float()


class KnotGraphNet(nn.Module):
    """KnotGraphNet V5 — exakte Kopie der Trainings-Architektur."""

    def __init__(
        self,
        d: int = 64,
        num_heads: int = 2,
        num_layers: int = 2,
        max_tokens: int = 40,
        num_fourier: int = 8,
        dropout: float = 0.3,
        skel_bias_weight: float = 2.0,
        max_neighbor_dist: float = 0.85,
        soft_dist_penalty: float = -8.0,
    ):
        super().__init__()
        self.d = d
        self.max_tokens = max_tokens
        self.num_fourier = num_fourier
        self.max_neighbor_dist = max_neighbor_dist
        self.soft_dist_penalty = soft_dist_penalty

        self.image_encoder = ImageEncoder(in_ch=4, out_dim=d)

        fourier_dim = 2 * 2 * num_fourier
        self.token_proj = nn.Linear(fourier_dim, d)
        self.partner_proj = nn.Linear(fourier_dim, d)
        self.type_emb = nn.Embedding(2, d)
        self.pair_emb = nn.Embedding(6, d)
        self.has_partner_emb = nn.Embedding(2, d)
        self.token_norm = nn.LayerNorm(d)
        self.token_dropout = nn.Dropout(dropout)

        self.cross_attn = nn.MultiheadAttention(
            d, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(d)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=num_heads,
            dim_feedforward=2 * d,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.self_attn = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)

        self.skel_bias_scale = nn.Parameter(torch.tensor(skel_bias_weight))

    def forward(
        self,
        img: torch.Tensor,
        skel: torch.Tensor,
        token_xy: torch.Tensor,
        token_type: torch.Tensor,
        token_cid: torch.Tensor,
        token_pair: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, N = token_mask.shape

        img_feat = self.image_encoder(img, skel).flatten(2).transpose(1, 2)

        partners = compute_internal_partners(token_cid, token_pair, token_mask)
        has_partner = (partners >= 0).long()
        p_safe = partners.clamp(min=0)
        partner_xy = torch.gather(
            token_xy, 1, p_safe.unsqueeze(-1).expand(-1, -1, 2)
        )
        no_p = (partners < 0).unsqueeze(-1).float()
        partner_xy = partner_xy * (1 - no_p) + token_xy * no_p
        delta = token_xy - partner_xy

        tok = self.token_proj(fourier_encode(token_xy, self.num_fourier))
        tok = tok + self.type_emb(token_type)
        tok = tok + self.pair_emb(token_pair)
        tok = tok + self.partner_proj(fourier_encode(delta, self.num_fourier))
        tok = tok + self.has_partner_emb(has_partner)
        tok = self.token_dropout(self.token_norm(tok))

        tok2, _ = self.cross_attn(tok, img_feat, img_feat)
        tok = self.cross_norm(tok + tok2)
        tok = self.self_attn(tok, src_key_padding_mask=~token_mask)

        q = self.q_proj(tok)
        k = self.k_proj(tok)
        logits = torch.einsum("bnd,bmd->bnm", q, k) / (self.d**0.5)

        # Direction alignment
        delta_n = F.normalize(delta, dim=-1, eps=1e-6)
        to_j = F.normalize(
            token_xy.unsqueeze(1) - token_xy.unsqueeze(2), dim=-1, eps=1e-6
        )
        dir_alignment = (delta_n.unsqueeze(2) * to_j).sum(dim=-1)
        logits = logits + dir_alignment * 2.0 * (partners >= 0).float().unsqueeze(-1)

        # Skeleton bias
        skel_bias = compute_skeleton_bias_batch(skel, token_xy, token_mask)
        logits = logits + skel_bias * self.skel_bias_scale

        # Masks
        eye = torch.eye(N, device=logits.device, dtype=torch.bool).unsqueeze(0)
        logits = logits.masked_fill(eye, float("-inf"))
        logits = logits.masked_fill(
            (~token_mask).unsqueeze(1).expand(-1, N, -1), float("-inf")
        )

        same_cx = (token_cid.unsqueeze(2) == token_cid.unsqueeze(1)) & (
            token_cid.unsqueeze(2) >= 0
        )
        logits = logits.masked_fill(same_cx, float("-inf"))

        far = (torch.cdist(token_xy, token_xy) > self.max_neighbor_dist).float()
        logits = logits + far * self.soft_dist_penalty

        return logits