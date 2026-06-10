import torch
import torch.nn as nn


class ForwardPredictor(nn.Module):
    """Simple MLP predictor that maps an embedding to the next-step embedding."""
    def __init__(self, embed_dim: int, hidden: int = None):
        super().__init__()
        hidden = hidden or max(embed_dim // 2, 128)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


class BackwardHead(nn.Module):
    """Reconstruction head that maps a terminal embedding back to an earlier embedding."""
    def __init__(self, embed_dim: int, hidden: int = None):
        super().__init__()
        hidden = hidden or max(embed_dim // 2, 128)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


class SliceCausalPredictor(nn.Module):
    """Causal transformer that predicts the next slice embedding from past slice embeddings.

    Used as H_fwd in medical Cycle-NEPA. Takes a sequence of slice-level embeddings
    [z_1, ..., z_T] (possibly with mask tokens) and at each position t produces a
    prediction for z_{t+1}.

    Unlike ForwardPredictor (single-step MLP), this module sees the full causal
    context z_{<=t} at every position.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        num_layers: int = 2,
        max_slices: int = 256,
        ffn_dim: int = None,
    ):
        super().__init__()
        ffn_dim = ffn_dim or embed_dim * 4
        self.pos_embed = nn.Embedding(max_slices, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # pre-norm (more stable)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]  sequence of slice embeddings (may contain mask tokens)
        Returns:
            [B, T, D]  causal output — position t predicts z_{t+1}
        """
        B, T, D = x.shape
        pos = torch.arange(T, device=x.device)
        x = x + self.pos_embed(pos)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=x.device, dtype=x.dtype
        )
        out = self.transformer(x, mask=causal_mask, is_causal=True)
        return self.norm(out)


__all__ = ["ForwardPredictor", "BackwardHead", "SliceCausalPredictor"]
