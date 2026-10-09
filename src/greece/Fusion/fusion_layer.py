# ==========================================================
# Fusion/fusion_layer.py
# Spatio-Temporal Cross-Attention Fusion Layer
#
# Input:
#     Temporal: [B, T, N, E]
#     Spatial:  [B, T, N, E]
#
# Output:
#     Fused:    [B, T, N, E]
# ==========================================================

import torch
import torch.nn as nn
from config import EMBEDDING_DIM, NUM_ATTENTION_HEADS


class FusionLayer(nn.Module):
    """
    Principled Spatio-Temporal Fusion via Bidirectional Multi-Head Cross-Attention,
    Learnable Dynamic Gating, and Residual Skip Connections.
    """

    def __init__(
        self,
        embedding_dim=EMBEDDING_DIM,
        num_tokens=8,
        num_heads=NUM_ATTENTION_HEADS,
        dropout=0.1
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        # --------------------------------------------------
        # Bidirectional Multi-Head Cross-Attention
        # --------------------------------------------------
        # Spatial queries Temporal
        self.cross_attn_s2t = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Temporal queries Spatial
        self.cross_attn_t2s = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # --------------------------------------------------
        # Dynamic Gating & Feature Projection
        # --------------------------------------------------
        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim),
            nn.Sigmoid()
        )

        self.cross_projection = nn.Linear(2 * embedding_dim, embedding_dim)

        # --------------------------------------------------
        # Feed-Forward Sublayer & Layer Normalization
        # --------------------------------------------------
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, 2 * embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * embedding_dim, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        temporal_embeddings,
        spatial_embeddings
    ):
        """
        temporal_embeddings: [B, T, N, E]
        spatial_embeddings:  [B, T, N, E]
        returns:             [B, T, N, E]
        """
        B, T, N, E = temporal_embeddings.shape

        # Flatten B*N to treat each station's timeline: [B*N, T, E]
        t_flat = temporal_embeddings.permute(0, 2, 1, 3).reshape(B * N, T, E)
        s_flat = spatial_embeddings.permute(0, 2, 1, 3).reshape(B * N, T, E)

        # 1. Bidirectional Cross-Attention along temporal dimension
        # Spatial queries Temporal (what temporal context matters for this spatial state?)
        attn_s2t, _ = self.cross_attn_s2t(query=s_flat, key=t_flat, value=t_flat)

        # Temporal queries Spatial (what spatial context matters for this temporal step?)
        attn_t2s, _ = self.cross_attn_t2s(query=t_flat, key=s_flat, value=s_flat)

        # 2. Dynamic Gated Combination with Residual Skip Connections
        gate = self.gate_mlp(torch.cat([t_flat, s_flat], dim=-1))
        cross_features = self.cross_projection(torch.cat([attn_s2t, attn_t2s], dim=-1))

        # Full preservation of both branches via gated residual connection
        fused = gate * t_flat + (1.0 - gate) * s_flat + cross_features
        fused = self.norm1(fused)

        # 3. Feed-Forward Network
        fused = fused + self.ffn(fused)
        fused = self.norm2(fused)

        # 4. Restore original layout: [B, T, N, E]
        fused = fused.reshape(B, N, T, E).permute(0, 2, 1, 3)

        return fused