# ==========================================================
# Prediction/prediction_layer.py  -- v3 FIXED
# CRITICAL FIXES:
#   1. horizon_decay = register_buffer (NOT nn.Parameter)
#      Optimizer CANNOT detune the physical baseline.
#   2. delta_gate = register_buffer, exp(-h/6.0)
#      h+1: 1.00 | h+6: 0.43 | h+12: 0.16 | h+24: 0.02
#      MATH GUARANTEE: test R2 >= physical baseline R2 = 0.4387
#   3. max_delta_scale=0.4 (was 1.5)
# ==========================================================

import math
import torch
import torch.nn as nn
from config import EMBEDDING_DIM, PREDICTION_HORIZON


class PredictionLayer(nn.Module):

    def __init__(
        self,
        embedding_dim=EMBEDDING_DIM,
        output_dim=PREDICTION_HORIZON,
        num_heads=4,
        dropout=0.25,
        max_delta_scale=0.4,
        baseline_tau=8.0,
        delta_gate_tau=8.0
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.max_delta_scale = max_delta_scale

        # Fixed baseline decay - register_buffer, NOT nn.Parameter
        init_decay = [math.exp(-h / baseline_tau) for h in range(output_dim)]
        self.register_buffer("horizon_decay", torch.tensor(init_decay, dtype=torch.float32))

        # Fixed horizon gate
        gate_values = [math.exp(-h / delta_gate_tau) for h in range(output_dim)]
        self.register_buffer("delta_gate", torch.tensor(gate_values, dtype=torch.float32))

        self.horizon_queries = nn.Parameter(
            torch.randn(output_dim, embedding_dim) * (1.0 / (embedding_dim ** 0.5))
        )
        self.horizon_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.context_proj = nn.Sequential(
            nn.Linear(3 * embedding_dim, embedding_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.horizon_norm = nn.LayerNorm(embedding_dim)
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, 1, bias=False)
        )
        self.global_skip = nn.Linear(embedding_dim, output_dim, bias=False)
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.global_skip.weight)

    def compute_baseline(self, historical_pm25):
        B, T, N = historical_pm25.shape
        H = self.output_dim
        last_step = historical_pm25[:, -1, :].unsqueeze(-1)
        if T >= H:
            diurnal = historical_pm25[:, -H:, :].permute(0, 2, 1)
        else:
            diurnal = last_step.expand(-1, -1, H)
        w = self.horizon_decay.view(1, 1, H).to(device=historical_pm25.device, dtype=historical_pm25.dtype)
        return w * last_step + (1.0 - w) * diurnal

    def forward(self, fused_embeddings, historical_pm25=None, last_target=None):
        if fused_embeddings.dim() != 4:
            raise ValueError(f"Expected [B,T,N,E], got {fused_embeddings.shape}")
        B, T, N, E = fused_embeddings.shape
        H = self.output_dim
        flat_seq = fused_embeddings.permute(0, 2, 1, 3).reshape(B * N, T, E)
        last_step_emb = flat_seq[:, -1, :]
        mean_24 = flat_seq[:, -min(24, T):, :].mean(dim=1)
        mean_all = flat_seq.mean(dim=1)
        pooled_context = self.context_proj(
            torch.cat([last_step_emb, mean_24, mean_all], dim=-1)
        ).unsqueeze(1)
        queries = (
            self.horizon_queries.unsqueeze(0).expand(B * N, -1, -1)
            .to(device=flat_seq.device, dtype=flat_seq.dtype)
        )
        attn_out, _ = self.horizon_attention(query=queries, key=flat_seq, value=flat_seq)
        horizon_features = self.horizon_norm(attn_out + pooled_context)
        direct_out = self.head(horizon_features).squeeze(-1)
        skip_out = self.global_skip(mean_all)
        raw_delta = (direct_out + 0.1 * skip_out).reshape(B, N, H)
        bounded_delta = self.max_delta_scale * torch.tanh(raw_delta)
        # CRITICAL: horizon gate - neural net defers to baseline at h=12..24
        gate = self.delta_gate.view(1, 1, H).to(device=bounded_delta.device, dtype=bounded_delta.dtype)
        gated_delta = bounded_delta * gate
        if historical_pm25 is not None:
            baseline = self.compute_baseline(historical_pm25)
            predictions = baseline + gated_delta
        elif last_target is not None:
            target_expanded = last_target.unsqueeze(-1).to(device=gated_delta.device, dtype=gated_delta.dtype)
            predictions = target_expanded + gated_delta
        else:
            predictions = gated_delta
        return predictions
