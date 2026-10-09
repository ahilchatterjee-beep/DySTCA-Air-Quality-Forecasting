# ==========================================================
# Model/revin.py
# Reversible Instance Normalization (RevIN)
# Reference: Kim et al., ICLR 2022
# ==========================================================

import torch
import torch.nn as nn


class RevIN(nn.Module):
    """
    Reversible Instance Normalization for Non-Stationary Time Series.
    
    Normalizes each sample window by its local station-level statistics (mean and std),
    and denormalizes predictions back to the original or global space.
    This eliminates seasonal distribution shift between summer/monsoon and winter smog.
    """

    def __init__(self, num_nodes=6, eps=1e-5, affine=False):
        super().__init__()
        self.num_nodes = num_nodes
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, num_nodes))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_nodes))

    def normalize(self, x, window_len=24):
        """
        x: [B, T, N] historical observations for the target pollutant (PM2.5).
        Normalizes using statistics from the most recent window_len steps (e.g. 24h).
        Returns:
            x_norm : [B, T, N]
            mean   : [B, 1, N]
            stdev  : [B, 1, N]
        """
        recent = x[:, -window_len:, :] if x.shape[1] >= window_len else x
        mean = torch.mean(recent, dim=1, keepdim=True).detach()
        stdev = (torch.std(recent, dim=1, keepdim=True) + self.eps).detach()

        x_norm = (x - mean) / stdev

        if self.affine:
            x_norm = x_norm * self.affine_weight + self.affine_bias

        return x_norm, mean, stdev

    def denormalize(self, y, mean, stdev):
        """
        y: [B, N, H] normalized predictions.
        mean: [B, 1, N]
        stdev: [B, 1, N]
        Returns:
            y_denorm : [B, N, H]
        """
        m = mean.permute(0, 2, 1)
        s = stdev.permute(0, 2, 1)

        if self.affine:
            w = self.affine_weight.permute(0, 2, 1)
            b = self.affine_bias.permute(0, 2, 1)
            y = (y - b) / (w + self.eps)

        y_denorm = y * s + m
        return y_denorm
