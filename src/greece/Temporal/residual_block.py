import torch
import torch.nn as nn

class Chomp1D(nn.Module):
    """
    Removes the extra padding introduced by
    causal convolutions.
    """

    def __init__(self, chomp_size):
        super().__init__()

        self.chomp_size = chomp_size

    def forward(self, x):

        return x[:, :, :-self.chomp_size].contiguous()
    



class ResidualBlock(nn.Module):
    """
    Temporal Residual Block.

    Responsibilities
    ----------------
    1. Extract temporal features.
    2. Preserve information through a residual connection.
    3. Return updated temporal features.
    """
    
    def __init__(
        self,
        channels,
        kernel_size,
        dilation,
        dropout
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation
        self.chomp1 = Chomp1D(padding)

        self.chomp2 = Chomp1D(padding)

        # ---------------------------------------
        # First Dilated Convolution
        # ---------------------------------------

        self.conv1 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding
        )

        # ---------------------------------------
        # Second Dilated Convolution
        # ---------------------------------------

        self.conv2 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding
        )

        # ---------------------------------------
        # Normalization
        # GroupNorm with 1 group is LayerNorm over channels
        # ---------------------------------------

        self.norm1 = nn.GroupNorm(1, channels)
        self.norm2 = nn.GroupNorm(1, channels)

        # ---------------------------------------
        # Activation
        # ---------------------------------------

        self.act = nn.GELU()

        # ---------------------------------------
        # Dropout
        # ---------------------------------------

        self.dropout = nn.Dropout(
            p=dropout
        )
        
    # ---------------------------------------------------------
    # Forward Pass
    # ---------------------------------------------------------

    def forward(
        self,
        x
    ):
        """
        Forward propagation through the
        Temporal Residual Block.
        """

        residual = x

        # ---------------------------------------
        # First Convolution
        # ---------------------------------------

        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.norm1(out)
        out = self.act(out)
        out = self.dropout(out)

        # ---------------------------------------
        # Second Convolution
        # (no second dropout: single dropout per
        # block is standard TCN practice and
        # preserves gradient flow better)
        # ---------------------------------------

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.norm2(out)
        out = self.act(out)

        # ---------------------------------------
        # Residual Connection
        # ---------------------------------------

        out = out + residual

        return out