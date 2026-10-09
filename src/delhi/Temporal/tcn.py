import torch.nn as nn

from .residual_block import ResidualBlock
from config import NUM_TCN_BLOCKS


class TCN(nn.Module):
    """
    Temporal Convolutional Network.

    Responsibilities
    ----------------
    1. Stack multiple Residual Blocks.
    2. Learn hierarchical temporal features.
    3. Return temporal feature representations.
    """
    
    def __init__(
        self,
        channels,
        kernel_size=3,
        dropout=0.2,
        num_blocks=NUM_TCN_BLOCKS,
        dilations=None
    ):
        super().__init__()

        self.tcn_blocks = nn.ModuleList()

        for block in range(num_blocks):

            if dilations is not None and block < len(dilations):
                dilation = dilations[block]
            else:
                dilation = 2 ** block

            self.tcn_blocks.append(

                ResidualBlock(
                    channels=channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )

            )

        self.final_norm = nn.GroupNorm(1, channels)
            
        # ---------------------------------------------------------
    # Forward Pass
    # ---------------------------------------------------------

    def forward(
        self,
        x
    ):
        """
        Forward propagation through the
        Temporal Convolutional Network.
        """

        out = x

        # ---------------------------------------
        # Sequential Residual Blocks
        # ---------------------------------------

        for block in self.tcn_blocks:

            out = block(out)

        out = self.final_norm(out)

        return out