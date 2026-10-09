import torch
import torch.nn as nn

class GRUEncoderDecoder(nn.Module):
    def __init__(self, num_nodes, input_dim=14, hidden_dim=64, horizon=24):
        super(GRUEncoderDecoder, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        
        # Flatten spatial and feature dims for standard GRU processing
        # Input to GRU: [Batch, SeqLen, Nodes * Features]
        self.encoder = nn.GRU(num_nodes * input_dim, hidden_dim, batch_first=True)
        
        # Decoder takes the last hidden state and generates outputs recursively
        # Or generates direct multi-horizon for simplicity
        self.decoder_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, num_nodes)
        
        self.projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, graph_info=None):
        """
        x: [B, L, N, F]
        """
        B, L, N, F = x.shape
        # Flatten N and F
        x_flat = x.view(B, L, N * F)
        
        # Encode
        _, hidden = self.encoder(x_flat) # hidden: [1, B, hidden_dim]
        
        # Decode recursively
        decoder_input = self.projection(hidden.transpose(0, 1)) # [B, 1, hidden_dim]
        
        outputs = []
        for _ in range(self.horizon):
            out, hidden = self.decoder_gru(decoder_input, hidden)
            pred = self.output_layer(out) # [B, 1, N]
            outputs.append(pred)
            decoder_input = out # feed previous hidden output as next input
            
        outputs = torch.cat(outputs, dim=1) # [B, horizon, N]
        return outputs.transpose(1, 2) # [B, N, horizon]
