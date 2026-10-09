import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConv(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConv, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        # x: [B, N, F], adj: [B, N, N] or [N, N]
        if adj.dim() == 2:
            support = torch.matmul(x, self.weight)
            output = torch.matmul(adj, support)
        else:
            support = torch.matmul(x, self.weight)
            output = torch.bmm(adj, support)
        return output + self.bias

class STConvBlock(nn.Module):
    def __init__(self, in_channels, spatial_channels, out_channels, num_nodes):
        super(STConvBlock, self).__init__()
        self.temporal1 = nn.Conv2d(in_channels, spatial_channels, kernel_size=(1, 3), padding=(0, 1))
        self.graph_conv = GraphConv(spatial_channels, spatial_channels)
        self.temporal2 = nn.Conv2d(spatial_channels, out_channels, kernel_size=(1, 3), padding=(0, 1))
        self.norm = nn.BatchNorm2d(num_nodes)

    def forward(self, x, adj):
        # x: [B, C, N, L]
        t1 = F.relu(self.temporal1(x))
        # graph conv
        B, C, N, L = t1.shape
        t1_perm = t1.permute(0, 3, 2, 1).contiguous().view(B*L, N, C)
        
        # We need an adjacency matrix. If none is passed dynamically, we make a static eye
        if adj is None:
            adj = torch.eye(N, device=x.device)
            
        t2 = F.relu(self.graph_conv(t1_perm, adj))
        t2 = t2.view(B, L, N, C).permute(0, 3, 2, 1).contiguous()
        
        t3 = F.relu(self.temporal2(t2))
        return self.norm(t3.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)

class STGCN(nn.Module):
    def __init__(self, num_nodes, input_dim=14, hidden_dim=64, horizon=24):
        super(STGCN, self).__init__()
        self.st_block1 = STConvBlock(input_dim, hidden_dim, hidden_dim, num_nodes)
        self.st_block2 = STConvBlock(hidden_dim, hidden_dim, hidden_dim, num_nodes)
        self.fc = nn.Linear(hidden_dim * 168, horizon) # Assuming seq_len=168
        self.num_nodes = num_nodes

    def forward(self, x, graph_info=None):
        # x: [B, L, N, F] -> [B, F, N, L]
        x = x.permute(0, 3, 2, 1)
        
        adj = None
        if graph_info is not None and len(graph_info) > 0 and 'adj' in graph_info[0]:
            adj = graph_info[0]['adj']
        
        x = self.st_block1(x, adj)
        x = self.st_block2(x, adj)
        
        B, C, N, L = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(B, N, C * L)
        out = self.fc(x) # [B, N, horizon]
        return out
