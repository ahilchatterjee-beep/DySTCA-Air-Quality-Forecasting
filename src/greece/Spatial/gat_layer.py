import torch
import torch.nn as nn
import torch.nn.functional as F
from config import EMBEDDING_DIM


class GATLayer(nn.Module):
    """
    Single Graph Attention Layer.

    Responsibilities
    ----------------
    1. Transform node features.
    2. Compute attention coefficients.
    3. Aggregate neighbour information.
    4. Produce updated node embeddings.

    Inputs
    ------
    x : Node feature matrix
        Shape -> [num_nodes, input_dim]

    edge_index : Graph connectivity
        Shape -> [2, num_edges]

    Output
    ------
    Updated node embeddings
    """
    
    def __init__(
        self,
        input_dim,
        output_dim
    ):
        super().__init__()

        # ---------------------------------------
        # Linear Feature Transformation
        # ---------------------------------------

        self.W = nn.Linear(
            input_dim,
            output_dim,
            bias=False
        )

        # ---------------------------------------
        # Attention Mechanism
        # ---------------------------------------

        self.attention = nn.Linear(
            2 * output_dim,
            1,
            bias=False
        )

        # ---------------------------------------
        # Activation
        # ---------------------------------------

        self.leaky_relu = nn.LeakyReLU(
            negative_slope=0.2
        )
        
    # ---------------------------------------------------------
    # Attention Score Computation
    # ---------------------------------------------------------

    def compute_attention_scores(
        self,
        h,
        edge_index
    ):
        """
        Computes the raw attention scores
        for every directed edge.

        Parameters
        ----------
        h : torch.FloatTensor
            Transformed node features.

        edge_index : torch.LongTensor
            Graph connectivity.

        Returns
        -------
        attention_scores : torch.FloatTensor
        """

        source_nodes = edge_index[0]
        target_nodes = edge_index[1]

        source_features = h[source_nodes]
        target_features = h[target_nodes]

        concatenated_features = torch.cat(
            [
                source_features,
                target_features
            ],
            dim=1
        )

        attention_scores = self.attention(
            concatenated_features
        )

        attention_scores = self.leaky_relu(
            attention_scores
        )

        return attention_scores.squeeze(-1)
    
    # ---------------------------------------------------------
    # Attention Coefficient Computation
    # ---------------------------------------------------------

    def compute_attention_coefficients(
        self,
        attention_scores,
        edge_index,
        num_nodes
    ):
        """
        Normalize attention scores over incoming edges for each
        target node, without Python loops.
        """

        if attention_scores.numel() == 0:
            return attention_scores

        target_nodes = edge_index[1]

        # Per-target maximum for numerically stable softmax.
        max_per_target = torch.full(
            (num_nodes,),
            -torch.inf,
            dtype=attention_scores.dtype,
            device=attention_scores.device
        )

        max_per_target.scatter_reduce_(
            0,
            target_nodes,
            attention_scores,
            reduce="amax",
            include_self=True
        )

        exp_scores = torch.exp(
            attention_scores -
            max_per_target[target_nodes]
        )

        # Sum exponentials per target node.
        denominator = torch.zeros(
            num_nodes,
            dtype=attention_scores.dtype,
            device=attention_scores.device
        )

        denominator.index_add_(
            0,
            target_nodes,
            exp_scores
        )

        return (
            exp_scores /
            denominator[target_nodes].clamp_min(
                torch.finfo(
                    attention_scores.dtype
                ).eps
            )
        )

    # ---------------------------------------------------------
    # Neighbor Feature Aggregation
    # ---------------------------------------------------------

    def aggregate_neighbors(
        self,
        h,
        edge_index,
        attention_coefficients
    ):
        """
        Aggregate source-node messages into target nodes without
        a Python loop over edges.
        """

        num_nodes = h.size(0)
        output_dim = h.size(1)

        source_nodes = edge_index[0]
        target_nodes = edge_index[1]

        messages = (
            attention_coefficients.unsqueeze(-1)
            * h[source_nodes]
        )

        updated_features = torch.zeros(
            (num_nodes, output_dim),
            dtype=h.dtype,
            device=h.device
        )

        updated_features.index_add_(
            0,
            target_nodes,
            messages
        )

        return updated_features

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # Forward Pass
    # ---------------------------------------------------------

    def forward(
        self,
        x,
        edge_index=None,
        adj=None
    ):
        """
        Forward propagation through the Graph Attention Layer.
        Supports both:
          1. Batched dense adjacency: x [..., N, F], adj [..., N, N]
          2. Sparse edge_index: x [N, F], edge_index [2, num_edges]
        """

        # -----------------------------------------------------
        # Batched / Dense Adjacency Path (Vectorized)
        # -----------------------------------------------------
        if adj is not None or (x.dim() == 3 and edge_index is None):
            h = self.W(x)  # [..., N, E]
            E = h.shape[-1]
            a_src = self.attention.weight[:, :E]  # [1, E]
            a_tgt = self.attention.weight[:, E:]  # [1, E]

            score_src = torch.matmul(h, a_src.t())  # [..., N, 1]
            score_tgt = torch.matmul(h, a_tgt.t())  # [..., N, 1]
            raw_scores = self.leaky_relu(score_src + score_tgt.transpose(-2, -1))  # [..., N, N]

            if adj is not None:
                mask = (adj > 0)
                raw_scores = raw_scores.masked_fill(~mask, -1e9)

            alpha = F.softmax(raw_scores, dim=-1)
            updated_features = torch.matmul(alpha, h)  # [..., N, E]
            return updated_features

        # -----------------------------------------------------
        # Edge Index Path (2D x, 2xM edge_index)
        # -----------------------------------------------------
        if edge_index is not None and x.dim() == 2:
            num_nodes = x.size(0)
            adj_mat = torch.zeros(
                (num_nodes, num_nodes),
                dtype=x.dtype,
                device=x.device
            )
            adj_mat[edge_index[0], edge_index[1]] = 1.0
            # Ensure self-loops
            adj_mat.fill_diagonal_(1.0)
            return self.forward(x.unsqueeze(0), adj=adj_mat.unsqueeze(0)).squeeze(0)

        # Fallback to standard linear + edge computation
        h = self.W(x)
        attention_scores = self.compute_attention_scores(h, edge_index)
        attention_coefficients = self.compute_attention_coefficients(
            attention_scores, edge_index, h.size(0)
        )
        return self.aggregate_neighbors(h, edge_index, attention_coefficients)

# ==========================================================
# DIRECT SANITY TEST
# ==========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("GAT LAYER SANITY TEST")
    print("=" * 70)

    torch.manual_seed(42)

    x = torch.randn(10, 14)

    edge_index = torch.tensor(
        [
            [0, 0, 1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9],
            [0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9]
        ],
        dtype=torch.long
    )

    model = GATLayer(
        input_dim=14,
        output_dim=EMBEDDING_DIM
    )

    output = model(x, edge_index)

    print("Input shape :", x.shape)
    print("Edges       :", edge_index.shape[1])
    print("Output shape:", output.shape)
    print("NaNs        :", torch.isnan(output).sum().item())
    print("Infs        :", torch.isinf(output).sum().item())

    assert output.shape == (10, EMBEDDING_DIM)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

    print("=" * 70)
    print("GAT LAYER SANITY TEST PASSED")
    print("=" * 70)
