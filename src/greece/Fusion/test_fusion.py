import torch
from .fusion_layer import FusionLayer


# Create the Fusion Layer
fusion = FusionLayer()


# Fake embeddings just for testing
batch_size = 2
sequence_length = 24
embedding_dim = 64

spatial_embeddings = torch.randn(
    batch_size,
    sequence_length,
    embedding_dim
)

temporal_embeddings = torch.randn(
    batch_size,
    sequence_length,
    embedding_dim
)


# Pass them through the Fusion Layer
output = fusion(
    spatial_embeddings,
    temporal_embeddings
)


print("Spatial input shape :", spatial_embeddings.shape)
print("Temporal input shape:", temporal_embeddings.shape)
print("Fusion output shape :", output.shape)
print("Output mean:", output.mean().item())
print("Output std :", output.std().item())