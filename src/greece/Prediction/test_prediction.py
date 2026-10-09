import torch
from .prediction_layer import PredictionLayer

prediction_layer = PredictionLayer(
    sequence_length=24,
    embedding_dim=64
)

batch_size = 2
sequence_length = 24
num_nodes = 10
embedding_dim = 64

fused_embeddings = torch.randn(
    batch_size,
    sequence_length,
    num_nodes,
    embedding_dim
)

output = prediction_layer(fused_embeddings)

print("Fusion input shape :", fused_embeddings.shape)
print("Prediction shape   :", output.shape)