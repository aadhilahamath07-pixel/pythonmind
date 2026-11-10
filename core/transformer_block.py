
import torch
import torch.nn as nn

from core.self_attention import CausalSelfAttention


class FeedForward(nn.Module):

    def __init__(self, embedding_dim, dropout=0.1):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.network(x)


class TransformerBlock(nn.Module):

    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

        self.attention = CausalSelfAttention(
            embedding_dim,
            num_heads
        )

        self.feed_forward = FeedForward(
            embedding_dim,
            dropout
        )

    def forward(self, x):

        # Attention + residual connection
        x = x + self.attention(
            self.layer_norm1(x)
        )

        # Feed-forward + residual connection
        x = x + self.feed_forward(
            self.layer_norm2(x)
        )

        return x


if __name__ == "__main__":

    embedding_dim = 64
    num_heads = 4
    sequence_length = 8
    batch_size = 1

    model = TransformerBlock(
        embedding_dim,
        num_heads
    )

    x = torch.randn(
        batch_size,
        sequence_length,
        embedding_dim
    )

    output = model(x)

    print("Input Shape:", x.shape)
    print("Output Shape:", output.shape)

    total_parameters = sum(
        p.numel() for p in model.parameters()
    )

    print("Transformer Block Parameters:", total_parameters)

    print("\nTransformer Block test successful!")
