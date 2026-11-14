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
            num_heads,
            dropout=dropout
        )

        self.feed_forward = FeedForward(
            embedding_dim,
            dropout
        )

    def forward(self, x):

        # Pre-LayerNorm attention with residual connection.
        x = x + self.attention(self.layer_norm1(x))

        # Pre-LayerNorm feed-forward with residual connection.
        x = x + self.feed_forward(self.layer_norm2(x))

        return x


if __name__ == "__main__":

    model = TransformerBlock(
        embedding_dim=64,
        num_heads=4,
        dropout=0.1
    )

    x = torch.randn(2, 8, 64)
    output = model(x)

    assert output.shape == x.shape

    print("Input Shape:", x.shape)
    print("Output Shape:", output.shape)
    print(
        "Transformer Block Parameters:",
        sum(p.numel() for p in model.parameters())
    )
    print("\nTransformer Block test successful!")
