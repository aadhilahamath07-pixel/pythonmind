
import torch
import torch.nn as nn
import math


class CausalSelfAttention(nn.Module):

    def __init__(self, embedding_dim, num_heads):

        super().__init__()

        assert embedding_dim % num_heads == 0

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # Generate Query, Key and Value
        self.qkv = nn.Linear(
            embedding_dim,
            embedding_dim * 3
        )

        # Final output projection
        self.output_projection = nn.Linear(
            embedding_dim,
            embedding_dim
        )

    def forward(self, x):

        batch_size, seq_length, embedding_dim = x.shape

        # Generate Q, K and V
        qkv = self.qkv(x)

        qkv = qkv.reshape(
            batch_size,
            seq_length,
            3,
            self.num_heads,
            self.head_dim
        )

        qkv = qkv.permute(2, 0, 3, 1, 4)

        query, key, value = qkv.unbind(0)

        # Calculate attention scores
        scores = (
            query @ key.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        # Prevent future-token access
        mask = torch.triu(
            torch.ones(
                seq_length,
                seq_length,
                device=x.device,
                dtype=torch.bool
            ),
            diagonal=1
        )

        scores = scores.masked_fill(
            mask,
            float("-inf")
        )

        # Calculate attention probabilities
        attention_weights = torch.softmax(
            scores,
            dim=-1
        )

        # Combine information from values
        output = attention_weights @ value

        # Restore original dimensions
        output = output.transpose(1, 2).contiguous()

        output = output.reshape(
            batch_size,
            seq_length,
            embedding_dim
        )

        return self.output_projection(output)


# Test the implementation
if __name__ == "__main__":

    embedding_dim = 64
    num_heads = 4
    sequence_length = 8
    batch_size = 1

    model = CausalSelfAttention(
        embedding_dim,
        num_heads
    )

    # Simulated embedding input
    x = torch.randn(
        batch_size,
        sequence_length,
        embedding_dim
    )

    output = model(x)

    print("Input Shape:", x.shape)
    print("Output Shape:", output.shape)
    print("Attention Parameters:",
          sum(p.numel() for p in model.parameters()))

    print("\nSelf-attention test successful!")
