import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):

    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.dropout = dropout

        # One projection produces Query, Key and Value.
        self.qkv = nn.Linear(embedding_dim, embedding_dim * 3)

        self.output_projection = nn.Linear(
            embedding_dim,
            embedding_dim
        )

    def forward(self, x):

        if x.ndim != 3:
            raise ValueError(
                "Expected input shape (batch, sequence, embedding)"
            )

        batch_size, seq_length, embedding_dim = x.shape

        if embedding_dim != self.embedding_dim:
            raise ValueError(
                f"Expected embedding size {self.embedding_dim}, "
                f"received {embedding_dim}"
            )

        qkv = self.qkv(x)

        qkv = qkv.reshape(
            batch_size,
            seq_length,
            3,
            self.num_heads,
            self.head_dim
        )

        # Shape: (3, batch, heads, sequence, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)

        dropout_p = self.dropout if self.training else 0.0

        # PyTorch's optimized implementation can use fused attention
        # kernels on supported devices. Keep a compatible fallback.
        if hasattr(F, "scaled_dot_product_attention"):
            output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=dropout_p,
                is_causal=True
            )
        else:
            scores = (
                query @ key.transpose(-2, -1)
            ) / (self.head_dim ** 0.5)

            causal_mask = torch.triu(
                torch.ones(
                    seq_length,
                    seq_length,
                    device=x.device,
                    dtype=torch.bool
                ),
                diagonal=1
            )

            scores = scores.masked_fill(
                causal_mask,
                float("-inf")
            )

            attention_weights = torch.softmax(scores, dim=-1)
            attention_weights = F.dropout(
                attention_weights,
                p=dropout_p,
                training=self.training
            )

            output = attention_weights @ value

        # (batch, heads, sequence, head_dim)
        # -> (batch, sequence, embedding)
        output = output.transpose(1, 2).contiguous()
        output = output.reshape(
            batch_size,
            seq_length,
            self.embedding_dim
        )

        return self.output_projection(output)


if __name__ == "__main__":

    embedding_dim = 64
    num_heads = 4
    sequence_length = 8
    batch_size = 2

    model = CausalSelfAttention(
        embedding_dim,
        num_heads,
        dropout=0.1
    )

    x = torch.randn(
        batch_size,
        sequence_length,
        embedding_dim
    )

    output = model(x)

    assert output.shape == x.shape

    print("Input Shape:", x.shape)
    print("Output Shape:", output.shape)
    print(
        "Attention Parameters:",
        sum(p.numel() for p in model.parameters())
    )
    print("\nCausal self-attention test successful!")
}
