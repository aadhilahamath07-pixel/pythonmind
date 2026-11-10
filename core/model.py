from tokenizers import Tokenizer
import torch
import torch.nn as nn

from core.transformer_block import TransformerBlock


class PythonMindModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=64,
        num_heads=4,
        num_layers=2,
        max_seq_length=128
    ):
        super().__init__()

        self.max_seq_length = max_seq_length

        # Token embeddings
        self.token_embedding = nn.Embedding(
            vocab_size, embedding_dim
        )

        # Positional embeddings
        self.position_embedding = nn.Embedding(
            max_seq_length, embedding_dim
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads)
            for _ in range(num_layers)
        ])

        # Final normalization
        self.final_layer_norm = nn.LayerNorm(embedding_dim)

        # Predict next-token scores
        self.language_model_head = nn.Linear(
            embedding_dim, vocab_size, bias=False
        )

    def forward(self, input_ids):

        batch_size, sequence_length = input_ids.shape

        if sequence_length > self.max_seq_length:
            raise ValueError("Input exceeds maximum sequence length")

        positions = torch.arange(
            sequence_length, device=input_ids.device
        )

        token_vectors = self.token_embedding(input_ids)
        position_vectors = self.position_embedding(positions)

        x = token_vectors + position_vectors

        for block in self.blocks:
            x = block(x)

        x = self.final_layer_norm(x)

        logits = self.language_model_head(x)

        return logits



if __name__ == "__main__":

    # Load our trained BPE tokenizer
    tokenizer = Tokenizer.from_file(
        "tokenizer/pythonmind_bpe.json"
    )

    vocab_size = tokenizer.get_vocab_size()

    # Create the model using the actual vocabulary
    model = PythonMindModel(vocab_size=vocab_size)

    # Test with real Python code
    sample_code = "def calculate_sum(10, 20)"

    encoded = tokenizer.encode(sample_code)

    input_ids = torch.tensor([encoded.ids])

    output = model(input_ids)

    print("Input Code:", sample_code)
    print("Tokens:", encoded.tokens)
    print("Token IDs:", encoded.ids)
    print("Vocabulary Size:", vocab_size)
    print("Input Shape:", input_ids.shape)
    print("Output Shape:", output.shape)
    print(
        "Total Parameters:",
        sum(p.numel() for p in model.parameters())
    )

    print("\nBPE + PythonMind integration successful!")
