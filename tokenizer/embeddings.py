
import torch
import torch.nn as nn
from tokenizers import Tokenizer

# Load BPE tokenizer
tokenizer = Tokenizer.from_file(
    "tokenizer/pythonmind_bpe.json"
)

vocab_size = tokenizer.get_vocab_size()
embedding_dim = 8
max_seq_length = 128

# Token embedding
token_embedding = nn.Embedding(
    vocab_size,
    embedding_dim
)

# Position embedding
position_embedding = nn.Embedding(
    max_seq_length,
    embedding_dim
)

# Input text
text = "def calculate_sum(10, 20)"

encoded = tokenizer.encode(text)
input_ids = torch.tensor(
    encoded.ids,
    dtype=torch.long
)

# Generate position IDs
positions = torch.arange(
    len(input_ids)
)

# Generate embeddings
token_vectors = token_embedding(input_ids)
position_vectors = position_embedding(positions)

# Combine both embeddings
final_embeddings = token_vectors + position_vectors

print("Tokens:", encoded.tokens)
print("Input IDs:", input_ids)
print("Position IDs:", positions)

print("\nToken Embedding Shape:",
      token_vectors.shape)

print("Position Embedding Shape:",
      position_vectors.shape)

print("Final Embedding Shape:",
      final_embeddings.shape)
