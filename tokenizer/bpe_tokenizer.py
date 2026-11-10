
from tokenizers import Tokenizer, decoders
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel


# 1. Read our Python training dataset
with open("data/python_train.txt", "r", encoding="utf-8") as file:
    python_corpus = file.readlines()


# 2. Create BPE tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# Preserve Python code spacing and formatting
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()


# 3. Configure tokenizer training
trainer = BpeTrainer(
    vocab_size=500,
    min_frequency=2,
    special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"]
)


# 4. Train tokenizer using Python dataset
tokenizer.train_from_iterator(
    python_corpus,
    trainer=trainer
)


# 5. Save tokenizer
tokenizer.save("tokenizer/pythonmind_bpe.json")


# 6. Display results
print("PYTHONMIND BPE Tokenizer Training Successful!")
print("Training Lines:", len(python_corpus))
print("Vocabulary Size:", tokenizer.get_vocab_size())

sample = "def subtract(a, b): return a - b"

encoded = tokenizer.encode(sample)

print("\nSample Code:", sample)
print("Tokens:", encoded.tokens)
print("Decoded:", tokenizer.decode(encoded.ids))
