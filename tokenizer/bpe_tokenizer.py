
from tokenizers import Tokenizer, decoders
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

from pathlib import Path

# 1. New experiment paths
TRAIN_PATH = Path("data/python_train_targeted_split.txt")
TOKENIZER_PATH = Path("tokenizer/pythonmind_targeted_bpe.json")

# 2. Read only the 173 training functions
with open(TRAIN_PATH, "r", encoding="utf-8") as file:
    python_corpus = file.readlines()

# 3. Create BPE tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

# 4. Configure training
trainer = BpeTrainer(
    vocab_size=800,
    min_frequency=2,
    special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"]
)

# 5. Train tokenizer
tokenizer.train_from_iterator(
    python_corpus,
    trainer=trainer
)

# 6. Save separately
tokenizer.save(str(TOKENIZER_PATH))

# 7. Verify tokenizer
loaded_tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

sample = "def calculate_rectangle_area(length, width): return length * width"

encoded = loaded_tokenizer.encode(sample)
decoded = loaded_tokenizer.decode(encoded.ids)

print("\n========== PYTHONMIND TARGETED TOKENIZER ==========")
print("Training File:", TRAIN_PATH)
print("Training Lines:", len(python_corpus))
print("Vocabulary Size:", loaded_tokenizer.get_vocab_size())
print("Tokenizer Saved:", TOKENIZER_PATH)
print("\nSample Code:", sample)
print("Tokens:", encoded.tokens)
print("Decoded:", decoded)
print("Round-trip Match:", sample == decoded)
print("===================================================")
