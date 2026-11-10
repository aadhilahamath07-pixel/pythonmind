
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tokenizers import Tokenizer

from core.model import PythonMindModel


# Configuration
DATA_PATH = "data/python_train.txt"
TOKENIZER_PATH = "tokenizer/pythonmind_bpe.json"
CHECKPOINT_DIR = "checkpoints"

BLOCK_SIZE = 32
BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 0.001



def prepare_dataset(tokenizer):

    import re

    with open(DATA_PATH, "r", encoding="utf-8") as file:
        text = file.read()

    examples = [
        example.strip()
        for example in re.split(r"\n\s*\n", text.strip())
        if example.strip()
    ]

    eos_id = tokenizer.token_to_id("[EOS]")
    pad_id = tokenizer.token_to_id("[PAD]")

    BLOCK_SIZE = 64

    inputs = []
    targets = []

    for example in examples:

        encoded = tokenizer.encode(example)

        # Add EOS to mark the end of each function
        token_ids = encoded.ids + [eos_id]

        # Create input and next-token target sequences
        x = token_ids[:-1]
        y = token_ids[1:]

        # Split longer examples into independent chunks
        for start in range(0, len(x), BLOCK_SIZE):

            input_chunk = x[start:start + BLOCK_SIZE]
            target_chunk = y[start:start + BLOCK_SIZE]

            if not input_chunk:
                continue

            # Pad input
            padding = BLOCK_SIZE - len(input_chunk)

            input_chunk += [pad_id] * padding

            # Ignore padding when calculating loss
            target_chunk += [-100] * padding

            inputs.append(input_chunk)
            targets.append(target_chunk)

    x = torch.tensor(inputs, dtype=torch.long)
    y = torch.tensor(targets, dtype=torch.long)

    dataset = TensorDataset(x, y)

    print("Training Examples:", len(examples))
    print("Training Sequences:", len(dataset))
    print("Sequence Length:", BLOCK_SIZE)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )



def train():

    torch.manual_seed(42)

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    vocab_size = tokenizer.get_vocab_size()

    dataloader = prepare_dataset(tokenizer)

    model = PythonMindModel(vocab_size=vocab_size)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    loss_function = nn.CrossEntropyLoss(ignore_index=-100)

    print("\nStarting PYTHONMIND training...\n")

    for epoch in range(EPOCHS):

        model.train()
        total_loss = 0

        for input_ids, target_ids in dataloader:

            optimizer.zero_grad()

            logits = model(input_ids)

            loss = loss_function(
                logits.reshape(-1, vocab_size),
                target_ids.reshape(-1)
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(dataloader)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch + 1}/{EPOCHS}] "
                f"Loss: {average_loss:.4f}"
            )

    # Save trained model
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        "pythonmind_core.pt"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": vocab_size,
            "embedding_dim": 64,
            "num_heads": 4,
            "num_layers": 2,
            "max_seq_length": 128
        },
        checkpoint_path
    )

    print("\nTraining completed!")
    print("Model saved at:", checkpoint_path)


if __name__ == "__main__":
    train()
