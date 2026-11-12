
import os
import ast
import random
import torch
import torch.nn as nn

from tokenizers import Tokenizer
from torch.utils.data import DataLoader, TensorDataset

from core.model import PythonMindModel


# ==========================================
# 1. CONFIGURATION
# ==========================================

TRAIN_PATH = "data/python_train_combined_split.txt"
VALIDATION_PATH = "data/python_validation_combined.txt"

TOKENIZER_PATH = "tokenizer/pythonmind_combined_bpe.json"

CHECKPOINT_PATH = "checkpoints/pythonmind_fullseq_core.pt"

BLOCK_SIZE = 128
OVERLAP = 0
STRIDE = 128

BATCH_SIZE = 4
EPOCHS = 50

LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0.01

EMBEDDING_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 2
MAX_SEQ_LENGTH = 128

PATIENCE = 7
RANDOM_SEED = 42


# ==========================================
# 2. REPRODUCIBILITY
# ==========================================

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


# ==========================================
# 3. DATASET PREPARATION
# ==========================================

def prepare_dataset(file_path, tokenizer):

    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    syntax_tree = ast.parse(text)

    examples = [
        ast.get_source_segment(text, node)
        for node in syntax_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    if not examples:
        raise ValueError(f"No functions found in {file_path}")

    eos_id = tokenizer.token_to_id("[EOS]")
    pad_id = tokenizer.token_to_id("[PAD]")

    if eos_id is None or pad_id is None:
        raise ValueError("Required special tokens are missing!")

    inputs = []
    targets = []

    for example in examples:

        encoded = tokenizer.encode(example)
        token_ids = encoded.ids + [eos_id]

        x = token_ids[:-1]
        y = token_ids[1:]

        if len(x) > BLOCK_SIZE:
            raise ValueError(
                f"Function exceeds block size: {len(x)} tokens"
            )

        padding = BLOCK_SIZE - len(x)

        input_chunk = x + [pad_id] * padding
        target_chunk = y + [-100] * padding

        inputs.append(input_chunk)
        targets.append(target_chunk)

    input_tensor = torch.tensor(inputs, dtype=torch.long)
    target_tensor = torch.tensor(targets, dtype=torch.long)

    dataset = TensorDataset(input_tensor, target_tensor)

    return dataset, len(examples)


# ==========================================
# 4. LOSS CALCULATION
# ==========================================

def calculate_loss(model, dataloader, loss_function, optimizer=None):

    training_mode = optimizer is not None

    if training_mode:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_tokens = 0

    for input_ids, target_ids in dataloader:

        if training_mode:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training_mode):

            logits = model(input_ids)

            loss_sum = loss_function(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1)
            )

            valid_tokens = (target_ids != -100).sum().item()

            if valid_tokens == 0:
                continue

            loss = loss_sum / valid_tokens

            if training_mode:
                loss.backward()
                optimizer.step()

        total_loss += loss_sum.item()
        total_tokens += valid_tokens

    if total_tokens == 0:
        raise ValueError("No valid target tokens found!")

    return total_loss / total_tokens


# ==========================================
# 5. MAIN TRAINING FUNCTION
# ==========================================

def train():

    set_seed(RANDOM_SEED)

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    vocab_size = tokenizer.get_vocab_size()

    print("Tokenizer Vocabulary Size:", vocab_size)

    train_dataset, train_functions = prepare_dataset(
        TRAIN_PATH, tokenizer
    )

    validation_dataset, validation_functions = prepare_dataset(
        VALIDATION_PATH, tokenizer
    )

    if train_functions != 196 or validation_functions != 49:
        raise ValueError(
            f"Unexpected split: train={train_functions}, "
            f"validation={validation_functions}"
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print("\n========== FULL-FUNCTION DATASET ==========")
    print("Training Functions:", train_functions)
    print("Training Sequences:", len(train_dataset))
    print("Validation Functions:", validation_functions)
    print("Validation Sequences:", len(validation_dataset))
    print("Sequence Length:", BLOCK_SIZE)
    print("Overlap:", OVERLAP)
    print("===========================================")

    model = PythonMindModel(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_seq_length=MAX_SEQ_LENGTH
    )

    total_parameters = sum(
        parameter.numel() for parameter in model.parameters()
    )

    print("Model Parameters:", total_parameters)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    loss_function = nn.CrossEntropyLoss(
        ignore_index=-100,
        reduction="sum"
    )

    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    os.makedirs("checkpoints", exist_ok=True)

    print("\nStarting PYTHONMIND FULL-FUNCTION training...\n")

    # ======================================
    # 6. TRAINING LOOP
    # ======================================

    for epoch in range(EPOCHS):

        train_loss = calculate_loss(
            model,
            train_loader,
            loss_function,
            optimizer
        )

        validation_loss = calculate_loss(
            model,
            validation_loader,
            loss_function
        )

        if validation_loss < best_validation_loss:

            best_validation_loss = validation_loss
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab_size": vocab_size,
                    "embedding_dim": EMBEDDING_DIM,
                    "num_heads": NUM_HEADS,
                    "num_layers": NUM_LAYERS,
                    "max_seq_length": MAX_SEQ_LENGTH,
                    "block_size": BLOCK_SIZE,
                    "overlap": OVERLAP,
                    "epoch": epoch + 1,
                    "validation_loss": validation_loss,
                    "training_functions": train_functions,
                    "validation_functions": validation_functions,
                    "random_seed": RANDOM_SEED
                },
                CHECKPOINT_PATH
            )

            print("Best checkpoint updated at epoch:", epoch + 1)

        else:
            epochs_without_improvement += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:

            print(
                f"Epoch [{epoch + 1}/{EPOCHS}] "
                f"Training Loss: {train_loss:.4f} "
                f"Validation Loss: {validation_loss:.4f}"
            )

        if epochs_without_improvement >= PATIENCE:

            print("\nEarly stopping triggered!")
            print("Stopped at Epoch:", epoch + 1)
            break

    # ======================================
    # 7. FINAL RESULTS
    # ======================================

    print("\n========== FULL-FUNCTION TRAINING COMPLETED ==========")
    print("Best Validation Loss:", round(best_validation_loss, 4))
    print("Checkpoint:", CHECKPOINT_PATH)
    print("=======================================================")


if __name__ == "__main__":
    train()
