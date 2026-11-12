
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

TRAIN_PATH = "data/python_train_targeted_split.txt"
VALIDATION_PATH = "data/python_validation_targeted.txt"

TOKENIZER_PATH = "tokenizer/pythonmind_targeted_bpe.json"

CHECKPOINT_PATH = "checkpoints/pythonmind_config_d.pt"

BLOCK_SIZE = 128
BATCH_SIZE = 4
EPOCHS = 50

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0
GRADIENT_CLIP = 1.0

EMBEDDING_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 2
MAX_SEQ_LENGTH = 128

PATIENCE = 10
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
        raise ValueError("Tokenizer is missing EOS or PAD token")

    inputs = []
    targets = []

    for example in examples:

        token_ids = tokenizer.encode(example).ids + [eos_id]

        x = token_ids[:-1]
        y = token_ids[1:]

        if len(x) > BLOCK_SIZE:
            raise ValueError(
                f"Function exceeds block size: {len(x)} tokens"
            )

        padding = BLOCK_SIZE - len(x)

        inputs.append(x + [pad_id] * padding)
        targets.append(y + [-100] * padding)

    input_tensor = torch.tensor(inputs, dtype=torch.long)
    target_tensor = torch.tensor(targets, dtype=torch.long)

    return TensorDataset(input_tensor, target_tensor), len(examples)


# ==========================================
# 4. TRAINING AND EVALUATION
# ==========================================

def run_epoch(model, loader, loss_function, optimizer=None):

    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

    for input_ids, target_ids in loader:

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):

            logits = model(input_ids)

            loss_sum = loss_function(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1)
            )

            valid = target_ids != -100
            valid_tokens = valid.sum().item()

            if valid_tokens == 0:
                continue

            loss = loss_sum / valid_tokens

            if training:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRADIENT_CLIP
                )

                optimizer.step()

        predictions = logits.argmax(dim=-1)

        correct = (
            (predictions == target_ids) & valid
        ).sum().item()

        total_correct += correct
        total_loss += loss_sum.item()
        total_tokens += valid_tokens

    if total_tokens == 0:
        raise ValueError("No valid target tokens found")

    average_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens * 100

    return average_loss, accuracy


# ==========================================
# 5. MAIN TRAINING FUNCTION
# ==========================================

def train():

    set_seed(RANDOM_SEED)

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    vocab_size = tokenizer.get_vocab_size()

    train_dataset, train_functions = prepare_dataset(
        TRAIN_PATH, tokenizer
    )

    validation_dataset, validation_functions = prepare_dataset(
        VALIDATION_PATH, tokenizer
    )

    if train_functions != 173 or validation_functions != 43:
        raise ValueError("Unexpected targeted dataset split")

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

    model = PythonMindModel(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_seq_length=MAX_SEQ_LENGTH,
        dropout=0.0
    )

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

    print("\n========== PYTHONMIND CONFIG D ==========")
    print("Training Functions:", train_functions)
    print("Validation Functions:", validation_functions)
    print("Vocabulary:", vocab_size)
    print("Parameters:", sum(p.numel() for p in model.parameters()))
    print("Learning Rate:", LEARNING_RATE)
    print("Weight Decay:", WEIGHT_DECAY)
    print("Gradient Clipping:", GRADIENT_CLIP)
    print("Dropout: Disabled during training")
    print("Experiment: Targeted Pattern Expansion")
    print("=========================================\n")

    for epoch in range(1, EPOCHS + 1):

        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            loss_function,
            optimizer
        )

        validation_loss, validation_accuracy = run_epoch(
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
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                    "validation_accuracy": validation_accuracy,
                    "training_functions": train_functions,
                    "validation_functions": validation_functions,
                    "random_seed": RANDOM_SEED,
                    "learning_rate": LEARNING_RATE,
                    "weight_decay": WEIGHT_DECAY,
                    "experiment": "targeted_pattern_expansion"
                },
                CHECKPOINT_PATH
            )

            print("Best checkpoint updated:", epoch)

        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 5 == 0:

            print(
                f"Epoch [{epoch}/{EPOCHS}] "
                f"Train Loss: {train_loss:.4f} "
                f"Train Acc: {train_accuracy:.2f}% "
                f"Val Loss: {validation_loss:.4f} "
                f"Val Acc: {validation_accuracy:.2f}%"
            )

        if epochs_without_improvement >= PATIENCE:

            print("\nEarly stopping triggered.")
            print("Stopped at epoch:", epoch)
            break

    print("\n========== CONFIG D COMPLETED ==========")
    print("Best Validation Loss:", round(best_validation_loss, 4))
    print("Checkpoint:", CHECKPOINT_PATH)
    print("========================================")


if __name__ == "__main__":
    train()
