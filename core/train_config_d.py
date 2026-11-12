import os
import ast
import csv
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
HISTORY_PATH = "checkpoints/pythonmind_config_d_history.csv"

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
DROPOUT = 0.0

PATIENCE = 10
SCHEDULER_PATIENCE = 3
RANDOM_SEED = 42

EXPECTED_TRAIN_FUNCTIONS = 173
EXPECTED_VALIDATION_FUNCTIONS = 43


# ==========================================
# 2. REPRODUCIBILITY AND DEVICE
# ==========================================

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    examples = [example for example in examples if example]

    if not examples:
        raise ValueError(f"No functions found in {file_path}")

    eos_id = tokenizer.token_to_id("[EOS]")
    pad_id = tokenizer.token_to_id("[PAD]")

    if eos_id is None or pad_id is None:
        raise ValueError("Tokenizer is missing [EOS] or [PAD]")

    inputs = []
    targets = []

    for example in examples:
        token_ids = tokenizer.encode(example).ids + [eos_id]
        x = token_ids[:-1]
        y = token_ids[1:]

        if len(x) > BLOCK_SIZE:
            raise ValueError(
                f"Function in {file_path} exceeds BLOCK_SIZE: "
                f"{len(x)} > {BLOCK_SIZE}"
            )

        padding = BLOCK_SIZE - len(x)
        inputs.append(x + [pad_id] * padding)
        targets.append(y + [-100] * padding)

    return (
        TensorDataset(
            torch.tensor(inputs, dtype=torch.long),
            torch.tensor(targets, dtype=torch.long)
        ),
        len(examples)
    )


# ==========================================
# 4. TRAINING AND EVALUATION
# ==========================================

def run_epoch(model, loader, loss_function, device, optimizer=None):

    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

    for input_ids, target_ids in loader:
        input_ids = input_ids.to(device, non_blocking=True)
        target_ids = target_ids.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(input_ids)

            loss_sum = loss_function(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1)
            )

            valid = target_ids.ne(-100)
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
        total_correct += (
            (predictions == target_ids) & valid
        ).sum().item()

        total_loss += loss_sum.detach().item()
        total_tokens += valid_tokens

    if total_tokens == 0:
        raise ValueError("No valid target tokens found in this epoch")

    return (
        total_loss / total_tokens,
        total_correct / total_tokens * 100
    )


# ==========================================
# 5. CHECKPOINT AND HISTORY
# ==========================================

def save_checkpoint(model, path, metadata):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    temporary_path = path + ".tmp"
    torch.save(
        {
            **metadata,
            "model_state_dict": model.state_dict()
        },
        temporary_path
    )
    os.replace(temporary_path, path)


def append_history(path, row):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    file_exists = os.path.exists(path)

    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ==========================================
# 6. MAIN TRAINING FUNCTION
# ==========================================

def train():

    set_seed(RANDOM_SEED)
    device = get_device()

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    vocab_size = tokenizer.get_vocab_size()

    train_dataset, train_functions = prepare_dataset(
        TRAIN_PATH, tokenizer
    )
    validation_dataset, validation_functions = prepare_dataset(
        VALIDATION_PATH, tokenizer
    )

    if train_functions != EXPECTED_TRAIN_FUNCTIONS:
        raise ValueError(
            f"Expected {EXPECTED_TRAIN_FUNCTIONS} training functions, "
            f"found {train_functions}"
        )

    if validation_functions != EXPECTED_VALIDATION_FUNCTIONS:
        raise ValueError(
            f"Expected {EXPECTED_VALIDATION_FUNCTIONS} validation functions, "
            f"found {validation_functions}"
        )

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=pin_memory
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=pin_memory
    )

    model = PythonMindModel(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_seq_length=MAX_SEQ_LENGTH,
        dropout=DROPOUT
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=SCHEDULER_PATIENCE,
        min_lr=1e-5
    )

    loss_function = nn.CrossEntropyLoss(
        ignore_index=-100,
        reduction="sum"
    )

    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    os.makedirs("checkpoints", exist_ok=True)

    # Start a fresh history for this training run.
    if os.path.exists(HISTORY_PATH):
        os.remove(HISTORY_PATH)

    print("\n========== PYTHONMIND CONFIG D ==========")
    print("Device:", device)
    print("Training Functions:", train_functions)
    print("Validation Functions:", validation_functions)
    print("Vocabulary:", vocab_size)
    print("Parameters:", sum(p.numel() for p in model.parameters()))
    print("Learning Rate:", LEARNING_RATE)
    print("Weight Decay:", WEIGHT_DECAY)
    print("Gradient Clipping:", GRADIENT_CLIP)
    print("Dropout:", DROPOUT)
    print("Scheduler: ReduceLROnPlateau")
    print("Experiment: Targeted Pattern Expansion")
    print("=========================================\n")

    for epoch in range(1, EPOCHS + 1):

        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            loss_function,
            device,
            optimizer
        )

        validation_loss, validation_accuracy = run_epoch(
            model,
            validation_loader,
            loss_function,
            device
        )

        scheduler.step(validation_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        improved = validation_loss < best_validation_loss

        if improved:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0

            save_checkpoint(
                model,
                CHECKPOINT_PATH,
                {
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
                    "dropout": DROPOUT,
                    "experiment": "targeted_pattern_expansion"
                }
            )
            print("Best checkpoint updated:", epoch)
        else:
            epochs_without_improvement += 1

        append_history(
            HISTORY_PATH,
            {
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "train_accuracy": f"{train_accuracy:.4f}",
                "validation_loss": f"{validation_loss:.6f}",
                "validation_accuracy": f"{validation_accuracy:.4f}",
                "learning_rate": f"{current_lr:.8f}",
                "best_checkpoint_updated": improved
            }
        )

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"Epoch [{epoch}/{EPOCHS}] "
                f"Train Loss: {train_loss:.4f} "
                f"Train Acc: {train_accuracy:.2f}% "
                f"Val Loss: {validation_loss:.4f} "
                f"Val Acc: {validation_accuracy:.2f}% "
                f"LR: {current_lr:.6f}"
            )

        if epochs_without_improvement >= PATIENCE:
            print("\nEarly stopping triggered.")
            print("Stopped at epoch:", epoch)
            break

    print("\n========== CONFIG D COMPLETED ==========")
    print("Best Validation Loss:", round(best_validation_loss, 4))
    print("Checkpoint:", CHECKPOINT_PATH)
    print("Training History:", HISTORY_PATH)
    print("========================================")


if __name__ == "__main__":
    train()
