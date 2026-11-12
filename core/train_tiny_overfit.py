
import ast
import random
import torch
import torch.nn as nn

from tokenizers import Tokenizer
from core.model import PythonMindModel


TOKENIZER_PATH = "tokenizer/pythonmind_combined_bpe.json"
DATASET_PATH = "data/python_train_combined_split.txt"

CHECKPOINT_PATH = "checkpoints/pythonmind_tiny_overfit.pt"

FUNCTION_NAMES = [
    "add",
    "cube",
    "calculate_average",
    "reverse_string"
]

BLOCK_SIZE = 128
EPOCHS = 500
LEARNING_RATE = 0.001

SEED = 42


def main():

    random.seed(SEED)
    torch.manual_seed(SEED)

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        source = file.read()

    tree = ast.parse(source)

    examples = {
        node.name: ast.get_source_segment(source, node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }

    eos_id = tokenizer.token_to_id("[EOS]")
    pad_id = tokenizer.token_to_id("[PAD]")

    inputs = []
    targets = []

    for name in FUNCTION_NAMES:

        if name not in examples:
            raise ValueError(f"Missing function: {name}")

        token_ids = tokenizer.encode(examples[name]).ids
        token_ids.append(eos_id)

        x = token_ids[:-1]
        y = token_ids[1:]

        if len(x) > BLOCK_SIZE:
            raise ValueError(f"{name} exceeds block size")

        padding = BLOCK_SIZE - len(x)

        x += [pad_id] * padding
        y += [-100] * padding

        inputs.append(x)
        targets.append(y)

    input_ids = torch.tensor(inputs, dtype=torch.long)
    target_ids = torch.tensor(targets, dtype=torch.long)

    valid_tokens = (target_ids != -100).sum().item()

    model = PythonMindModel(
        vocab_size=tokenizer.get_vocab_size(),
        embedding_dim=64,
        num_heads=4,
        num_layers=2,
        max_seq_length=128
    )

    # Disable dropout for this memorization diagnostic.
    model.eval()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0
    )

    loss_function = nn.CrossEntropyLoss(
        ignore_index=-100,
        reduction="sum"
    )

    print("\n========== TINY DATASET EXPERIMENT ==========")
    print("Functions:", FUNCTION_NAMES)
    print("Training Examples:", len(inputs))
    print("Valid Tokens:", valid_tokens)
    print("Maximum Epochs:", EPOCHS)
    print("============================================")

    for epoch in range(1, EPOCHS + 1):

        optimizer.zero_grad()

        logits = model(input_ids)

        loss_sum = loss_function(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1)
        )

        loss = loss_sum / valid_tokens

        loss.backward()
        optimizer.step()

        if epoch % 25 == 0 or epoch == 1:

            with torch.no_grad():

                predictions = logits.argmax(dim=-1)

                correct = (
                    (predictions == target_ids)
                    & (target_ids != -100)
                ).sum().item()

                accuracy = correct / valid_tokens * 100

            print(
                f"Epoch [{epoch}/{EPOCHS}] "
                f"Loss: {loss.item():.4f} "
                f"Accuracy: {accuracy:.2f}%"
            )

    # Final evaluation
    with torch.no_grad():

        logits = model(input_ids)
        predictions = logits.argmax(dim=-1)

        print("\n========== FINAL RESULTS ==========")

        for index, name in enumerate(FUNCTION_NAMES):

            valid = target_ids[index] != -100

            correct = (
                (predictions[index] == target_ids[index])
                & valid
            ).sum().item()

            total = valid.sum().item()

            print(
                f"{name}: {correct}/{total} "
                f"({correct / total * 100:.2f}%)"
            )

        total_correct = (
            (predictions == target_ids)
            & (target_ids != -100)
        ).sum().item()

        print(
            "\nOverall Accuracy:",
            round(total_correct / valid_tokens * 100, 2),
            "%"
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": tokenizer.get_vocab_size(),
            "embedding_dim": 64,
            "num_heads": 4,
            "num_layers": 2,
            "max_seq_length": 128,
            "functions": FUNCTION_NAMES
        },
        CHECKPOINT_PATH
    )

    print("\nCheckpoint saved:", CHECKPOINT_PATH)


if __name__ == "__main__":
    main()
