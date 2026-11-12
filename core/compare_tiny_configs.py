
import ast
import random
import torch
import torch.nn as nn

from tokenizers import Tokenizer
from core.model import PythonMindModel


TOKENIZER_PATH = "tokenizer/pythonmind_combined_bpe.json"
DATASET_PATH = "data/python_train_combined_split.txt"

FUNCTION_NAMES = [
    "add",
    "cube",
    "calculate_average",
    "reverse_string"
]

EPOCHS = 500
BLOCK_SIZE = 128
SEED = 42

CONFIGS = [
    {
        "name": "A - Original",
        "lr": 0.0003,
        "weight_decay": 0.01,
        "dropout": True
    },
    {
        "name": "B - No Regularization",
        "lr": 0.0003,
        "weight_decay": 0.0,
        "dropout": False
    },
    {
        "name": "C - Higher Learning Rate",
        "lr": 0.001,
        "weight_decay": 0.0,
        "dropout": False
    }
]


def prepare_data(tokenizer):

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

        token_ids = tokenizer.encode(examples[name]).ids
        token_ids.append(eos_id)

        x = token_ids[:-1]
        y = token_ids[1:]

        padding = BLOCK_SIZE - len(x)

        if padding < 0:
            raise ValueError(f"{name} exceeds block size")

        inputs.append(x + [pad_id] * padding)
        targets.append(y + [-100] * padding)

    return (
        torch.tensor(inputs, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long)
    )


def run_experiment(config, input_ids, targets, vocab_size):

    random.seed(SEED)
    torch.manual_seed(SEED)

    model = PythonMindModel(
        vocab_size=vocab_size,
        embedding_dim=64,
        num_heads=4,
        num_layers=2,
        max_seq_length=128
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    loss_function = nn.CrossEntropyLoss(
        ignore_index=-100,
        reduction="sum"
    )

    valid_tokens = (targets != -100).sum().item()

    first_perfect_epoch = None

    for epoch in range(1, EPOCHS + 1):

        if config["dropout"]:
            model.train()
        else:
            model.eval()

        optimizer.zero_grad()

        logits = model(input_ids)

        loss_sum = loss_function(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1)
        )

        loss = loss_sum / valid_tokens

        loss.backward()
        optimizer.step()

        # Evaluate without dropout.
        model.eval()

        with torch.no_grad():

            eval_logits = model(input_ids)
            predictions = eval_logits.argmax(dim=-1)

            correct = (
                (predictions == targets)
                & (targets != -100)
            ).sum().item()

            accuracy = correct / valid_tokens * 100

            eval_loss = loss_function(
                eval_logits.reshape(-1, vocab_size),
                targets.reshape(-1)
            ).item() / valid_tokens

        if accuracy >= 100 and first_perfect_epoch is None:
            first_perfect_epoch = epoch

        if epoch % 100 == 0:
            print(
                f"  Epoch {epoch}: "
                f"Loss={eval_loss:.4f}, "
                f"Accuracy={accuracy:.2f}%"
            )

    return {
        "name": config["name"],
        "loss": eval_loss,
        "accuracy": accuracy,
        "perfect_epoch": first_perfect_epoch
    }


def main():

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

    input_ids, targets = prepare_data(tokenizer)

    print("\n========== CONFIGURATION COMPARISON ==========")

    results = []

    for config in CONFIGS:

        print(f"\nRunning: {config['name']}")
        print("Learning Rate:", config["lr"])
        print("Weight Decay:", config["weight_decay"])
        print("Dropout:", config["dropout"])

        result = run_experiment(
            config,
            input_ids,
            targets,
            tokenizer.get_vocab_size()
        )

        results.append(result)

    print("\n========== FINAL COMPARISON ==========")

    for result in results:

        print("\nConfiguration:", result["name"])
        print("Final Loss:", round(result["loss"], 4))
        print("Final Accuracy:", round(result["accuracy"], 2), "%")
        print("First 100% Epoch:", result["perfect_epoch"])

    print("\n=======================================")


if __name__ == "__main__":
    main()
