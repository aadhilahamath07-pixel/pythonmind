
import ast
import torch
import torch.nn as nn

from tokenizers import Tokenizer
from core.model import PythonMindModel


TOKENIZER_PATH = "tokenizer/pythonmind_combined_bpe.json"
CHECKPOINT_PATH = "checkpoints/pythonmind_config_c.pt"
VALIDATION_PATH = "data/python_validation_combined.txt"

BLOCK_SIZE = 128


def load_validation_functions(tokenizer):

    with open(VALIDATION_PATH, "r", encoding="utf-8") as file:
        source = file.read()

    tree = ast.parse(source)

    functions = [
        ast.get_source_segment(source, node)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    eos_id = tokenizer.token_to_id("[EOS]")
    pad_id = tokenizer.token_to_id("[PAD]")

    results = []

    for function in functions:

        token_ids = tokenizer.encode(function).ids + [eos_id]

        x = token_ids[:-1]
        y = token_ids[1:]

        if len(x) > BLOCK_SIZE:
            raise ValueError("Function exceeds block size")

        padding = BLOCK_SIZE - len(x)

        input_ids = torch.tensor(
            [x + [pad_id] * padding],
            dtype=torch.long
        )

        targets = torch.tensor(
            [y + [-100] * padding],
            dtype=torch.long
        )

        results.append((function, input_ids, targets))

    return results


def main():

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True
    )

    model = PythonMindModel(
        vocab_size=checkpoint["vocab_size"],
        embedding_dim=checkpoint["embedding_dim"],
        num_heads=checkpoint["num_heads"],
        num_layers=checkpoint["num_layers"],
        max_seq_length=checkpoint["max_seq_length"],
        dropout=0.0
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    functions = load_validation_functions(tokenizer)

    loss_function = nn.CrossEntropyLoss(
        ignore_index=-100,
        reduction="sum"
    )

    results = []

    with torch.no_grad():

        for source, input_ids, targets in functions:

            logits = model(input_ids)

            predictions = logits.argmax(dim=-1)

            valid_mask = targets != -100

            correct = (
                (predictions == targets) & valid_mask
            ).sum().item()

            total = valid_mask.sum().item()

            loss = loss_function(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            ).item()

            accuracy = correct / total * 100

            function_name = ast.parse(source).body[0].name

            results.append({
                "name": function_name,
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "loss": loss / total
            })

    results.sort(key=lambda item: item["accuracy"])

    print("\n========== PER-FUNCTION DIAGNOSTICS ==========")

    for item in results:

        print(
            f"{item['name']:<30} "
            f"Accuracy: {item['accuracy']:6.2f}% "
            f"Tokens: {item['correct']}/{item['total']} "
            f"Loss: {item['loss']:.4f}"
        )

    average_accuracy = sum(
        item["correct"] for item in results
    ) / sum(item["total"] for item in results) * 100

    print("\nTotal Functions:", len(results))
    print("Overall Token Accuracy:", round(average_accuracy, 2), "%")

    print("===============================================")


if __name__ == "__main__":
    main()
