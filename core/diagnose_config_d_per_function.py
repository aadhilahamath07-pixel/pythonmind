
import ast
import torch

from tokenizers import Tokenizer
from core.model import PythonMindModel


TOKENIZER_PATH = "tokenizer/pythonmind_targeted_bpe.json"
CHECKPOINT_PATH = "checkpoints/pythonmind_config_d.pt"
TRAIN_PATH = "data/python_train_targeted_split.txt"

BLOCK_SIZE = 128


def load_functions(tokenizer):

    with open(TRAIN_PATH, "r", encoding="utf-8") as file:
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

        name = ast.parse(function).body[0].name

        results.append((name, input_ids, targets))

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

    functions = load_functions(tokenizer)

    results = []

    with torch.no_grad():

        for name, input_ids, targets in functions:

            logits = model(input_ids)

            predictions = logits.argmax(dim=-1)

            valid = targets != -100

            correct = (
                (predictions == targets) & valid
            ).sum().item()

            total = valid.sum().item()

            accuracy = correct / total * 100

            results.append({
                "name": name,
                "accuracy": accuracy,
                "correct": correct,
                "total": total
            })

    results.sort(key=lambda item: item["accuracy"])

    total_correct = sum(item["correct"] for item in results)
    total_tokens = sum(item["total"] for item in results)

    print("\n========== CONFIG D PER-FUNCTION DIAGNOSTICS ==========")

    for item in results:

        print(
            f"{item['name']:<35} "
            f"Accuracy: {item['accuracy']:6.2f}% "
            f"Tokens: {item['correct']}/{item['total']}"
        )

    print("\nTotal Functions:", len(results))
    print("Overall Token Accuracy:",
          round(total_correct / total_tokens * 100, 2), "%")

    print("=======================================================")


if __name__ == "__main__":
    main()
