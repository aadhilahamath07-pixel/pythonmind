
import torch
from tokenizers import Tokenizer

from core.model import PythonMindModel


TOKENIZER_PATH = "tokenizer/pythonmind_combined_bpe.json"
CHECKPOINT_PATH = "checkpoints/pythonmind_fullseq_core.pt"
DATASET_PATH = "data/python_train_combined_split.txt"

FUNCTIONS = [
    "add",
    "cube",
    "calculate_average",
    "reverse_string"
]


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
        max_seq_length=checkpoint["max_seq_length"]
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        source = file.read()

    import ast

    tree = ast.parse(source)

    examples = {
        node.name: ast.get_source_segment(source, node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }

    eos_id = tokenizer.token_to_id("[EOS]")
    pad_id = tokenizer.token_to_id("[PAD]")

    print("\n========== TRAINING EXAMPLE DIAGNOSTIC ==========")

    for name in FUNCTIONS:

        function = examples[name]

        token_ids = tokenizer.encode(function).ids + [eos_id]

        x = token_ids[:-1]
        y = token_ids[1:]

        x += [pad_id] * (128 - len(x))
        y += [-100] * (128 - len(y))

        input_ids = torch.tensor([x], dtype=torch.long)
        targets = torch.tensor([y], dtype=torch.long)

        with torch.no_grad():
            logits = model(input_ids)

        predictions = logits.argmax(dim=-1)

        valid = targets != -100

        correct = (
            (predictions == targets) & valid
        ).sum().item()

        total = valid.sum().item()

        accuracy = correct / total * 100

        print(f"\nFunction: {name}")
        print("Tokens:", total)
        print("Correct:", correct)
        print("Accuracy:", round(accuracy, 2), "%")


if __name__ == "__main__":
    main()
