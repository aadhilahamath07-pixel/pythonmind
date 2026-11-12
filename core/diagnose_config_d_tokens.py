
import ast
import torch

from tokenizers import Tokenizer
from core.model import PythonMindModel


TOKENIZER_PATH = "tokenizer/pythonmind_targeted_bpe.json"
CHECKPOINT_PATH = "checkpoints/pythonmind_config_d.pt"
TRAIN_PATH = "data/python_train_targeted_split.txt"

TARGET_FUNCTION = "calculate_rectangle_area"

BLOCK_SIZE = 128
TOP_K = 3


def load_function():

    with open(TRAIN_PATH, "r", encoding="utf-8") as file:
        source = file.read()

    tree = ast.parse(source)

    for node in tree.body:

        if isinstance(node, ast.FunctionDef):
            if node.name == TARGET_FUNCTION:
                return ast.get_source_segment(source, node)

    raise ValueError(
        f"Function not found: {TARGET_FUNCTION}"
    )


def load_model(tokenizer):

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

    return model


def main():

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

    model = load_model(tokenizer)

    source = load_function()

    eos_id = tokenizer.token_to_id("[EOS]")
    pad_id = tokenizer.token_to_id("[PAD]")

    token_ids = tokenizer.encode(source).ids + [eos_id]

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

    with torch.no_grad():

        logits = model(input_ids)

        probabilities = torch.softmax(logits, dim=-1)

        predictions = logits.argmax(dim=-1)

    print("\n========== CONFIG D TOKEN DIAGNOSTIC ==========")
    print("Function:", TARGET_FUNCTION)
    print("\nSource Code:\n")
    print(source)

    correct = 0
    total = 0

    print("\n========== TOKEN PREDICTIONS ==========\n")

    for position in range(len(x)):

        expected_id = targets[0, position].item()

        if expected_id == -100:
            continue

        predicted_id = predictions[0, position].item()

        expected_token = tokenizer.id_to_token(expected_id)
        predicted_token = tokenizer.id_to_token(predicted_id)

        expected_probability = probabilities[
            0, position, expected_id
        ].item() * 100

        is_correct = expected_id == predicted_id

        if is_correct:
            correct += 1

        total += 1

        top_probs, top_ids = torch.topk(
            probabilities[0, position],
            k=TOP_K
        )

        top_predictions = [
            f"{tokenizer.id_to_token(token_id.item())}: "
            f"{prob.item() * 100:.2f}%"
            for prob, token_id in zip(top_probs, top_ids)
        ]

        status = "CORRECT" if is_correct else "WRONG"

        print(f"Position: {position}")
        print("Expected:", repr(expected_token))
        print("Predicted:", repr(predicted_token))
        print(f"Expected Probability: {expected_probability:.4f}%")
        print("Top Predictions:", " | ".join(top_predictions))
        print("Status:", status)
        print("-" * 55)

    print("\n========== SUMMARY ==========")
    print(f"Correct Tokens: {correct}/{total}")
    print(f"Token Accuracy: {correct / total * 100:.2f}%")
    print("=============================")


if __name__ == "__main__":
    main()

