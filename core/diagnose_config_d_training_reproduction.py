
import ast
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from core.model import PythonMindModel


TOKENIZER_PATH = "tokenizer/pythonmind_targeted_bpe.json"
CHECKPOINT_PATH = "checkpoints/pythonmind_config_d.pt"
TRAIN_PATH = "data/python_train_targeted_split.txt"

TARGET_FUNCTIONS = [
    "calculate_rectangle_area",
    "binary_search",
    "dictionary_contains_key",
    "find_second_occurrence",
    "calculate_weighted_percentage"
]


def load_model():

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu"
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

    return model, tokenizer


def load_functions():

    with open(TRAIN_PATH, "r", encoding="utf-8") as file:
        source = file.read()

    tree = ast.parse(source)

    functions = {}

    for node in tree.body:

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):

            name = node.name
            code = ast.get_source_segment(source, node)

            if code:
                functions[name] = code

    return functions


def evaluate_function(model, tokenizer, code):

    eos_id = tokenizer.token_to_id("[EOS]")

    token_ids = tokenizer.encode(code).ids + [eos_id]

    if len(token_ids) - 1 > model.max_seq_length:
        raise ValueError("Function exceeds model sequence length")

    input_ids = torch.tensor(
        [token_ids[:-1]],
        dtype=torch.long
    )

    target_ids = torch.tensor(
        [token_ids[1:]],
        dtype=torch.long
    )

    with torch.no_grad():

        logits = model(input_ids)

        predictions = logits.argmax(dim=-1)

        correct = (predictions == target_ids)

        correct_count = correct.sum().item()
        total_tokens = target_ids.numel()

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
            reduction="mean"
        ).item()

        probabilities = torch.softmax(logits, dim=-1)

        expected_probabilities = probabilities.gather(
            -1,
            target_ids.unsqueeze(-1)
        ).squeeze(-1)

    first_error = None

    wrong_positions = (~correct[0]).nonzero(
        as_tuple=True
    )[0]

    if len(wrong_positions) > 0:

        position = wrong_positions[0].item()

        expected_id = target_ids[0, position].item()
        predicted_id = predictions[0, position].item()

        first_error = {
            "position": position,
            "expected": tokenizer.id_to_token(expected_id),
            "predicted": tokenizer.id_to_token(predicted_id),
            "probability": expected_probabilities[
                0, position
            ].item() * 100
        }

    return {
        "correct": correct_count,
        "total": total_tokens,
        "accuracy": correct_count / total_tokens * 100,
        "loss": loss,
        "first_error": first_error
    }


def main():

    model, tokenizer = load_model()
    functions = load_functions()

    print("\n========== CONFIG D TRAINING REPRODUCTION ==========")

    total_correct = 0
    total_tokens = 0

    for name in TARGET_FUNCTIONS:

        if name not in functions:
            print("\nMissing function:", name)
            continue

        result = evaluate_function(
            model,
            tokenizer,
            functions[name]
        )

        total_correct += result["correct"]
        total_tokens += result["total"]

        print("\n" + "=" * 60)
        print("FUNCTION:", name)
        print(
            f"Token Accuracy: {result['correct']}/"
            f"{result['total']} "
            f"({result['accuracy']:.2f}%)"
        )
        print(f"Average Loss: {result['loss']:.4f}")

        error = result["first_error"]

        if error:
            print("First Incorrect Prediction:")
            print("Position:", error["position"])
            print("Expected:", repr(error["expected"]))
            print("Predicted:", repr(error["predicted"]))
            print(
                "Expected Token Probability:",
                f"{error['probability']:.4f}%"
            )
        else:
            print("All tokens predicted correctly.")

    print("\n========== FIVE-FUNCTION SUMMARY ==========")

    if total_tokens:
        print("Correct Tokens:", total_correct)
        print("Total Tokens:", total_tokens)
        print(
            "Overall Accuracy:",
            f"{total_correct / total_tokens * 100:.2f}%"
        )

    # Measure the complete training split as a reference.
    all_correct = 0
    all_tokens = 0

    for name, code in functions.items():

        result = evaluate_function(model, tokenizer, code)

        all_correct += result["correct"]
        all_tokens += result["total"]

    print("\n========== FULL TRAINING SET ==========")
    print("Training Functions:", len(functions))
    print("Correct Tokens:", all_correct)
    print("Total Tokens:", all_tokens)
    print(
        "Training Token Accuracy:",
        f"{all_correct / all_tokens * 100:.2f}%"
    )
    print("=======================================")


if __name__ == "__main__":
    main()
