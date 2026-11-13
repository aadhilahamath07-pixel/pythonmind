
import torch
from tokenizers import Tokenizer

from core.model import PythonMindModel


TOKENIZER_PATH = "tokenizer/pythonmind_targeted_bpe.json"
CHECKPOINT_PATH = "checkpoints/pythonmind_config_d.pt"
TRAIN_PATH = "data/python_train_targeted_split.txt"

FUNCTION_NAMES = [
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


def load_training_functions():

    with open(TRAIN_PATH, "r", encoding="utf-8") as file:
        source = file.read()

    functions = {}

    for name in FUNCTION_NAMES:

        marker = f"def {name}("
        start = source.find(marker)

        if start == -1:
            raise ValueError(f"Function not found: {name}")

        next_function = source.find("\ndef ", start + 1)

        if next_function == -1:
            function_source = source[start:]
        else:
            function_source = source[start:next_function]

        functions[name] = function_source.strip()

    return functions


def diagnose():

    model, tokenizer = load_model()
    functions = load_training_functions()

    eos_id = tokenizer.token_to_id("[EOS]")

    print("\n========== CONFIG D FIRST TOKEN DIAGNOSTIC ==========")

    for name, source in functions.items():

        prompt = source.splitlines()[0] + "\n"

        prompt_ids = tokenizer.encode(prompt).ids

        input_ids = torch.tensor(
            [prompt_ids],
            dtype=torch.long
        )

        with torch.no_grad():

            logits = model(input_ids)
            next_logits = logits[0, -1, :]
            probabilities = torch.softmax(next_logits, dim=-1)

        top_values, top_indices = torch.topk(
            probabilities,
            k=10
        )

        print("\n" + "=" * 55)
        print("FUNCTION:", name)
        print("PROMPT:", repr(prompt))

        print("\nTOP 10 NEXT TOKEN PREDICTIONS:")

        for probability, token_id in zip(
            top_values.tolist(),
            top_indices.tolist()
        ):

            token = tokenizer.id_to_token(token_id)

            print(
                repr(token),
                f"{probability * 100:.4f}%"
            )

        if eos_id is not None:
            print(
                "\nEOS Probability:",
                f"{probabilities[eos_id].item() * 100:.4f}%"
            )

        print("=" * 55)


if __name__ == "__main__":
    diagnose()
