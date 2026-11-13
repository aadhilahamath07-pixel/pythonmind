
import torch
from tokenizers import Tokenizer

from core.model import PythonMindModel
from core.generate import generate


# ==========================================
# CONFIGURATION
# ==========================================

TOKENIZER_PATH = "tokenizer/pythonmind_targeted_bpe.json"
CHECKPOINT_PATH = "checkpoints/pythonmind_config_e_final.pt"

MAX_NEW_TOKENS = 100

TEST_FUNCTIONS = [
    {
        "name": "calculate_rectangle_area",
        "prompt": "def calculate_rectangle_area(length, width):\n"
    },
    {
        "name": "binary_search",
        "prompt": "def binary_search(items, target):\n"
    },
    {
        "name": "dictionary_contains_key",
        "prompt": "def dictionary_contains_key(data, key):\n"
    },
    {
        "name": "find_second_occurrence",
        "prompt": "def find_second_occurrence(items, target):\n"
    },
    {
        "name": "calculate_weighted_percentage",
        "prompt": "def calculate_weighted_percentage(values, weights):\n"
    }
]


# ==========================================
# LOAD MODEL
# ==========================================

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


# ==========================================
# TEST GENERATION
# ==========================================

def test_training_generation():

    model, tokenizer = load_model()

    eos_id = tokenizer.token_to_id("[EOS]")

    print("\n========== CONFIG E TRAINING GENERATION ==========")
    print("Checkpoint:", CHECKPOINT_PATH)
    print("Tokenizer:", TOKENIZER_PATH)
    print("Maximum New Tokens:", MAX_NEW_TOKENS)

    for test in TEST_FUNCTIONS:

        encoded = tokenizer.encode(test["prompt"])

        input_ids = torch.tensor(
            [encoded.ids],
            dtype=torch.long
        )

        output_ids = generate(
            model,
            input_ids,
            eos_id,
            max_new_tokens=MAX_NEW_TOKENS
        )

        generated_ids = output_ids[0][len(encoded.ids):].tolist()

        eos_generated = (
            eos_id is not None and eos_id in generated_ids
        )

        generated_code = tokenizer.decode(
            output_ids[0].tolist(),
            skip_special_tokens=True
        )

        print("\n" + "=" * 60)
        print("FUNCTION:", test["name"])
        print("PROMPT:", test["prompt"].rstrip())
        print("\nGENERATED CODE:")
        print(generated_code)

        print("\nGENERATION DIAGNOSTICS")
        print("Generated Tokens:", len(generated_ids))
        print("EOS Generated:", eos_generated)

        if eos_generated:
            eos_position = generated_ids.index(eos_id) + 1
            print("EOS Position:", eos_position)

        print("=" * 60)


if __name__ == "__main__":
    test_training_generation()
