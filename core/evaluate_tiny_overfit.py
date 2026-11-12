
import torch
from tokenizers import Tokenizer

from core.model import PythonMindModel
from core.generate import generate


TOKENIZER_PATH = "tokenizer/pythonmind_combined_bpe.json"
CHECKPOINT_PATH = "checkpoints/pythonmind_tiny_overfit.pt"

PROMPTS = [
    "def add(a, b):",
    "def cube(n):",
    "def calculate_average(numbers):",
    "def reverse_string(text):"
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

    eos_id = tokenizer.token_to_id("[EOS]")

    print("\n========== TINY MODEL GENERATION ==========")

    for prompt in PROMPTS:

        prompt_ids = tokenizer.encode(prompt).ids

        input_ids = torch.tensor(
            [prompt_ids],
            dtype=torch.long
        )

        output_ids = generate(
            model,
            input_ids,
            eos_id,
            max_new_tokens=40
        )

        generated = tokenizer.decode(
            output_ids[0].tolist(),
            skip_special_tokens=True
        )

        print("\nPROMPT:", prompt)
        print("GENERATED:")
        print(generated)

    print("\n===========================================")


if __name__ == "__main__":
    main()

