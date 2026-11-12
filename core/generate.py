
import torch
from tokenizers import Tokenizer

from core.model import PythonMindModel


TOKENIZER_PATH = "tokenizer/pythonmind_bpe.json"
CHECKPOINT_PATH = "checkpoints/pythonmind_core.pt"


def generate(model, input_ids, eos_id, max_new_tokens=40):

    model.eval()

    with torch.no_grad():
        for _ in range(max_new_tokens):

            input_context = input_ids[:, -model.max_seq_length:]

            logits = model(input_context)
            next_token_logits = logits[:, -1, :]

            next_token = torch.argmax(
                next_token_logits,
                dim=-1,
                keepdim=True
            )

            input_ids = torch.cat(
                (input_ids, next_token),
                dim=1
            )

            if next_token.item() == eos_id:
                break

    return input_ids


if __name__ == "__main__":

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

    eos_id = tokenizer.token_to_id("[EOS]")

    prompts = [
    "def calculate_average(numbers):",
    "def reverse_string(text):",
    "def find_positive(numbers):",
    "def count_vowels(text):",
    "def is_palindrome_text(text):",
    "def find_maximum(numbers):"
    ]

    for prompt in prompts:

        encoded = tokenizer.encode(prompt)

        input_ids = torch.tensor([encoded.ids])

        output_ids = generate(
            model,
            input_ids,
            eos_id
        )

        generated_text = tokenizer.decode(
            output_ids[0].tolist(),
            skip_special_tokens=True
        )

        print("\n" + "=" * 45)
        print("PROMPT:", prompt)
        print("GENERATED CODE:\n", generated_text)
