
import torch
from tokenizers import Tokenizer

from core.model import PythonMindModel
from core.generate import generate


TOKENIZER_PATH = "tokenizer/pythonmind_combined_bpe.json"
CHECKPOINT_PATH = "checkpoints/pythonmind_fullseq_core.pt"

PROMPTS = [
    "def add(a, b):",
    "def cube(n):",
    "def reverse_string(text):"
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
        max_seq_length=checkpoint["max_seq_length"]
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, tokenizer


def test_memorization():

    model, tokenizer = load_model()

    eos_id = tokenizer.token_to_id("[EOS]")

    print("\n========== PYTHONMIND MEMORIZATION TEST ==========")

    for prompt in PROMPTS:

        encoded = tokenizer.encode(prompt)

        input_ids = torch.tensor(
            [encoded.ids],
            dtype=torch.long
        )

        output_ids = generate(
            model,
            input_ids,
            eos_id,
            max_new_tokens=40
        )

        generated_code = tokenizer.decode(
            output_ids[0].tolist(),
            skip_special_tokens=True
        )

        print("\n" + "=" * 45)
        print("PROMPT:", prompt)
        print("GENERATED CODE:\n", generated_code)


if __name__ == "__main__":
    test_memorization()
