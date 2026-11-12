
import torch
from tokenizers import Tokenizer

from core.model import PythonMindModel


TOKENIZER_PATH = "tokenizer/pythonmind_targeted_bpe.json"
CHECKPOINT_PATH = "checkpoints/pythonmind_config_d.pt"

PROMPT = "def celsius_to_fahrenheit(celsius):"


def main():

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

    input_ids = torch.tensor(
        [tokenizer.encode(PROMPT).ids],
        dtype=torch.long
    )

    with torch.no_grad():

        logits = model(input_ids)

        next_token_logits = logits[0, -1]

        probabilities = torch.softmax(
            next_token_logits,
            dim=-1
        )

        top_probabilities, top_indices = torch.topk(
            probabilities,
            k=15
        )

    print("\n========== CONFIG D NEXT TOKEN DIAGNOSTIC ==========")
    print("Prompt:", PROMPT)
    print("\nTop 15 predicted tokens:\n")

    for probability, token_id in zip(
        top_probabilities,
        top_indices
    ):

        token = tokenizer.id_to_token(token_id.item())

        print(
            f"{token!r:<25} "
            f"{probability.item() * 100:.4f}%"
        )

    print("====================================================")


if __name__ == "__main__":
    main()
