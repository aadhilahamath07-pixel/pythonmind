
import torch
from tokenizers import Tokenizer

from core.model import PythonMindModel


TOKENIZER_PATH = "tokenizer/pythonmind_combined_bpe.json"
CHECKPOINT_PATH = "checkpoints/pythonmind_fullseq_core.pt"

PROMPT = "def add(a, b):"
FUNCTION = "def add(a, b):\n    return a + b"


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

    prompt_ids = tokenizer.encode(PROMPT).ids
    function_ids = tokenizer.encode(FUNCTION).ids + [eos_id]

    input_ids = torch.tensor(
        [function_ids[:-1]],
        dtype=torch.long
    )

    targets = torch.tensor(
        [function_ids[1:]],
        dtype=torch.long
    )

    with torch.no_grad():
        logits = model(input_ids)

    predictions = logits.argmax(dim=-1)

    probabilities = torch.softmax(logits, dim=-1)

    print("\n========== TOKEN PROBABILITY ANALYSIS ==========")

    for i in range(len(targets[0])):

        expected_id = targets[0, i].item()
        predicted_id = predictions[0, i].item()

        expected = tokenizer.id_to_token(expected_id)
        predicted = tokenizer.id_to_token(predicted_id)

        probability = probabilities[0, i, expected_id].item()

        status = "PASS" if expected_id == predicted_id else "FAIL"

        print(
            f"Position {i:02d} | "
            f"Expected: {expected!r:15} | "
            f"Predicted: {predicted!r:15} | "
            f"Correct Probability: {probability:.4f} | "
            f"{status}"
        )


if __name__ == "__main__":
    main()
