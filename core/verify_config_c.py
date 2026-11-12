
import torch
import torch.nn as nn

from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from core.model import PythonMindModel
from core.train_config_c import prepare_dataset

TOKENIZER_PATH = "tokenizer/pythonmind_combined_bpe.json"
CHECKPOINT_PATH = "checkpoints/pythonmind_config_c.pt"
VALIDATION_PATH = "data/python_validation_combined.txt"


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
        max_seq_length=checkpoint["max_seq_length"],
        dropout=0.0
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset, count = prepare_dataset(
        VALIDATION_PATH,
        tokenizer
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False
    )

    loss_function = nn.CrossEntropyLoss(
        ignore_index=-100,
        reduction="sum"
    )

    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

    with torch.no_grad():

        for input_ids, target_ids in loader:

            logits = model(input_ids)

            loss = loss_function(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1)
            )

            mask = target_ids != -100

            total_loss += loss.item()
            total_tokens += mask.sum().item()

            predictions = logits.argmax(dim=-1)

            total_correct += (
                (predictions == target_ids) & mask
            ).sum().item()

    print("\n========== CONFIG C VERIFICATION ==========")
    print("Validation Functions:", count)
    print("Checkpoint Epoch:", checkpoint["epoch"])

    print("Saved Validation Loss:",
          checkpoint["validation_loss"])

    print("Recalculated Validation Loss:",
          total_loss / total_tokens)

    print("Saved Validation Accuracy:",
          checkpoint["validation_accuracy"])

    print("Recalculated Validation Accuracy:",
          total_correct / total_tokens * 100)

    print("===========================================")


if __name__ == "__main__":
    main()
