
import torch
import torch.nn as nn

from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from core.model import PythonMindModel
from core.train_config_c import prepare_dataset


TOKENIZER_PATH = "tokenizer/pythonmind_combined_bpe.json"
CHECKPOINT_PATH = "checkpoints/pythonmind_config_c.pt"
TRAIN_PATH = "data/python_train_combined_split.txt"
VALIDATION_PATH = "data/python_validation_combined.txt"

BATCH_SIZE = 4


def evaluate(model, dataset):

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    loss_function = nn.CrossEntropyLoss(
        ignore_index=-100,
        reduction="sum"
    )

    total_loss = 0
    correct_tokens = 0
    total_tokens = 0

    model.eval()

    with torch.no_grad():

        for input_ids, target_ids in loader:

            logits = model(input_ids)

            predictions = logits.argmax(dim=-1)

            valid_mask = target_ids != -100

            loss = loss_function(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1)
            )

            total_loss += loss.item()

            correct_tokens += (
                (predictions == target_ids) & valid_mask
            ).sum().item()

            total_tokens += valid_mask.sum().item()

    average_loss = total_loss / total_tokens

    accuracy = correct_tokens / total_tokens * 100

    return average_loss, accuracy


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

    train_dataset, _ = prepare_dataset(
        TRAIN_PATH,
        tokenizer
    )

    validation_dataset, _ = prepare_dataset(
        VALIDATION_PATH,
        tokenizer
    )

    train_loss, train_accuracy = evaluate(
        model,
        train_dataset
    )

    validation_loss, validation_accuracy = evaluate(
        model,
        validation_dataset
    )

    print("\n========== PYTHONMIND CONFIG C DIAGNOSTICS ==========")

    print("\nTRAINING DATA")
    print("Loss:", round(train_loss, 4))
    print("Token Accuracy:", round(train_accuracy, 2), "%")

    print("\nVALIDATION DATA")
    print("Loss:", round(validation_loss, 4))
    print("Token Accuracy:", round(validation_accuracy, 2), "%")

    print("\nGENERALIZATION GAP")
    print("Loss Gap:", round(validation_loss - train_loss, 4))
    print("Accuracy Gap:", round(train_accuracy - validation_accuracy, 2), "%")

    print("\nCheckpoint Epoch:", checkpoint["epoch"])
    print("Best Validation Loss:", checkpoint["validation_loss"])

    print("============================================")


if __name__ == "__main__":
    main()
