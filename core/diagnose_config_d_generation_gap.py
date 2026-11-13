
import torch
from tokenizers import Tokenizer

from core.model import PythonMindModel


TOKENIZER_PATH = "tokenizer/pythonmind_targeted_bpe.json"
CHECKPOINT_PATH = "checkpoints/pythonmind_config_d.pt"
TRAIN_PATH = "data/python_train_targeted_split.txt"

FUNCTION_NAME = "calculate_rectangle_area"
MAX_STEPS = 25


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


def get_function_source():

    with open(TRAIN_PATH, "r", encoding="utf-8") as file:
        source = file.read()

    marker = f"def {FUNCTION_NAME}("
    start = source.find(marker)

    if start == -1:
        raise ValueError("Function not found in training data")

    end = source.find("\ndef ", start + 1)

    if end == -1:
        function_source = source[start:]
    else:
        function_source = source[start:end]

    return function_source.strip() + "\n"


def diagnose():

    model, tokenizer = load_model()

    source = get_function_source()

    prompt = source.splitlines()[0] + "\n"

    full_ids = tokenizer.encode(source).ids
    prompt_ids = tokenizer.encode(prompt).ids

    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "Prompt tokenization does not match the training source prefix"
        )

    eos_id = tokenizer.token_to_id("[EOS]")

    if eos_id is None:
        raise ValueError("[EOS] token not found")

    full_ids.append(eos_id)

    prompt_length = len(prompt_ids)

    teacher_input = torch.tensor(
        [full_ids[:-1]],
        dtype=torch.long
    )

    with torch.no_grad():
        teacher_logits = model(teacher_input)

    generated_ids = prompt_ids.copy()

    print("\n========== CONFIG D GENERATION GAP ==========")
    print("Function:", FUNCTION_NAME)
    print("\nOriginal Function:\n")
    print(source)

    print("\nTOKEN-BY-TOKEN COMPARISON")
    print("-" * 90)

    print(
        f"{'Step':<6}"
        f"{'Expected':<20}"
        f"{'Teacher Pred':<20}"
        f"{'Generated Pred':<20}"
        f"{'Teacher P':<12}"
    )

    print("-" * 90)

    for step in range(min(MAX_STEPS, len(full_ids) - prompt_length)):

        target_position = prompt_length + step
        teacher_position = target_position - 1

        expected_id = full_ids[target_position]

        # Teacher-forced prediction
        teacher_next = teacher_logits[
            0, teacher_position, :
        ]

        teacher_probabilities = torch.softmax(
            teacher_next, dim=-1
        )

        teacher_pred_id = torch.argmax(teacher_next).item()

        expected_probability = teacher_probabilities[
            expected_id
        ].item() * 100

        # Autoregressive prediction
        context = generated_ids[-model.max_seq_length:]

        context_tensor = torch.tensor(
            [context],
            dtype=torch.long
        )

        with torch.no_grad():
            generated_logits = model(context_tensor)

        next_logits = generated_logits[0, -1, :]

        generated_pred_id = torch.argmax(next_logits).item()

        expected_token = tokenizer.id_to_token(expected_id)
        teacher_token = tokenizer.id_to_token(teacher_pred_id)
        generated_token = tokenizer.id_to_token(generated_pred_id)

        print(
            f"{step + 1:<6}"
            f"{str(expected_token):<20}"
            f"{str(teacher_token):<20}"
            f"{str(generated_token):<20}"
            f"{expected_probability:<12.4f}"
        )

        generated_ids.append(generated_pred_id)

        if generated_pred_id == eos_id:
            print("\nGeneration stopped: model predicted EOS.")
            break

    print("-" * 90)

    print("\nGenerated Output:\n")
    print(
        tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        )
    )


if __name__ == "__main__":
    diagnose()
