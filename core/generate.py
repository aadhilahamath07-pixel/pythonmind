import torch
from tokenizers import Tokenizer

from core.model import PythonMindModel


TOKENIZER_PATH = "tokenizer/pythonmind_bpe.json"
CHECKPOINT_PATH = "checkpoints/pythonmind_core.pt"


def _filter_top_k(logits, top_k):
    if top_k is None:
        return logits
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    top_k = min(top_k, logits.size(-1))
    threshold = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def _filter_top_p(logits, top_p):
    if top_p is None:
        return logits
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in the range (0, 1]")

    sorted_logits, sorted_indices = torch.sort(
        logits, descending=True, dim=-1
    )
    cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

    # Keep the first token that crosses the threshold as well.
    remove = cumulative_probs > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False

    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    filtered = torch.full_like(logits, float("-inf"))
    return filtered.scatter(-1, sorted_indices, sorted_logits)


def generate(
    model,
    input_ids,
    eos_id,
    max_new_tokens=40,
    temperature=0.0,
    top_k=None,
    top_p=None
):
    """
    Generate tokens for one or more prompts.

    temperature=0.0 uses deterministic greedy decoding.
    A positive temperature enables sampling; top_k and top_p are optional.
    """
    if input_ids.ndim != 2 or input_ids.size(1) == 0:
        raise ValueError("input_ids must have shape (batch, non-empty sequence)")
    if eos_id is None:
        raise ValueError("A valid EOS token ID is required")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")
    if temperature < 0:
        raise ValueError("temperature cannot be negative")
    if temperature == 0 and (top_k is not None or top_p is not None):
        raise ValueError("top_k/top_p sampling requires temperature > 0")

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    generated_ids = input_ids.clone()

    was_training = model.training
    model.eval()

    finished = torch.zeros(
        generated_ids.size(0),
        dtype=torch.bool,
        device=device
    )

    try:
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                context = generated_ids[:, -model.max_seq_length:]
                logits = model(context)[:, -1, :]

                if temperature == 0:
                    next_token = logits.argmax(dim=-1, keepdim=True)
                else:
                    logits = logits / temperature
                    logits = _filter_top_k(logits, top_k)
                    logits = _filter_top_p(logits, top_p)
                    probabilities = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(
                        probabilities, num_samples=1
                    )

                # Once a row reaches EOS, keep padding its continuation with EOS.
                eos_tokens = torch.full_like(next_token, eos_id)
                next_token = torch.where(
                    finished.unsqueeze(1),
                    eos_tokens,
                    next_token
                )

                generated_ids = torch.cat(
                    (generated_ids, next_token),
                    dim=1
                )

                finished |= next_token.squeeze(1).eq(eos_id)
                if finished.all():
                    break
    finally:
        model.train(was_training)

    return generated_ids


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
    if eos_id is None:
        raise ValueError("Tokenizer does not contain [EOS]")

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
        input_ids = torch.tensor([encoded.ids], dtype=torch.long)

        output_ids = generate(
            model,
            input_ids,
            eos_id,
            max_new_tokens=40
        )

        generated_text = tokenizer.decode(
            output_ids[0].tolist(),
            skip_special_tokens=True
        )

        print("\n" + "=" * 45)
        print("PROMPT:", prompt)
        print("GENERATED CODE:\n", generated_text)
