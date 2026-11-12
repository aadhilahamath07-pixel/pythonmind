import ast
import json
import time
import torch
from tokenizers import Tokenizer

from core.model import PythonMindModel
from core.generate import generate


# ==========================================
# CONFIGURATION
# ==========================================

TOKENIZER_PATH = "tokenizer/pythonmind_targeted_bpe.json"
CHECKPOINT_PATH = "checkpoints/pythonmind_config_d.pt"
TEST_DATA_PATH = "data/python_unseen_tests_signatures.json"

MAX_NEW_TOKENS = 80

# WARNING:
# Enabling this executes model-generated Python in the current process.
# Restricted builtins are NOT a secure sandbox. Keep False unless all
# generated code and test inputs are trusted. Use an isolated container
# with resource/network/filesystem restrictions for untrusted code.
RUN_FUNCTIONAL_TESTS = False


# ==========================================
# LOAD MODEL
# ==========================================

def load_model():

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True
    )

    if tokenizer.get_vocab_size() != checkpoint["vocab_size"]:
        raise ValueError(
            "Tokenizer vocabulary size does not match checkpoint. "
            "Use the tokenizer that trained this checkpoint."
        )

    eos_id = tokenizer.token_to_id("[EOS]")
    if eos_id is None:
        raise ValueError("Tokenizer does not contain [EOS]")

    model = PythonMindModel(
        vocab_size=checkpoint["vocab_size"],
        embedding_dim=checkpoint["embedding_dim"],
        num_heads=checkpoint["num_heads"],
        num_layers=checkpoint["num_layers"],
        max_seq_length=checkpoint["max_seq_length"]
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, tokenizer, eos_id


# ==========================================
# UNSEEN EVALUATION
# ==========================================

def evaluate_unseen():

    model, tokenizer, eos_id = load_model()

    with open(TEST_DATA_PATH, "r", encoding="utf-8") as file:
        tests = json.load(file)

    if not isinstance(tests, list) or not tests:
        raise ValueError("Test file must contain a non-empty JSON list")

    syntax_passes = 0
    functional_passes = 0
    generation_seconds = 0.0
    generated_token_count = 0

    print("\n========== PYTHONMIND CONFIG D EVALUATION ==========")
    if RUN_FUNCTIONAL_TESTS:
        print("WARNING: Functional mode executes generated Python code.")
    else:
        print("Functional execution: DISABLED (safe default)")

    for test in tests:

        required = {"name", "prompt", "function", "args", "expected"}
        missing = required.difference(test)
        if missing:
            raise ValueError(
                f"Test entry is missing fields: {sorted(missing)}"
            )

        encoded = tokenizer.encode(test["prompt"])
        input_ids = torch.tensor([encoded.ids], dtype=torch.long)

        start_time = time.perf_counter()
        output_ids = generate(
            model,
            input_ids,
            eos_id,
            max_new_tokens=MAX_NEW_TOKENS
        )
        generation_seconds += time.perf_counter() - start_time
        generated_token_count += output_ids.size(1) - input_ids.size(1)

        generated_code = tokenizer.decode(
            output_ids[0].tolist(),
            skip_special_tokens=True
        )

        print("\n" + "=" * 50)
        print("TEST:", test["name"])
        print("Generated Code:\n", generated_code)

        try:
            compiled_code = compile(
                generated_code,
                "<generated>",
                "exec"
            )
            ast.parse(generated_code)
            syntax_valid = True
        except (SyntaxError, ValueError, TypeError) as error:
            print("Syntax Error:", error)
            syntax_valid = False

        if not syntax_valid:
            print("Syntax: FAILED")
            print("Functional Test: NOT RUN")
            continue

        syntax_passes += 1
        print("Syntax: PASSED")

        if not RUN_FUNCTIONAL_TESTS:
            print("Functional Test: NOT RUN (execution disabled)")
            continue

        try:
            # This is a convenience restriction, NOT a security boundary.
            namespace = {}
            safe_builtins = {
                "sum": sum,
                "len": len,
                "range": range,
                "min": min,
                "max": max,
                "abs": abs
            }

            exec(
                compiled_code,
                {"__builtins__": safe_builtins},
                namespace
            )

            function = namespace.get(test["function"])
            if not callable(function):
                raise ValueError("Expected function was not generated")

            actual = function(*test["args"])
            functional_valid = actual == test["expected"]

            if functional_valid:
                functional_passes += 1
                print("Functional Test: PASSED")
            else:
                print("Functional Test: FAILED")
                print("Expected:", test["expected"])
                print("Actual:", actual)

        except Exception as error:
            print("Execution Error:", error)
            print("Functional Test: FAILED")

    total = len(tests)

    print("\n========== PYTHONMIND CONFIG D SUMMARY ==========")
    print("Evaluation Type: UNSEEN TASKS")
    print("Model: PYTHONMIND CONFIG D")
    print("Total Tests:", total)
    print(f"Syntax Passes: {syntax_passes}/{total}")
    print(f"Syntax Success Rate: {syntax_passes / total * 100:.2f}%")

    if RUN_FUNCTIONAL_TESTS:
        print(f"Functional Passes: {functional_passes}/{total}")
        print(
            f"Functional Success Rate: "
            f"{functional_passes / total * 100:.2f}%"
        )
    else:
        print("Functional Pass Rate: Not measured")

    print(f"Generation Time: {generation_seconds:.4f} seconds")
    print("Generated Tokens:", generated_token_count)
    print("==============================================")


if __name__ == "__main__":
    evaluate_unseen()
