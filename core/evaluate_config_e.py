
import ast
import json
import torch
from tokenizers import Tokenizer

from core.model import PythonMindModel
from core.generate import generate


# ==========================================
# CONFIGURATION
# ==========================================

TOKENIZER_PATH = "tokenizer/pythonmind_targeted_bpe.json"

CHECKPOINT_PATH = "checkpoints/pythonmind_config_e_final.pt"
TEST_DATA_PATH = "data/python_unseen_tests_signatures.json"


# ==========================================
# LOAD MODEL
# ==========================================

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


# ==========================================
# UNSEEN EVALUATION
# ==========================================

def evaluate_unseen():

    model, tokenizer = load_model()

    with open(TEST_DATA_PATH, "r", encoding="utf-8") as file:
        tests = json.load(file)

    syntax_passes = 0
    functional_passes = 0

    print("\n========== PYTHONMIND CONFIG E EVALUATION ==========")

    for test in tests:

        encoded = tokenizer.encode(test["prompt"])

        input_ids = torch.tensor(
            [encoded.ids],
            dtype=torch.long
        )

        eos_id = tokenizer.token_to_id("[EOS]")

        output_ids = generate(
            model,
            input_ids,
            eos_id,
            max_new_tokens=80
        )

        generated_code = tokenizer.decode(
            output_ids[0].tolist(),
            skip_special_tokens=True
        )

        print("\n" + "=" * 50)
        print("TEST:", test["name"])
        print("Generated Code:\n", generated_code)

        # 1. Syntax validation
        try:
            compiled_code = compile(
                generated_code,
                "<generated>",
                "exec"
            )

            ast.parse(generated_code)
            syntax_valid = True

        except SyntaxError as error:
            print("Syntax Error:", error)
            syntax_valid = False

        if not syntax_valid:
            print("Syntax: FAILED")
            print("Functional Test: SKIPPED")
            continue

        syntax_passes += 1
        print("Syntax: PASSED")

        # 2. Functional validation
        try:
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
    print(f"Functional Passes: {functional_passes}/{total}")

    print(f"Syntax Success Rate: {syntax_passes / total * 100:.2f}%")
    print(f"Functional Success Rate: {functional_passes / total * 100:.2f}%")

    print("==============================================")


if __name__ == "__main__":
    evaluate_unseen()
