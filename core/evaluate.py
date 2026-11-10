
import ast
import torch
from tokenizers import Tokenizer

from core.model import PythonMindModel
from core.generate import generate


TOKENIZER_PATH = "tokenizer/pythonmind_bpe.json"
CHECKPOINT_PATH = "checkpoints/pythonmind_core.pt"


TEST_CASES = [
    {
        "name": "Addition",
        "prompt": "def add(",
        "function": "add",
        "args": (5, 3),
        "expected": 8
    },
    {
        "name": "Subtraction",
        "prompt": "def subtract(",
        "function": "subtract",
        "args": (10, 4),
        "expected": 6
    },
    {
        "name": "Multiplication",
        "prompt": "def multiply(",
        "function": "multiply",
        "args": (4, 5),
        "expected": 20
    },
    {
        "name": "Square",
        "prompt": "def square(",
        "function": "square",
        "args": (6,),
        "expected": 36
    },
    {
        "name": "Even Check",
        "prompt": "def is_even(",
        "function": "is_even",
        "args": (8,),
        "expected": True
    }
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


def evaluate():

    model, tokenizer = load_model()

    syntax_passes = 0
    functional_passes = 0

    for test in TEST_CASES:

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
            max_new_tokens=30
        )

        generated_code = tokenizer.decode(
            output_ids[0].tolist(),
            skip_special_tokens=True
        )

        print("\n" + "=" * 45)
        print("TEST:", test["name"])
        print("Generated Code:")
        print(generated_code)

        # Syntax validation
        try:
            compiled_code = compile(
                generated_code,
                "<generated>",
                "exec"
            )
            syntax_valid = True

        except SyntaxError as error:
            print("Syntax Error:", error)
            syntax_valid = False

        if syntax_valid:
            syntax_passes += 1
            print("Syntax: PASSED")

            # Execute only in this local prototype evaluator.
            try:
                namespace = {}
                exec(
                    compiled_code,
                    {"__builtins__": {}},
                    namespace
                )

                function = namespace.get(test["function"])

                if callable(function):
                    actual = function(*test["args"])
                    functional_valid = actual == test["expected"]
                else:
                    functional_valid = False

                if functional_valid:
                    functional_passes += 1
                    print("Functional Test: PASSED")
                else:
                    print("Functional Test: FAILED")

            except Exception as error:
                print("Execution Error:", error)
                print("Functional Test: FAILED")

        else:
            print("Syntax: FAILED")
            print("Functional Test: SKIPPED")

    total = len(TEST_CASES)

    print("\n========== FINAL RESULTS ==========")
    print(f"Total Tests: {total}")
    print(f"Syntax Passes: {syntax_passes}/{total}")
    print(f"Functional Passes: {functional_passes}/{total}")

    print(
        f"Syntax Success Rate: {syntax_passes / total * 100:.2f}%"
    )

    print(
        f"Functional Success Rate: {functional_passes / total * 100:.2f}%"
    )

    print("===================================")


if __name__ == "__main__":
    evaluate()
