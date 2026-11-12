
import ast
import random
from pathlib import Path

SOURCE_PATH = Path("data/python_train.txt")
TRAIN_PATH = Path("data/python_train_split.txt")
VALIDATION_PATH = Path("data/python_validation.txt")

VALIDATION_RATIO = 0.20
RANDOM_SEED = 42


def split_dataset():
    text = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(text)

    functions = [
        ast.get_source_segment(text, node)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    if len(functions) != 205:
        print("Warning: Expected 205 functions, found", len(functions))

    indices = list(range(len(functions)))
    random.Random(RANDOM_SEED).shuffle(indices)

    validation_count = round(len(functions) * VALIDATION_RATIO)

    validation_indices = set(indices[:validation_count])

    train_functions = [
        function for i, function in enumerate(functions)
        if i not in validation_indices
    ]

    validation_functions = [
        function for i, function in enumerate(functions)
        if i in validation_indices
    ]

    TRAIN_PATH.write_text(
        "\n\n".join(train_functions) + "\n",
        encoding="utf-8"
    )

    VALIDATION_PATH.write_text(
        "\n\n".join(validation_functions) + "\n",
        encoding="utf-8"
    )

    print("========== DATASET SPLIT ==========")
    print("Original Functions:", len(functions))
    print("Training Functions:", len(train_functions))
    print("Validation Functions:", len(validation_functions))
    print("Random Seed:", RANDOM_SEED)
    print("Training File:", TRAIN_PATH)
    print("Validation File:", VALIDATION_PATH)


if __name__ == "__main__":
    split_dataset()
