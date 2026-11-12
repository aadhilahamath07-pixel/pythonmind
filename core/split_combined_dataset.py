
import ast
import random
from pathlib import Path

SOURCE_PATH = Path("data/python_train_combined.txt")

TRAIN_PATH = Path("data/python_train_combined_split.txt")
VALIDATION_PATH = Path("data/python_validation_combined.txt")

VALIDATION_RATIO = 0.20
RANDOM_SEED = 42


def split_dataset():
    text = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(text)

    functions = [
        ast.get_source_segment(text, node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    ]

    if len(functions) != 245:
        raise ValueError(
            f"Expected 245 functions, found {len(functions)}"
        )

    indices = list(range(len(functions)))
    random.Random(RANDOM_SEED).shuffle(indices)

    validation_count = round(
        len(functions) * VALIDATION_RATIO
    )

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

    # Verify split integrity
    train_names = {
        node.name
        for node in ast.parse(TRAIN_PATH.read_text()).body
        if isinstance(node, ast.FunctionDef)
    }

    validation_names = {
        node.name
        for node in ast.parse(VALIDATION_PATH.read_text()).body
        if isinstance(node, ast.FunctionDef)
    }

    overlap = train_names.intersection(validation_names)

    assert not overlap, f"Function overlap detected: {overlap}"
    assert len(train_names) + len(validation_names) == 245

    print("\n========== COMBINED DATASET SPLIT ==========")
    print("Original Functions:", len(functions))
    print("Training Functions:", len(train_functions))
    print("Validation Functions:", len(validation_functions))
    print("Overlap:", len(overlap))
    print("Random Seed:", RANDOM_SEED)
    print("Split Validation: PASSED")
    print("Training File:", TRAIN_PATH)
    print("Validation File:", VALIDATION_PATH)
    print("============================================")


if __name__ == "__main__":
    split_dataset()
