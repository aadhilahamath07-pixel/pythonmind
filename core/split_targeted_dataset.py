
import ast
import random
from pathlib import Path

SOURCE_PATH = Path("data/python_train_targeted_combined.txt")

TRAIN_PATH = Path("data/python_train_targeted_split.txt")
VALIDATION_PATH = Path("data/python_validation_targeted.txt")

VALIDATION_RATIO = 0.20
RANDOM_SEED = 42


def split_dataset():

    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)

    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
    ]

    random.seed(RANDOM_SEED)
    random.shuffle(functions)

    validation_count = round(len(functions) * VALIDATION_RATIO)

    validation_functions = functions[:validation_count]
    training_functions = functions[validation_count:]

    train_source = "\n\n".join(
        ast.unparse(function) for function in training_functions
    ) + "\n"

    validation_source = "\n\n".join(
        ast.unparse(function) for function in validation_functions
    ) + "\n"

    ast.parse(train_source)
    ast.parse(validation_source)

    train_names = {function.name for function in training_functions}
    validation_names = {function.name for function in validation_functions}

    overlap = train_names & validation_names

    if overlap:
        raise ValueError(f"Data leakage detected: {overlap}")

    TRAIN_PATH.write_text(train_source, encoding="utf-8")
    VALIDATION_PATH.write_text(validation_source, encoding="utf-8")

    print("\n========== TARGETED DATASET SPLIT ==========")
    print("Total Functions:", len(functions))
    print("Training Functions:", len(training_functions))
    print("Validation Functions:", len(validation_functions))
    print("Overlap:", len(overlap))
    print("Random Seed:", RANDOM_SEED)
    print("Split Validation: PASSED")
    print("============================================")


if __name__ == "__main__":
    split_dataset()
