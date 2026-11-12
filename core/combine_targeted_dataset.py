
import ast
from pathlib import Path

BASE_PATH = Path("data/python_train_combined_split.txt")
TARGET_PATH = Path("data/python_train_targeted.txt")
OUTPUT_PATH = Path("data/python_train_targeted_combined.txt")


def combine():

    base_source = BASE_PATH.read_text(encoding="utf-8")
    target_source = TARGET_PATH.read_text(encoding="utf-8")

    base_tree = ast.parse(base_source)
    target_tree = ast.parse(target_source)

    base_names = {
        node.name
        for node in base_tree.body
        if isinstance(node, ast.FunctionDef)
    }

    target_names = {
        node.name
        for node in target_tree.body
        if isinstance(node, ast.FunctionDef)
    }

    duplicates = base_names & target_names

    if duplicates:
        print("Duplicate functions:", sorted(duplicates))
        return

    combined = base_source.rstrip() + "\n\n" + target_source.strip() + "\n"

    ast.parse(combined)

    OUTPUT_PATH.write_text(combined, encoding="utf-8")

    print("\n========== TARGETED DATASET COMBINATION ==========")
    print("Original Training Functions:", len(base_names))
    print("Targeted Functions:", len(target_names))
    print("Combined Functions:", len(base_names) + len(target_names))
    print("Duplicate Names:", len(duplicates))
    print("Syntax Validation: PASSED")
    print("Saved:", OUTPUT_PATH)
    print("==================================================")


if __name__ == "__main__":
    combine()
