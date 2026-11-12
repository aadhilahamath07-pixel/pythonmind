
import ast
from collections import Counter

DATASET_PATH = "data/python_train_targeted.txt"


def validate():

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        source = file.read()

    tree = ast.parse(source)

    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
    ]

    names = [function.name for function in functions]
    duplicates = [
        name for name, count in Counter(names).items()
        if count > 1
    ]

    print("\n========== TARGETED DATASET VALIDATION ==========")

    print("Total Functions:", len(functions))
    print("Unique Functions:", len(set(names)))
    print("Duplicate Names:", duplicates)

    print("\nSyntax Validation: PASSED")

    if len(functions) != 20:
        print("ERROR: Expected 20 functions")

    if duplicates:
        print("ERROR: Duplicate function names found")

    if len(functions) == 20 and not duplicates:
        print("\nDATASET STRUCTURE: PASSED")
    else:
        print("\nDATASET STRUCTURE: FAILED")

    print("=================================================")


if __name__ == "__main__":
    validate()
