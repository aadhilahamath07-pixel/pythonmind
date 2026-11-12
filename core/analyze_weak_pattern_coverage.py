
import ast

TRAIN_PATH = "data/python_train_combined_split.txt"

PATTERNS = {
    "Multiplication": ast.Mult,
    "Power": ast.Pow,
    "Membership": ast.In,
    "While Loop": ast.While,
}


def analyze():

    with open(TRAIN_PATH, "r", encoding="utf-8") as file:
        source = file.read()

    tree = ast.parse(source)

    print("\n========== RELATED TRAINING EXAMPLES ==========")

    for pattern_name, pattern_type in PATTERNS.items():

        print(f"\n--- {pattern_name} ---")

        matches = []

        for function in tree.body:

            if not isinstance(function, ast.FunctionDef):
                continue

            if any(
                isinstance(node, pattern_type)
                for node in ast.walk(function)
            ):
                matches.append(function)

        print("Count:", len(matches))

        for function in matches:
            print("\n" + ast.unparse(function))

    print("\n===============================================")


if __name__ == "__main__":
    analyze()
