
import ast
from collections import Counter
from pathlib import Path

DATASET_PATH = "data/python_train_combined_split.txt"

source = Path(DATASET_PATH).read_text(encoding="utf-8")

tree = ast.parse(source)

names = Counter()
operators = Counter()
returns = Counter()

for node in ast.walk(tree):

    if isinstance(node, ast.Name):
        names[node.id] += 1

    elif isinstance(node, ast.arg):
        names[node.arg] += 1

    elif isinstance(node, ast.Return):
        if node.value is not None:
            returns[ast.dump(node.value, include_attributes=False)] += 1

    elif isinstance(node, ast.Add):
        operators["Add"] += 1

    elif isinstance(node, ast.Sub):
        operators["Sub"] += 1

    elif isinstance(node, ast.Mult):
        operators["Mult"] += 1

    elif isinstance(node, ast.Div):
        operators["Div"] += 1

print("========== IDENTIFIER ANALYSIS ==========")

print("\nTop 20 identifiers:")
for name, count in names.most_common(20):
    print(f"{name:20} {count}")

print("\nOperators:")
for name, count in operators.most_common():
    print(f"{name:20} {count}")

print("\nMost common return expressions:")
for expression, count in returns.most_common(15):
    print(f"{count:4} | {expression}")
