
import ast
from pathlib import Path
from collections import Counter

# Dataset paths
original_path = Path("data/python_train.txt")
expansion_path = Path("data/python_train_expansion.txt")
combined_path = Path("data/python_train_combined.txt")

# Read both datasets
original_code = original_path.read_text(encoding="utf-8").strip()
expansion_code = expansion_path.read_text(encoding="utf-8").strip()

# Combine without modifying the source files
combined_code = original_code + "\n\n" + expansion_code + "\n"

# Validate Python syntax
tree = ast.parse(combined_code)

# Extract function definitions
functions = [
    node for node in tree.body
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
]

function_names = [function.name for function in functions]
name_counts = Counter(function_names)

duplicates = [
    name for name, count in name_counts.items()
    if count > 1
]

# Save combined dataset
combined_path.write_text(combined_code, encoding="utf-8")

# Results
print("\n========== PYTHONMIND DATASET VALIDATION ==========")
print("Original Functions:", len(ast.parse(original_code).body))
print("Expansion Functions:", len(ast.parse(expansion_code).body))
print("Combined Functions:", len(functions))
print("Unique Function Names:", len(set(function_names)))
print("Duplicate Function Names:", duplicates)
print("Syntax Validation: PASSED")
print("Combined Dataset Saved:", combined_path)
print("===================================================")

if duplicates:
    print("WARNING: Duplicate function names need review.")
else:
    print("DATASET AUDIT: PASSED")
