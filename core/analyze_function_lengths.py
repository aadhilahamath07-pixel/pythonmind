
import ast
from tokenizers import Tokenizer

TOKENIZER_PATH = "tokenizer/pythonmind_combined_bpe.json"

DATASETS = {
    "Training": "data/python_train_combined_split.txt",
    "Validation": "data/python_validation_combined.txt"
}

tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

for dataset_name, file_path in DATASETS.items():

    with open(file_path, "r", encoding="utf-8") as file:
        source = file.read()

    tree = ast.parse(source)

    lengths = []

    for node in tree.body:

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):

            code = ast.get_source_segment(source, node)

            token_count = len(tokenizer.encode(code).ids) + 1

            lengths.append(token_count)

    print(f"\n========== {dataset_name} ==========")

    print("Functions:", len(lengths))
    print("Average Tokens:", round(sum(lengths) / len(lengths), 2))
    print("Shortest Function:", min(lengths))
    print("Longest Function:", max(lengths))
    print("Functions over 64 tokens:", sum(x > 64 for x in lengths))
    print("Functions over 128 tokens:", sum(x > 128 for x in lengths))

    print("================================")
