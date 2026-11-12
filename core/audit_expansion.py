
import ast
from pathlib import Path

DATA_PATH = Path("data/python_train_expansion.txt")

source = DATA_PATH.read_text(encoding="utf-8")
tree = ast.parse(source)

namespace = {}
exec(compile(tree, str(DATA_PATH), "exec"), namespace)

tests = [
    ("count_negative_numbers", ([-2, 3, -5, 0],), 2),
    ("count_even_numbers", ([1, 2, 4, 7],), 2),
    ("find_smallest_word", (["apple", "cat", "banana"],), "cat"),
    ("find_longest_word", (["apple", "cat", "banana"],), "banana"),
    ("calculate_list_product", ([2, 3, 4],), 24),
    ("remove_vowels", ("hello",), "hll"),
    ("find_first_even", ([1, 3, 4, 6],), 4),
    ("get_middle_element", ([10, 20, 30],), 20),
    ("calculate_cumulative_sum", ([1, 2, 3],), [1, 3, 6]),
    ("is_power_of_two", (16,), True),
    ("count_divisors", (6,), 4),
    ("rotate_list_right", ([1, 2, 3, 4], 1), [4, 1, 2, 3]),
    ("find_second_largest_distinct", ([5, 2, 5, 3],), 3),
    ("calculate_euclidean_distance", (0, 0, 3, 4), 5.0),
    ("calculate_weighted_sum", ([2, 3], [4, 5]), 23),
]

passed = 0

for name, args, expected in tests:
    actual = namespace[name](*args)
    success = actual == expected

    if success:
        passed += 1

    print(
        f"{'PASS' if success else 'FAIL'}: {name} "
        f"| Expected: {expected} | Actual: {actual}"
    )

print(f"\nFunctional Tests: {passed}/{len(tests)}")
