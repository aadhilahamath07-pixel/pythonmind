
import ast

DATASET_PATH = "data/python_train_targeted.txt"

TESTS = {
    "calculate_rectangle_area": ((5, 4), 20),
    "calculate_rectangle_perimeter": ((5, 4), 18),
    "calculate_triangle_area": ((10, 6), 30),
    "calculate_sphere_volume": ((3,), 113.09733552923254),
    "calculate_midpoint": ((2, 4, 6, 8), (4, 6)),
    "dictionary_contains_key": (({"a": 10}, "a"), True),
    "dictionary_get_default": (({"a": 10}, "b", 0), 0),
    "count_dictionary_keys": (({"a": 1, "b": 2}, ["a", "c"]), 1),
    "find_missing_dictionary_keys": (({"a": 1}, ["a", "b"]), ["b"]),
    "binary_search": (([1, 3, 5, 7], 5), 2),
    "find_item_index": (([4, 8, 12], 8), 1),
    "find_first_occurrence": (([1, 2, 2, 2, 3], 2), 1),
    "find_last_occurrence": (([1, 2, 2, 2, 3], 2), 3),
    "calculate_percentage_change": ((100, 120), 20),
    "calculate_part_percentage": ((25, 100), 25),
    "calculate_average_change": (([10, 15, 20],), 5),
    "clamp_number": ((15, 0, 10), 10),
    "find_common_dictionary_keys": (({"a": 1, "b": 2}, {"b": 3}), ["b"]),
    "calculate_weighted_percentage": (([80, 90], [1, 1]), 85),
    "find_second_occurrence": (([4, 2, 4, 4], 4), 2),
}


def run_tests():

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        source = file.read()

    namespace = {}
    exec(compile(source, DATASET_PATH, "exec"), namespace)

    passed = 0
    failed = 0

    print("\n========== FUNCTIONAL TEST RESULTS ==========")

    for name, (arguments, expected) in TESTS.items():

        try:
            actual = namespace[name](*arguments)

            if actual == expected:
                passed += 1
                print(f"PASS: {name}")
            else:
                failed += 1
                print(f"FAIL: {name} | Expected: {expected} | Actual: {actual}")

        except Exception as error:
            failed += 1
            print(f"ERROR: {name} | {error}")

    print("\n========== SUMMARY ==========")
    print("Passed:", passed)
    print("Failed:", failed)
    print("Total:", len(TESTS))


if __name__ == "__main__":
    run_tests()
