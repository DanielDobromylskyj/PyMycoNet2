import os
import importlib
import pymyconet2.tests


class bcolors:
    HEADER = '\033[95m'
    PASS = '\033[92m'
    FAIL = '\033[91m'
    NO_TEST = '\033[96m'
    ENDC = '\033[0m'


def perform_tests():
    """ Runs all the tests, then displays them in a colour grid"""
    print("Compiling Tests...", end="")
    test_dir = pymyconet2.tests.__path__[0]

    tests = {
        file.removesuffix(".py"): [
            name

            for name in vars(importlib.import_module(
                f"{pymyconet2.tests.__name__}.{file.removesuffix('.py')}"
            ))
            if name.startswith("test_")
        ]
        for file in os.listdir(test_dir)
        if file.endswith(".py") and not file.startswith("__")
    }

    test_results = {}

    progress_bar_chunks = 40
    for j, sub_category in enumerate(tests):
        tests_in_cat = len(tests[sub_category])
        test_results[sub_category] = []

        imp = importlib.import_module(
                f"{pymyconet2.tests.__name__}.{sub_category}"
            )

        for i, test_name in enumerate(tests[sub_category]):
            try:
                result = getattr(imp, test_name)()
            except Exception as e:
                result = [False, f"Test Error -> {e}"]

            test_results[sub_category].append((*result, test_name))

            print(
                f"\r{sub_category} | {round(((i + 1) / tests_in_cat) * 100)}% | {'#' * round((j / len(tests) * progress_bar_chunks))}{' ' * (progress_bar_chunks - round((j / len(tests) * progress_bar_chunks)))} |",
                end="")

    print(f"\r{sub_category} | {round(((i + 1) / tests_in_cat) * 100)}% | {'#' * progress_bar_chunks} |", end="")

    print("\n")

    max_results = max([len(tests[sub_category]) for sub_category in tests])
    max_label_width = max(max([len(sub_category) for sub_category in tests]), 8) + 1

    spacing = ' '
    print(
        f"Category{spacing * (max_label_width - 8)}| {' | '.join([f'Test {i + 1}{spacing * (len(str(max_results)) - len(str(i + 1)))}' for i in range(max_results)])} |")
    print(
        f"{'-' * max_label_width}+-{'-+-'.join(['-----' + ('-' * len(str(max_results))) for i in range(max_results)])}-+")
    for sub_category in tests:
        print(
            f"{sub_category}{spacing * (max_label_width - len(sub_category))}| {' | '.join([(f' {bcolors.PASS}PASS{bcolors.ENDC}' if result[0] is True else f' {bcolors.FAIL}FAIL{bcolors.ENDC}') + ' ' * len(str(max_results)) for result in test_results[sub_category]])} |",
            end="")
        if max_results - len(test_results[sub_category]) != 0:
            print(" " + " | ".join([f" {bcolors.NO_TEST}None{bcolors.ENDC} " for i in
                                    range(max_results - len(test_results[sub_category]))]) + " |")
        else:
            print("")

        print(
            f"{'-' * max_label_width}+-{'-+-'.join(['-----' + ('-' * len(str(max_results))) for i in range(max_results)])}-+")

    print("\n")  # 2 new lines

    all_failures = [(sub_category, i, result[1]) for i, result in enumerate(test_results[sub_category]) for sub_category
                    in test_results if result[0] is False]

    if all_failures:
        print(f"{bcolors.FAIL}>> Failure Debug Output <<{bcolors.ENDC}\n")

        for sub_category in tests:
            failures = [(result[2], result[1]) for i, result in enumerate(test_results[sub_category]) if result[0] is False]

            if failures:
                print(f"{bcolors.NO_TEST}> {sub_category}{bcolors.ENDC}")

                for test_index, failure in failures:
                    print(f"{test_index.replace('_', ' ').removeprefix('test ')}: {failure}")

                print("\n")

    else:
        print(f"{bcolors.FAIL}> No tests have failed, this means you need more tests.{bcolors.ENDC}\n")


if __name__ == "__main__":
    perform_tests()