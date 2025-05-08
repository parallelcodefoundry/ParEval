""" Checks if the expected test results are present in the output JSON file.
    usage: python test/validate-test-results.py \
        --results <results.json> \
        --problem <problem_name> \
        --expected-write <expected_write_count> \
        --expected-source-valid <expected_source_valid_count> \
        --expected-build <expected_build_count> \
        --expected-run <expected_run_count> \
        --expected-correct <expected_correct_count>
"""
from argparse import ArgumentParser
import json
from collections import Counter


def parse_args():
    parser = ArgumentParser(description="Validate test results.")
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to the results JSON file.",
    )
    parser.add_argument(
        "--problem",
        type=str,
        required=True,
        help="Name of the problem to validate.",
    )
    parser.add_argument(
        "--expected-write",
        type=int,
        required=True,
        help="Expected number of write operations.",
    )
    parser.add_argument(
        "--expected-source-valid",
        type=int,
        required=True,
        help="Expected number of source valid operations.",
    )
    parser.add_argument(
        "--expected-build",
        type=int,
        required=True,
        help="Expected number of build operations.",
    )
    parser.add_argument(
        "--expected-run",
        type=int,
        required=True,
        help="Expected number of run operations.",
    )
    parser.add_argument(
        "--expected-correct",
        type=int,
        required=True,
        help="Expected number of correct operations.",
    )

    return parser.parse_args()


def validate_outputs(outputs, expected_counts):
    actual_counts = Counter()

    for output in outputs:
        if output.get("source_write_success", False):
            actual_counts["write"] += 1
        if output.get("is_source_valid", False):
            actual_counts["source_valid"] += 1
        if output.get("did_build", False):
            actual_counts["build"] += 1
        if output.get("did_all_run", False):
            actual_counts["run"] += 1
        if output.get("are_all_valid", False):
            actual_counts["correct"] += 1

    for key, expected in expected_counts.items():
        actual = actual_counts[key]
        if actual != expected:
            print(f"Expected {expected} for {key}, but got {actual}.")
            return False
    return True


def main():
    args = parse_args()

    # Load the results JSON file
    with open(args.results, "r") as f:
        results = json.load(f)

    # Validate the results
    expected_counts = {
        "write": args.expected_write,
        "source_valid": args.expected_source_valid,
        "build": args.expected_build,
        "run": args.expected_run,
        "correct": args.expected_correct,
    }

    results = [r for r in results if r["name"] == args.problem][0]

    if not validate_outputs(results["outputs"], expected_counts):
        print(f"Validation failed for problem {args.problem}.")
        return 1


if __name__ == "__main__":
    main()
    