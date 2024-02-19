""" A helper script to gather correct outputs for each of the prompts.
    author: Daniel Nichols
    date: December 2024
"""
# std imports
from argparse import ArgumentParser
from glob import glob
import json
from typing import List, Optional

# tpl imports
from alive_progress import alive_it


def get_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--generation-prompts", type=str, default="generation-prompts.json", help="Path to generation prompts.")
    parser.add_argument("--results-root", type=str, required=True, help="Path to results root directory.")
    parser.add_argument("-o", "--output", type=str, help="Path to output json.")
    return parser.parse_args()


def read_json(fpath) -> dict:
    with open(fpath, "r") as fp:
        return json.load(fp)


def find_correct_implementation(prompt: dict, all_results: List[list]) -> Optional[str]:
    for results in all_results:
        subset = list(filter(lambda x: x['name'] == prompt['name'] and x['parallelism_model'] == prompt['parallelism_model'], results))
        assert len(subset) == 1, f"There should only be 1 matching result for each prompt and model. Found: {len(subset)}"

        outputs = subset[0]['outputs']
        assert len(outputs) > 0, f"There should be at least one output for each prompt and model."

        correct_outputs = list(filter(lambda x: x['are_all_valid'] == True, outputs))
        if len(correct_outputs) == 0:
            continue
        else:
            return correct_outputs[0]['generated_output']

    return None


def main():
    args = get_args()

    # read in generation prompts
    prompts = read_json(args.generation_prompts)

    # load all the results
    results_jsons = glob(f"{args.results_root}/*/results.json")
    results = []
    for fpath in results_jsons:
        results.append(read_json(fpath))

    for prompt in alive_it(prompts, title="Finding correct outputs for each prompt"):
        correct_output = find_correct_implementation(prompt, results)
        if correct_output is None:
            print(f"Could not find a correct output for prompt: {prompt['name']}.{prompt['parallelism_model']}")

        prompt["correct_output"] = correct_output

    with open(args.output, "w") as fp:
        json.dump(prompts, fp, indent=4)


if __name__ == "__main__":
    main()