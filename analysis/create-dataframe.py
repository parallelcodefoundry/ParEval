""" Create a dataframe from the results of the run-all.py script.
"""
# std imports
import argparse
import json

# third party imports
import pandas as pd

def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_json", type=str, help="Input JSON file containing the test cases.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output csv file containing the results.")
    return parser.parse_args()

def has_outputs(prompt: dict) -> bool:
    """ Check if a prompt has outputs """
    if "outputs" not in prompt:
        return False
    
    if not isinstance(prompt["outputs"], list) or len(prompt["outputs"]) == 0:
        return False
    
    if all(isinstance(o, str) for o in prompt["outputs"]):
        return False
    
    return all(isinstance(o, dict) for o in prompt["outputs"])


def check(df: pd.DataFrame):
    # check for (name, parallelism_model) pairs that have zero successful builds
    agg = df.groupby(["name", "parallelism_model"]).agg({"did_build": "sum"})
    agg = agg[agg["did_build"] == 0]
    if len(agg) > 0:
        print("The following (name, parallelism_model) pairs have zero successful builds:")
        print(agg)

def main():
    args = get_args()

    # read in input
    with open(args.input_json, "r") as f:
        input_json = json.load(f)

    # filter out entries without outputs
    input_json = list(filter(lambda x: has_outputs(x), input_json))

    # parse out rows; each run becomes a row
    rows = []
    for prompt in input_json:
        for output_idx, output in enumerate(prompt["outputs"]):
            if output["runs"] is None:
                row = {
                    "prompt": prompt["prompt"],
                    "name": prompt["name"],
                    "problem_type": prompt["problem_type"],
                    "language": prompt["language"],
                    "parallelism_model": prompt["parallelism_model"],
                    "temperature": prompt["temperature"],
                    "top_p": prompt["top_p"],
                    "do_sample": prompt["do_sample"],
                    "max_new_tokens": prompt["max_new_tokens"],
                    "prompted": prompt.get("prompted", False),
                    "generated_output": output["generated_output"],
                    "did_build": output["did_build"],
                    "is_source_valid": output["is_source_valid"],
                    "best_sequential_runtime": output["best_sequential_runtime"],
                    "output_idx": output_idx
                }
                rows.append(row)
                continue

            for run_idx, run in enumerate(output["runs"]):
                row = {
                    "prompt": prompt["prompt"],
                    "name": prompt["name"],
                    "problem_type": prompt["problem_type"],
                    "language": prompt["language"],
                    "parallelism_model": prompt["parallelism_model"],
                    "temperature": prompt["temperature"],
                    "top_p": prompt["top_p"],
                    "do_sample": prompt["do_sample"],
                    "max_new_tokens": prompt["max_new_tokens"],
                    "prompted": prompt.get("prompted", False),
                    "generated_output": output["generated_output"],
                    "did_build": output["did_build"],
                    "is_source_valid": output["is_source_valid"],
                    "best_sequential_runtime": output["best_sequential_runtime"],
                    "output_idx": output_idx,
                    "run_idx": run_idx,
                    **run
                }
                rows.append(row)
    
    # create dataframe
    df = pd.DataFrame(rows)

    # check for some possible data issues
    check(df)

    # write to csv
    df.prompt = df.prompt.apply(lambda x: x.replace("\n", "\\n"))
    df.generated_output = df.generated_output.apply(lambda x: x.replace("\n", "\\n"))
    df.to_csv(args.output, index=False)

if __name__ == "__main__":
    main()