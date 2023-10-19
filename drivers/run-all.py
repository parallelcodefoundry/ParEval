""" Run all the generated code.
    author: Daniel Nichols
    date: October 2023
"""
# std imports
from argparse import ArgumentParser
import json
import logging
import os
import tempfile
from typing import Optional

# local imports
from driver_wrapper import DriverWrapper
from cpp.cpp_driver_wrapper import CppDriverWrapper


""" Map language names to driver wrappers """
LANGUAGE_DRIVERS = {
    "cpp": CppDriverWrapper,
}

def get_args():
    parser = ArgumentParser(description="Run all the generated code.")
    parser.add_argument("input_json", type=str, help="Input JSON file containing the test cases.")
    parser.add_argument("-o", "--output", type=str, help="Output JSON file containing the results.")
    parser.add_argument("--scratch-dir", type=str, help="If provided, put scratch files here.")
    parser.add_argument("--overwrite", action="store_true", help="If ouputs are already in DB for a given prompt, \
        then overwrite them. Default behavior is to skip existing results.")
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--exclude-models", nargs="+", type=str, choices=["serial", "omp", "mpi"], 
        help="Exclude the given parallelism models from testing.")
    model_group.add_argument("--include-models", nargs="+", type=str, choices=["serial", "omp", "mpi"],
        help="Only test the given parallelism models.")
    parser.add_argument("--log", choices=["INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"], default="INFO",
        type=str.upper, help="logging level")
    return parser.parse_args()

def get_driver(prompt: dict, scratch_dir: Optional[os.PathLike]) -> DriverWrapper:
    """ Get the language drive wrapper for this prompt """
    driver_cls = LANGUAGE_DRIVERS[prompt["language"]]
    return driver_cls(prompt["parallelism_model"], scratch_dir=scratch_dir)    

def already_has_results(prompt: dict) -> bool:
    """ Check if a prompt already has results stored in it. """
    if "outputs" not in prompt or not isinstance(prompt["outputs"], list):
        raise ValueError(f"Prompt {prompt.get('name', 'unknown')} does not have any outputs.")
    
    outputs = prompt["outputs"]
    if len(outputs) == 0 or all(isinstance(o, str) for o in outputs):
        return False

    if len(outputs) > 0 and all(isinstance(o, dict) for o in outputs):
        return True

    raise ValueError(f"Prompt {prompt.get('name', 'unknown')} has invalid outputs.")

def main():
    args = get_args()

    # setup logging
    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: {}".format(args.log))
    logging.basicConfig(format="%(asctime)s [%(levelname)s] -- %(message)s", level=numeric_level)

    # load in the generated text
    with open(args.input_json, "r") as fp:
        data = json.load(fp)
    logging.info(f"Loaded {len(data)} prompts from {args.input_json}.")

    # gather the list of parallelism models to test
    models_to_test = args.include_models if args.include_models else ["serial", "omp", "mpi"]
    if args.exclude_models:
        models_to_test = [m for m in models_to_test if m not in args.exclude_models]

    # run each prompt
    for prompt in data:
        if prompt["parallelism_model"] not in models_to_test:
            # skip if user asked not to run this parallelism model
            logging.debug(f"Skipping prompt {prompt['name']} because it uses {prompt['parallelism_model']}.")
            continue

        if already_has_results(prompt) and not args.overwrite:
            # skip if we already have results for this prompt
            logging.debug(f"Skipping prompt {prompt['name']} because it already has results.")
            continue

        driver = get_driver(prompt, args.scratch_dir)
        driver.test_all_outputs_in_prompt(prompt)

    # write out results
    if args.output and args.output != '-':
        with open(args.output, "w") as fp:
            json.dump(data, fp, indent=4)
        logging.info(f"Wrote results to {args.output}.")
    else:
        print(json.dumps(data, indent=4))

if __name__ == "__main__":
    main()