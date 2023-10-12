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

# local imports
from cpp.cpp_driver_wrapper import CppDriverWrapper


""" Map language names to driver wrappers """
LANGUAGE_DRIVERS = {
    "cpp": CppDriverWrapper,
}

def get_args():
    parser = ArgumentParser(description="Run all the generated code.")
    parser.add_argument("input_json", type=str, help="Input JSON file containing the test cases.")
    parser.add_argument("--log", choices=["INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"], default="INFO",
        type=str.upper, help="logging level")
    return parser.parse_args()
    

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

    # run each prompt
    for prompt in data:
        driver_cls = LANGUAGE_DRIVERS[prompt["language"]]
        driver = driver_cls(prompt["parallelism_model"])
        logging.info(f"Testing prompt {prompt['name']} with {driver}...")

        driver.test_all_outputs_in_prompt(prompt)


if __name__ == "__main__":
    main()