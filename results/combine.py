""" Helper script to combine multiple results files into one.
    author: Daniel Nichols
    date: January 2024
"""
# std imports
from argparse import ArgumentParser
import os
import sys
import json


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='+', help='Files to combine')
    parser.add_argument('-o', '--output', required=True, help='Output file')
    parser.add_argument("--check", type=str, help="path to prompts data set. If provided, will check that all prompts are present in the output file.")
    args = parser.parse_args()

    # check that all files exist
    for f in args.files:
        if not os.path.exists(f):
            print(f"File {f} does not exist.")
            sys.exit(1)

    # read in all files
    input_files = []
    for f in args.files:
        with open(f, 'r') as fp:
            input_files.append(json.load(fp))

    # combine
    results = []
    for f in input_files:
        for p in f:
            # only add if it has outputs and they are dicts
            if 'outputs' in p and len(p['outputs']) > 0 and all(isinstance(o, dict) for o in p['outputs']):
                results.append(p)

    # check that all prompts are present
    if args.check:
        with open(args.check, 'r') as fp:
            prompts = json.load(fp)
        
        # must have same length
        assert len(prompts) == len(results), f"Number of prompts in {args.check} ({len(prompts)}) does not match number of prompts in output file ({len(results)})."

        # must have the same values
        prompts_set = set([p['prompt'] for p in prompts])
        results_set = set([p['prompt'] for p in results])
        assert prompts_set == results_set, f"Prompts in {args.check} do not match prompts in output file."

        # all results have outputs and they are dicts, not strings
        for p in results:
            assert 'outputs' in p and len(p['outputs']) > 0 and all(isinstance(o, dict) for o in p['outputs']), f"Prompt {p['prompt']} in output file does not have outputs."

    # write out
    with open(args.output, 'w') as fp:
        json.dump(results, fp, indent=4)


if __name__ == '__main__':
    main()