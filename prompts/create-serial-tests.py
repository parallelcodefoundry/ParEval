""" Create a set of tests from the serial benchmarks in the drivers.
    author: Daniel Nichols
    date: November 2023
"""
# std imports
from argparse import ArgumentParser
import glob
import json
from os import PathLike
from os.path import join as path_join


def get_file_contents(fpath: PathLike) -> str:
    with open(fpath, 'r') as f:
        return f.read()

def get_substr_after_first_of(s: str, substr: str) -> str:
    """ Return the substring in s after the first instance of substr. """
    return s[s.find(substr) + len(substr):]

def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('benchmarks_root', help='Root directory of the benchmarks')
    parser.add_argument('prompts', help='Path to prompts json')
    parser.add_argument('output', help='Json output path')    
    args = parser.parse_args()

    with open(args.prompts, 'r') as f:
        prompts = json.load(f)
    
    for prompt in prompts:
        baseline_fpath = path_join(args.benchmarks_root, prompt['problem_type'], prompt['name'], 'baseline.hpp')

        if not PathLike(baseline_fpath).exists():
            continue

        baseline = get_file_contents(baseline_fpath)
        impl = get_substr_after_first_of(baseline, ') {')
        prompts['outputs'] = [impl, ' }', ' undefinedFunction(); }']

    with open(args.output, 'w') as f:
        json.dump(prompts, f, indent=4)


if __name__ == '__main__':
    main()