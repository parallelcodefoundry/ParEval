""" Run all the generated code.
    author: Daniel Nichols
    date: October 2023
"""
# std imports
from argparse import ArgumentParser
import json
import os
import tempfile


""" Map parallelism models to driver files """
DRIVER_MAP = {
    "serial": "serial-driver.o",
    "openmp": "omp-driver.o",
    "mpi": "mpi-driver.o",
}

""" Compiler settings """
COMPILER_SETTINGS = {
    "serial": {"CXX": "g++", "CXXFLAGS": "-std=c++17 -O3"},
    "openmp": {"CXX": "g++", "CXXFLAGS": "-std=c++17 -O3 -fopenmp"},
    "mpi": {"CXX": "mpicxx", "CXXFLAGS": "-std=c++17 -O3"},
}

""" Imports """
IMPORTS = {
    "serial": '#include "serial-driver.h"',
    "openmp": '#include "omp-driver.h"',
    "mpi": '#include "mpi-driver.h"',
}


def get_args():
    parser = ArgumentParser(description="Run all the generated code.")
    parser.add_argument("input_json", type=str, help="Input JSON file containing the test cases.")
    return parser.parse_args()


def write_source(text, file, parallelism_model):
    """ Write the given text to the given file. """
    includes = IMPORTS[parallelism_model]

    with open(file, "w") as fp:
        fp.write(includes + "\n\n")
        fp.write(text)


def compile(*binaries, output_path="a.out", CXX="g++", CXXFLAGS="-std=c++17 -O3"):
    """ Compile the given binaries into a single executable. """
    ret = os.system(f"{CXX} {CXXFLAGS} -Imodels {' '.join(binaries)} -o {output_path}")
    print(f"{CXX} {CXXFLAGS} -Imodels {' '.join(binaries)} -o {output_path}", ret)


def run_generated_output(prompt, output, model_driver_file, test_driver_file, parallelism_model):
    """ Run a single generated output. """

    with tempfile.TemporaryDirectory() as tmpdir:
        # write out the prompt + output
        src_path = os.path.join(tmpdir, "llm-output.cc")
        write_source(prompt+"\n"+output, src_path, parallelism_model)

        # compile and run the output
        exec_path = os.path.join(tmpdir, "a.out")
        build_success = compile(src_path, model_driver_file, test_driver_file, output_path=exec_path, **COMPILER_SETTINGS[parallelism_model])

        # run the code
    

def run_prompt(prompt):
    """ Run all the generated outputs for a single prompt. """
    parallelism_model = prompt["parallelism_model"]
    model_driver_file = os.path.join("models", DRIVER_MAP[parallelism_model])
    test_driver_filer = os.path.join("benchmarks", prompt["name"].lower() + "-driver.cc")

    # write, compile, and run each output
    for output in prompt["outputs"]:
        run_generated_output(prompt["prompt"], output, model_driver_file, test_driver_filer, parallelism_model)


def main():
    args = get_args()

    # load in the generated text
    with open(args.input_json, "r") as fp:
        data = json.load(fp)

    # run each prompt
    for prompt in data:
        run_prompt(prompt)


if __name__ == "__main__":
    main()