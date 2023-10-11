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
    "omp": "omp-driver.o",
}

""" Compiler settings """
COMPILER_SETTINGS = {
    "serial": {"CXX": "g++", "CXXFLAGS": "-std=c++17 -O3"},
    "omp": {"CXX": "g++", "CXXFLAGS": "-std=c++17 -O3 -fopenmp"},
}


def get_args():
    parser = ArgumentParser(description="Run all the generated code.")
    parser.add_argument("input_json", type=str, required=True help="Input JSON file containing the test cases.")
    return parser.parse_args()


def write_source(text, file):
    """ Write the given text to the given file. """
    with open(file, "w") as fp:
        fp.write(text)


def compile(*binaries, output_path="a.out", CXX="g++", CXXFLAGS="-std=c++17 -O3"):
    """ Compile the given binaries into a single executable. """
    os.system(f"{CXX} {CXXFLAGS} {' '.join(binaries)} -o {output_path}")


def run_generated_output(prompt, output, model_driver_file, test_driver_file, parallelism_model):
    """ Run a single generated output. """
    with tempfile.TemporaryDirectory() as tmpdir:
        # write out the prompt + output
        src_path = os.path.join(tmpdir, "llm-output.cc")
        write_source(prompt+output, src_path)

        # compile and run the output
        exec_path = os.path.join(tmpdir, "a.out")
        compile(src_path, model_driver_file, test_driver_file, output_path=exec_path, **COMPILER_SETTINGS[parallelism_model])

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