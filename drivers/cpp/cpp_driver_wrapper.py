""" Wrapper for calling c++ drivers
    author: Daniel Nichols
    date: October 2023
"""
# std imports
import os
import sys
import tempfile

# local imports
sys.path.append("..")
from drivers.driver_wrapper import DriverWrapper


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

class CppDriverWrapper(DriverWrapper):

    def __init__(self, parallelism_model):
        super().__init__(parallelism_model)
        self.model_driver_file = os.path.join("cpp", "models", DRIVER_MAP[parallelism_model])

    def write_source(self, content, fpath):
        """ Write the given c++ source to the given file. """
        includes = IMPORTS[self.parallelism_model]

        with open(fpath, "w") as fp:
            fp.write(includes + "\n\n")
            fp.write(content)

    def compile(self, *binaries, output_path="a.out", CXX="g++", CXXFLAGS="-std=c++17 -O3"):
        """ Compile the given binaries into a single executable. """
        ret = os.system(f"{CXX} {CXXFLAGS} -Icpp/models {' '.join(binaries)} -o {output_path}")
        print(f"{CXX} {CXXFLAGS} -Icpp/models {' '.join(binaries)} -o {output_path}", ret)

    def run(self, executable):
        """ Run the given executable. """
        os.system(executable)

    def test_single_output(self, prompt, output, test_driver_file):
        """ Test a single generated output. """
        with tempfile.TemporaryDirectory() as tmpdir:
            # write out the prompt + output
            src_path = os.path.join(tmpdir, "llm-output.cc")
            self.write_source(prompt+"\n"+output, src_path)

            # compile and run the output
            exec_path = os.path.join(tmpdir, "a.out")
            compiler_kwargs = COMPILER_SETTINGS[self.parallelism_model]
            self.compile(src_path, self.model_driver_file, test_driver_file, output_path=exec_path, **compiler_kwargs)

            # run the code
            self.run(exec_path)

    def test_all_outputs_in_prompt(self, prompt):
        """ Test all the generated outputs in the given prompt. """
        test_driver_file = os.path.join("cpp", "benchmarks", prompt["name"].lower() + "-driver.cc")

        # write, compile, and run each output
        for output in prompt["outputs"]:
            self.test_single_output(prompt["prompt"], output, test_driver_file)
