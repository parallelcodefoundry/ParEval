""" Wrapper for calling c++ drivers
    author: Daniel Nichols
    date: October 2023
"""
# std imports
import logging
import os
from os import PathLike
import shlex
import subprocess
import sys
import tempfile
from typing import List

# local imports
sys.path.append("..")
from drivers.driver_wrapper import DriverWrapper, BuildOutput, RunOutput, GeneratedTextResult
from util import run_command

""" Map parallelism models to driver files """
DRIVER_MAP = {
    "serial": "serial-driver.o",
    "omp": "omp-driver.o",
    "mpi": "mpi-driver.o",
}

""" Compiler settings """
COMPILER_SETTINGS = {
    "serial": {"CXX": "g++", "CXXFLAGS": "-std=c++17 -O3"},
    "omp": {"CXX": "g++", "CXXFLAGS": "-std=c++17 -O3 -fopenmp"},
    "mpi": {"CXX": "mpicxx", "CXXFLAGS": "-std=c++17 -O3"},
}

""" Imports """
IMPORTS = {
    "serial": '#include "serial-driver.h"',
    "omp": '#include "omp-driver.h"',
    "mpi": '#include "mpi-driver.h"',
}

class CppDriverWrapper(DriverWrapper):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_driver_file = os.path.join("cpp", "models", DRIVER_MAP[self.parallelism_model])

    def write_source(self, content: str, fpath: PathLike) -> bool:
        """ Write the given c++ source to the given file. """
        includes = IMPORTS[self.parallelism_model]

        with open(fpath, "w") as fp:
            fp.write(includes + "\n\n")
            fp.write(content)
        return True

    def compile(
        self, 
        *binaries: PathLike, 
        output_path: PathLike = "a.out", 
        CXX: str = "g++", 
        CXXFLAGS: str = "-std=c++17 -O3"
    ) -> BuildOutput:
        """ Compile the given binaries into a single executable. """
        binaries_str = ' '.join(binaries)
        cmd = f"{CXX} {CXXFLAGS} -Icpp/models {binaries_str} -o {output_path}"

        # let subprocess errors propagate up
        compile_process = run_command(cmd, timeout=10, dry=self.dry)
        return BuildOutput(compile_process.returncode, compile_process.stdout, compile_process.stderr)

    def run(self, executable: PathLike, **run_config) -> RunOutput:
        """ Run the given executable. """
        launch_format = self.launch_configs["format"]
        launch_cmd = launch_format.format(exec_path=executable, args="", **run_config).strip()
        run_process = run_command(launch_cmd, timeout=30, dry=self.dry)
        return RunOutput(run_process.returncode, run_process.stdout, run_process.stderr, config=run_config)

    def test_single_output(self, prompt: str, output: str, test_driver_file: PathLike) -> GeneratedTextResult:
        """ Test a single generated output. """
        logging.debug(f"Testing output:\n{output}")
        with tempfile.TemporaryDirectory(dir=self.scratch_dir) as tmpdir:
            # write out the prompt + output
            src_path = os.path.join(tmpdir, "llm-output.cc")
            write_success = self.write_source(prompt+"\n"+output, src_path)
            logging.debug(f"Wrote source to {src_path}.")

            # compile and run the output
            exec_path = os.path.join(tmpdir, "a.out")
            compiler_kwargs = COMPILER_SETTINGS[self.parallelism_model]
            build_result = self.compile(src_path, self.model_driver_file, test_driver_file, output_path=exec_path, **compiler_kwargs)
            logging.debug(f"Build result: {build_result}")

            # run the code
            configs = self.launch_configs["params"]
            run_results = [self.run(exec_path, **c) for c in configs] if build_result.did_build else None
            logging.debug(f"Run result: {run_results}")
            if run_results:
                for run_result in run_results:
                    if run_result.exit_code != 0:
                        logging.debug(f"Ouputs:\n\tstdout: {run_result.stdout}\n\tstderr: {run_result.stderr}")
        
        return GeneratedTextResult(write_success, build_result, run_results)
