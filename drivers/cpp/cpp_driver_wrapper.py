""" Wrapper for calling c++ drivers
    author: Daniel Nichols
    date: October 2023
"""
# std imports
import logging
import os
from os import PathLike, environ
import shlex
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
    "mpi+omp": "mpi-omp-driver.o",
    "kokkos": "kokkos-driver.o",
    "cuda": "cuda-driver.o",
}

""" Compiler settings """
COMPILER_SETTINGS = {
    "serial": {"CXX": "g++", "CXXFLAGS": "-std=c++17 -O3"},
    "omp": {"CXX": "g++", "CXXFLAGS": "-std=c++17 -O3 -fopenmp"},
    "mpi": {"CXX": "mpicxx", "CXXFLAGS": "-std=c++17 -O3"},
    "mpi+omp": {"CXX": "mpicxx", "CXXFLAGS": "-std=c++17 -O3 -fopenmp"},
    "kokkos": {"CXX": "g++", "CXXFLAGS": "-std=c++17 -O3 -fopenmp -I../tpl/kokkos/build/include ../tpl/kokkos/build/lib64/libkokkoscore.a ../tpl/kokkos/build/lib64/libkokkoscontainers.a ../tpl/kokkos/build/lib64/libkokkossimd.a"},
    "cuda": {"CXX": "nvcc", "CXXFLAGS": "-std=c++17 -O3"},
}

def build_kokkos(driver_src: PathLike, output_root: PathLike):
    """ Custom steps for the Kokkos programs, since they require cmake """
    # cp cmake file into the output directory
    cmake_path = "cpp/KokkosCMakeLists.txt"
    cmake_dest = os.path.join(output_root, "CMakeLists.txt")
    run_command(f"cp {cmake_path} {cmake_dest}", dry=False)

    # run cmake and make
    pwd = os.getcwd()
    cmake_flags = f"-DKokkos_DIR=../tpl/kokkos/build -DDRIVER_PATH={pwd} -DDRIVER_SRC_FILE={driver_src}"
    cmake_out = run_command(f"cmake -B{output_root} -S{output_root} {cmake_flags}", dry=False)
    return run_command(f"make -C {output_root}", dry=False)

class CppDriverWrapper(DriverWrapper):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_driver_file = os.path.join("cpp", "models", DRIVER_MAP[self.parallelism_model])

    def write_source(self, content: str, fpath: PathLike) -> bool:
        """ Write the given c++ source to the given file. """
        with open(fpath, "w") as fp:
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
        if self.parallelism_model == "kokkos":
            driver_src = [b for b in binaries if b.endswith(".cc")][0]
            compile_process = build_kokkos(driver_src, os.path.dirname(output_path))
        else:
            binaries_str = ' '.join(binaries)
            macro = f"-DUSE_{self.parallelism_model.upper()}"
            cmd = f"{CXX} {CXXFLAGS} -Icpp -Icpp/models {macro} {binaries_str} -o {output_path}"
            compile_process = run_command(cmd, timeout=10, dry=self.dry)
        return BuildOutput(compile_process.returncode, compile_process.stdout, compile_process.stderr)

    def run(self, executable: PathLike, **run_config) -> RunOutput:
        """ Run the given executable. """
        launch_format = self.launch_configs["format"]
        launch_cmd = launch_format.format(exec_path=executable, args="", **run_config).strip()
        run_process = run_command(launch_cmd, timeout=60, dry=self.dry)
        return RunOutput(run_process.returncode, run_process.stdout, run_process.stderr, config=run_config)

    def test_single_output(self, prompt: str, output: str, test_driver_file: PathLike) -> GeneratedTextResult:
        """ Test a single generated output. """
        logging.debug(f"Testing output:\n{output}")
        with tempfile.TemporaryDirectory(dir=self.scratch_dir) as tmpdir:
            # write out the prompt + output
            src_ext = "cuh" if self.parallelism_model in ["cuda", "hip"] else "hpp"
            src_path = os.path.join(tmpdir, f"generated-code.{src_ext}")
            write_success = self.write_source(prompt+"\n"+output, src_path)
            logging.debug(f"Wrote source to {src_path}.")

            # compile and run the output
            exec_path = os.path.join(tmpdir, "a.out")
            compiler_kwargs = COMPILER_SETTINGS[self.parallelism_model]
            compiler_kwargs["CXXFLAGS"] += f" -I{tmpdir}"
            build_result = self.compile(self.model_driver_file, test_driver_file, output_path=exec_path, **compiler_kwargs)
            logging.debug(f"Build result: {build_result}")
            print(build_result.stderr)

            # run the code
            configs = self.launch_configs["params"]
            run_results = [self.run(exec_path, **c) for c in configs] if build_result.did_build else None
            logging.debug(f"Run result: {run_results}")
            if run_results:
                for run_result in run_results:
                    if run_result.exit_code != 0:
                        logging.debug(f"Ouputs:\n\tstdout: {run_result.stdout}\n\tstderr: {run_result.stderr}")
        
        return GeneratedTextResult(write_success, build_result, run_results)
