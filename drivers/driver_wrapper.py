""" Wrap driver functionality.
    author: Daniel Nichols
    date: October 2023
"""
# std imports
from abc import ABC, abstractmethod
import logging
import os
from os import PathLike
from typing import List, Optional, Tuple

# local imports
from util import all_equal, mean
from cpp.parallel_validation import Validator, OMPValidator, MPIValidator, MPIandOMPValidator, EmptyValidator


class BuildOutput:
    """ Represents the output of a single build. """
    exit_code: int
    stdout: str
    stderr: str

    def __init__(self, exit_code: int, stdout: str, stderr: str):
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.did_build = self.exit_code == 0
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(exit_code={self.exit_code}, did_build={self.did_build})"

class RunOutput:
    """ Represents the output of a single run. """
    exit_code: int
    stdout: str
    stderr: str
    config: dict

    def __init__(self, exit_code: int, stdout: str, stderr: str, config: dict = {}):
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.config = config
        self.is_valid, self.runtime, self.best_sequential_runtime = self._parse_output(stdout)
        if self.is_valid and self.runtime == 0:
            logging.warning(f"Runtime is 0 for run with config {self.config}. Try increasing the problem size.")
        if self.is_valid and self.best_sequential_runtime == 0:
            logging.warning(f"The best sequential runtime is 0 for run with config {self.config}. Try increasing the problem size.")
        if self.is_valid and self.best_sequential_runtime and self.best_sequential_runtime < 0.001:
            logging.warning(f"The best sequential runtime is very small ({self.best_sequential_runtime}) for run with config {self.config}. Try increasing the problem size.")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(exit_code={self.exit_code}, is_valid={self.is_valid}, runtime={self.runtime}, best_sequential_runtime={self.best_sequential_runtime}, config={self.config})"

    def _parse_output(self, output: str) -> Tuple[Optional[bool], Optional[float], Optional[float]]:
        """ Parse the output of a single run. 
            Output should have two lines:
                Time: <runtime>
                BestSequential: <runtime>
                Validation: <PASS|FAIL>
            This returns a tuple of (validation, runtime, best_sequential_runtime)
        """
        validation, runtime, best_sequential_runtime = None, None, None
        lines = output.split("\n")
        for line in lines:
            if line.startswith("Time:"):
                runtime = float(line.split(":")[1].strip())
            elif line.startswith("Validation:"):
                validation = line.split(":")[1].strip() == "PASS"
            elif line.startswith("BestSequential:"):
                best_sequential_runtime = float(line.split(":")[1].strip())

        return validation, runtime, best_sequential_runtime


class GeneratedTextResult:
    """ The result of running a single prompt """
    source_write_success: bool
    build_output: BuildOutput
    run_outputs: Optional[List[RunOutput]]

    def __init__(self, source_write_success: bool, build_output: BuildOutput, run_outputs: Optional[List[RunOutput]] = None):
        self.source_write_success = source_write_success
        self.build_output = build_output
        self.run_outputs = run_outputs
        
        assert self.build_output.did_build == (self.run_outputs is not None), \
            "Build output and run output must be consistent."
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(did_write={self.source_write_success}, did_build={self.did_build()}, did_run={self.did_run()}, is_valid={self.is_valid()}, runtime={self.runtime()})"
    
    def did_build(self) -> bool:
        """ Return whether the code built successfully. """
        return self.build_output.did_build
    
    def did_any_run(self) -> bool:
        """ Return whether the any of the code ran successfully. """
        return self.run_outputs is not None and any(r.exit_code == 0 for r in self.run_outputs)
    
    def did_all_run(self) -> bool:
        """ Return whether all of the code ran successfully. """
        return self.run_outputs is not None and all(r.exit_code == 0 for r in self.run_outputs)
    
    def are_any_valid(self) -> bool:
        """ Return whether the code ran successfully and the output was valid. """
        return self.did_any_run() and any(r.is_valid for r in self.run_outputs)
    
    def are_all_valid(self) -> bool:
        """ Return whether the code ran successfully and the output was valid. """
        return self.did_all_run() and all(r.is_valid for r in self.run_outputs)

    def best_sequential_runtime(self) -> Optional[float]:
        """ Return the min value for sequential runtime. """
        if self.did_build() and self.did_any_run():
            return min((r.best_sequential_runtime for r in self.run_outputs if r.best_sequential_runtime is not None), default=None)
        else:
            return None


""" LANGUAGE EXTENSIONS """
LANGUAGE_EXTENSIONS = {
    "cpp": ".cc",
    "c": ".c",
    "python": ".py",
    "fortran": ".f90",
}

""" MODEL TO DRIVER FILE BASENAME MAPPING """
DRIVER_MAP = {
    "serial": "cpu",
    "omp": "cpu",
    "mpi": "cpu",
    "mpi+omp": "cpu",
    "kokkos": "kokkos",
    "cuda": "gpu",
    "hip": "gpu"
}

""" Validators """
VALIDATORS = {
    "serial": EmptyValidator(),
    "omp": OMPValidator(),
    "mpi": MPIValidator(),
    "mpi+omp": MPIandOMPValidator(),
    "kokkos": EmptyValidator(),
    "cuda": EmptyValidator(),
    "hip": EmptyValidator()
}

class DriverWrapper(ABC):
    """ Abstract base class for driver wrappers. """

    parallelism_model: str
    validator: Validator
    scratch_dir: Optional[PathLike]
    launch_configs: dict
    problem_sizes: dict
    build_timeout: int
    run_timeout: int
    display_build_errors: bool
    display_runs: bool
    early_exit_runs: bool
    dry: bool

    def __init__(
        self, 
        parallelism_model: str = "serial", 
        launch_configs: dict = {"format": "{exec_path} {args}", "params": [{}]},
        problem_sizes: dict = {},
        scratch_dir: Optional[PathLike] = None,
        build_timeout: int = 20,
        run_timeout: int = 180,
        display_build_errors: bool = False,
        display_runs: bool = False,
        early_exit_runs: bool = False,
        dry: bool = False
    ):
        self.parallelism_model = parallelism_model
        self.validator = VALIDATORS[parallelism_model]
        self.scratch_dir = scratch_dir
        self.launch_configs = launch_configs[parallelism_model]
        self.problem_sizes = problem_sizes
        self.build_timeout = build_timeout
        self.run_timeout = run_timeout
        self.display_build_errors = display_build_errors
        self.display_runs = display_runs
        self.early_exit_runs = early_exit_runs
        self.dry = dry

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(parallelism_model={self.parallelism_model}, scratch_dir={self.scratch_dir})"

    @abstractmethod
    def write_source(self, content: str, fpath: PathLike) -> bool:
        """ Write the given text to the given file. """
        pass

    @abstractmethod
    def compile(self, *binaries: PathLike, output_path: PathLike = "a.out") -> BuildOutput:
        """ Compile the given binaries into a single executable. """
        pass

    @abstractmethod
    def run(self, executable: PathLike) -> RunOutput:
        """ Run the given executable. """
        pass

    @abstractmethod
    def test_single_output(self, prompt: str, output: str, test_driver_file: PathLike) -> GeneratedTextResult:
        """ Run a single generated output. """
        pass

    def test_all_outputs_in_prompt(self, prompt: dict) -> dict:
        """ Run all the generated outputs in the given prompt. """
        root = prompt["language"]
        type = prompt["problem_type"]
        name = prompt["name"]
        ext = LANGUAGE_EXTENSIONS[prompt["language"]]
        if root == "cpp" and self.parallelism_model in ["cuda", "hip"]:
            ext = ".cu"
        driver_root = f"{name}"
        driver_base = DRIVER_MAP[self.parallelism_model]
        test_driver_file = os.path.join(root, "benchmarks", type, driver_root, driver_base + ext)
        problem_size = self.problem_sizes.get(name, {}).get(self.parallelism_model, "(1<<18)")

        outputs = []
        logging.info(f"Testing prompt {name} with {self}...")
        for generated_output in prompt["outputs"]:
            results = self.test_single_output(prompt["prompt"], generated_output, test_driver_file, problem_size)

            outputs.append({
                "generated_output": generated_output,
                "source_write_success": results.source_write_success,
                "did_build": results.did_build(),
                "is_source_valid": self.validator.validate(generated_output),
                "did_any_run": results.did_any_run(),
                "did_all_run": results.did_all_run(),
                "are_any_valid": results.are_any_valid(),
                "are_all_valid": results.are_all_valid(),
                "best_sequential_runtime": results.best_sequential_runtime(),
                "runs": [
                    {
                        "did_run": r.exit_code == 0,
                        "is_valid": r.is_valid,
                        "runtime": r.runtime,
                        **r.config
                    } for r in results.run_outputs
                ] if results.run_outputs is not None else None
            })
        prompt["outputs"] = outputs

        # log some stats
        num_outputs = len(outputs)
        num_successful_writes = sum(1 for o in outputs if o["source_write_success"])
        num_successful_builds = sum(1 for o in outputs if o["did_build"])
        num_successful_runs = sum(1 for o in outputs if o["did_all_run"])
        num_valid_outputs = sum(1 for o in outputs if o["are_all_valid"] if o["is_source_valid"])
        #mean_runtime = mean(r["runtime"] for o in outputs if o["runs"] is not None for r in o["runs"] if r["runtime"] is not None)
        logging.info(f"Results for prompt {prompt['name']}:")
        logging.info(f"  {num_outputs} total outputs")
        logging.info(f"  {num_successful_writes} successful writes")
        logging.info(f"  {num_successful_builds} successful builds")
        logging.info(f"  {num_successful_runs} successful runs (all tests)")
        logging.info(f"  {num_valid_outputs} valid outputs (all tests)")
        #logging.info(f"  {mean_runtime} mean runtime")

        return prompt
