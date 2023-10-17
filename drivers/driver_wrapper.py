""" Wrap driver functionality.
    author: Daniel Nichols
    date: October 2023
"""
# std imports
from abc import ABC, abstractmethod
import logging
import os
from os import PathLike
from typing import Optional, Tuple


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

    def __init__(self, exit_code: int, stdout: str, stderr: str):
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.is_valid, self.runtime = self._parse_output(stdout)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(exit_code={self.exit_code}, is_valid={self.is_valid}, runtime={self.runtime})"

    def _parse_output(self, output: str) -> Tuple[Optional[bool], Optional[float]]:
        """ Parse the output of a single run. 
            Output should have two lines:
                Time: <runtime>
                Validation: <PASS|FAIL>
            This returns a tuple of (validation, runtime)
        """
        validation, runtime = None, None
        lines = output.split("\n")
        for line in lines:
            if line.startswith("Time:"):
                runtime = float(line.split(":")[1].strip())
            elif line.startswith("Validation:"):
                validation = line.split(":")[1].strip() == "PASS"
        return validation, runtime


class GeneratedTextResult:
    """ The result of running a single prompt """
    source_write_success: bool
    build_output: BuildOutput
    run_output: Optional[RunOutput]

    def __init__(self, source_write_success: bool, build_output: BuildOutput, run_output: Optional[RunOutput] = None):
        self.source_write_success = source_write_success
        self.build_output = build_output
        self.run_output = run_output
        
        assert self.build_output.did_build == (self.run_output is not None), \
            "Build output and run output must be consistent."
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(did_write={self.source_write_success}, did_build={self.did_build()}, did_run={self.did_run()}, is_valid={self.is_valid()}, runtime={self.runtime()})"
    
    def did_build(self) -> bool:
        """ Return whether the code built successfully. """
        return self.build_output.did_build
    
    def did_run(self) -> bool:
        """ Return whether the code ran successfully. """
        return self.run_output is not None and self.run_output.exit_code == 0
    
    def is_valid(self) -> bool:
        """ Return whether the code ran successfully and the output was valid. """
        return self.did_run() and self.run_output.is_valid
    
    def runtime(self) -> Optional[float]:
        """ Return the runtime of the code, if it ran successfully. """
        return self.run_output.runtime if self.is_valid() else None


""" LANGUAGE EXTENSIONS """
LANGUAGE_EXTENSIONS = {
    "cpp": ".cc",
    "c": ".c",
    "python": ".py",
    "fortran": ".f90",
}

class DriverWrapper(ABC):
    """ Abstract base class for driver wrappers. """

    parallelism_model: str

    def __init__(self, parallelism_model: str):
        self.parallelism_model = parallelism_model

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(parallelism_model={self.parallelism_model})"

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
        ext = LANGUAGE_EXTENSIONS[prompt["language"]]
        test_driver_file = os.path.join(root, "benchmarks", prompt["name"].lower() + "-driver" + ext)

        outputs = []
        logging.info(f"Testing prompt {prompt['name']} with {self}...")
        for generated_output in prompt["outputs"]:
            result = self.test_single_output(prompt["prompt"], generated_output, test_driver_file)

            outputs.append({
                "generated_output": generated_output,
                "source_write_success": result.source_write_success,
                "did_build": result.did_build(),
                "did_run": result.did_run(),
                "is_valid": result.is_valid(),
                "runtime": result.runtime(),
            })
        prompt["outputs"] = outputs

        # log some stats
        num_outputs = len(outputs)
        num_successful_writes = sum(1 for o in outputs if o["source_write_success"])
        num_successful_builds = sum(1 for o in outputs if o["did_build"])
        num_successful_runs = sum(1 for o in outputs if o["did_run"])
        num_valid_outputs = sum(1 for o in outputs if o["is_valid"])
        mean_runtime = sum(o["runtime"] for o in outputs if o["runtime"] is not None) / num_valid_outputs if num_valid_outputs > 0 else None
        logging.info(f"Results for prompt {prompt['name']}:")
        logging.info(f"  {num_outputs} total outputs")
        logging.info(f"  {num_successful_writes} successful writes")
        logging.info(f"  {num_successful_builds} successful builds")
        logging.info(f"  {num_successful_runs} successful runs")
        logging.info(f"  {num_valid_outputs} valid outputs")
        logging.info(f"  {mean_runtime} mean runtime")

        return prompt