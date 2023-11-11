""" Use this script to gather the raw prompts from text files and export a 
    single json in the correct format. The script also optionally checks for
    prompt validity and can do some preprocessing on the prompts.
    
    author: Daniel Nichols
    date: October 2023
"""
# std imports
from abc import ABC, abstractmethod
from argparse import ArgumentParser
import json
import os
import re
from typing import List


class Prompt:
    def __init__(self, model: str):
        self.model = model
    
    @abstractmethod
    def check_valid(self, prompt: str):
        if prompt == "":
            raise ValueError("Prompt is empty")
        if not prompt.endswith("{"):
            raise ValueError(f"Prompt {prompt} does not end with {'{'}")
        for substr in ["Example", "input:", "output:"]:
            self.must_contain(prompt, substr)
    
    @abstractmethod
    def get_model_name_for_function_suffix(self):
        raise NotImplementedError(f"Prompt.get_model_name_for_function_suffix() not implemented for {self.__class__.__name__}")

    def must_contain(self, prompt: str, substr: str):
        if substr not in prompt:
            raise ValueError(f"Prompt '{prompt}' does not contain '{substr}'")
    
    def must_contain_all(self, prompt: str, substrs: List[str]):
        for substr in substrs:
            self.must_contain(prompt, substr)
    
    def must_not_contain(self, prompt: str, substr: str):
        if substr in prompt:
            raise ValueError(f"Prompt '{prompt}' contains '{substr}'")
    
    def must_not_contain_any(self, prompt: str, substrs: List[str]):
        for substr in substrs:
            self.must_not_contain(prompt, substr)

    def append_to_function_name(self, prompt: str, suffix: str) -> str:
        """ prompt is a c++ docstring and function header. Add a suffix to the
            function name.
        """
        lines = prompt.split("\n")
        if len(lines) < 2:
            raise ValueError(f"Prompt '{prompt}' does not have at least two lines")

        # get the function name
        header = lines[-1]
        match = re.search(r"[^\s]+\s+(\w+)\s*\(", header)
        if not match:
            raise ValueError(f"Could not find function name in '{header}'")
        
        # append the suffix
        function_name = match.group(1)
        new_function_name = function_name + suffix
        lines[-1] = header.replace(function_name, new_function_name)

        return "\n".join(lines)

    @abstractmethod
    def add_imports(self, prompt: str) -> str:
        raise NotImplementedError(f"Prompt.add_imports() not implemented for {self.__class__.__name__}")


class SerialPrompt(Prompt):
    def __init__(self):
        super().__init__("serial")
    
    def check_valid(self, prompt: str):
        super().check_valid(prompt)
    
    def get_model_name_for_function_suffix(self):
        return "" # append nothing for serial

    def add_imports(self, prompt: str) -> str:
        return prompt

class OpenMPPrompt(Prompt):
    def __init__(self):
        super().__init__("omp")
    
    def check_valid(self, prompt: str):
        super().check_valid(prompt)
        self.must_contain(prompt, "OpenMP")
    
    def get_model_name_for_function_suffix(self):
        return "OpenMP"

    def add_imports(self, prompt: str) -> str:
        return "#include <omp.h>\n\n" + prompt

class MPIPrompt(Prompt):

    def __init__(self):
        super().__init__("mpi")
    
    def check_valid(self, prompt: str):
        super().check_valid(prompt)
        self.must_contain_all(prompt, ["MPI", "initialized"])
    
    def get_model_name_for_function_suffix(self):
        return "MPI"

    def add_imports(self, prompt: str) -> str:
        return "#include <mpi.h>\n\n" + prompt

class MPIOpenMPPrompt(Prompt):

    def __init__(self):
        super().__init__("mpi+omp")
    
    def check_valid(self, prompt: str):
        super().check_valid(prompt)
        for substr in ["MPI", "OpenMP", "initialized"]:
            self.must_contain(prompt, substr)
    
    def get_model_name_for_function_suffix(self):
        return "MPIOpenMP"

    def add_imports(self, prompt: str) -> str:
        return "#include <mpi.h>\n#include <omp.h>\n\n" + prompt


class KokkosPrompt(Prompt):

    def __init__(self):
        super().__init__("kokkos")
    
    def check_valid(self, prompt: str):
        super().check_valid(prompt)
        self.must_contain_all(prompt, ["Kokkos", "initialized"])
        self.must_not_contain_any(prompt, ["MPI"])
    
    def get_model_name_for_function_suffix(self):
        return "Kokkos"

    def add_imports(self, prompt: str) -> str:
        return "#include <Kokkos_Core.hpp>\n\n" + prompt


class CUDAPrompt(Prompt):
    
    def __init__(self):
        super().__init__("cuda")
    
    def check_valid(self, prompt: str):
        super().check_valid(prompt)
        self.must_contain_all(prompt, ["CUDA", "thread", "__global__ void"])
        self.must_not_contain_any(prompt, ["MPI", "Kokkos", "std::vector", "thrust::", "device_vector", "HIP"])
    
    def get_model_name_for_function_suffix(self):
        return "CUDA"

    def add_imports(self, prompt: str) -> str:
        return prompt


class HIPPrompt(Prompt):
        
    def __init__(self):
        super().__init__("hip")
    
    def check_valid(self, prompt: str):
        super().check_valid(prompt)
        self.must_contain_all(prompt, ["AMD HIP", "thread", "__global__ void"])
        self.must_not_contain_any(prompt, ["MPI", "Kokkos", "std::vector", "thrust::", "device_vector", "CUDA"])
    
    def get_model_name_for_function_suffix(self):
        return "HIP"

    def add_imports(self, prompt: str) -> str:
        return prompt


def get_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("raw_prompts_root", type=str, help="path to root of raw prompts")
    parser.add_argument("-o", "--output", help="path to output json; defaults to stdout if not provided")
    parser.add_argument("--function-suffix", choices=["parallel", "model", "none"], default="none", help="suffix to add to function names")
    parser.add_argument("--add-imports", action="store_true", help="add imports above prompt")
    return parser.parse_args()


def parse_raw_prompt(
    type: str, 
    name: str, 
    fpath: os.PathLike, 
    function_suffix: str = "none", 
    add_imports: bool = False
) -> List[dict]:
    """ Parse the raw prompts from the text files. fpath points to a directory
        with a text prompt for each model. The text prompt should be named
        <model>.
    """
    prompt_paths = os.listdir(fpath)
    if set(prompt_paths) != {"serial", "omp", "mpi", "mpi+omp", "kokkos", "cuda", "hip"}:
        raise ValueError(f"{fpath} does not contain prompts for all models")
    
    parsers = {"serial": SerialPrompt(), "omp": OpenMPPrompt(), 
        "mpi": MPIPrompt(), "mpi+omp": MPIOpenMPPrompt(), 
        "kokkos": KokkosPrompt(), "cuda": CUDAPrompt(), "hip": HIPPrompt()}

    prompts = []
    for model in prompt_paths:
        model_path = os.path.join(fpath, model)
        if not os.path.isfile(model_path):
            raise ValueError(f"Expected {model_path} to be a file")

        # get the parser and read in the contents of the prompt
        parser = parsers[model]
        with open(model_path, "r") as fp:
            prompt = fp.read()
        
        # check the prompt is valid
        parser.check_valid(prompt)

        # add to the function name if necessary
        if function_suffix == "model":
            suffix = parser.get_model_name_for_function_suffix()
            prompt = parser.append_to_function_name(prompt, suffix)
        elif function_suffix == "parallel":
            prompt = parser.append_to_function_name(prompt, "Parallel")
        
        # add imports if necessary
        if add_imports:
            prompt = parser.add_imports(prompt)

        prompts.append({
            "problem_type": type,
            "language": "cpp",
            "name": name,
            "parallelism_model": model,
            "prompt": prompt
        })

    return prompts

def main():
    args = get_args()

    # prompts root is structured as <prompt_type>/<prompt_name>/<model>
    all_prompts = []
    for type in os.listdir(args.raw_prompts_root):
        type_path = os.path.join(args.raw_prompts_root, type)
        if not os.path.isdir(type_path):
            raise ValueError(f"Expected {type_path} to be a directory")

        for prompt_name in os.listdir(type_path):
            prompt_path = os.path.join(type_path, prompt_name)
            if not os.path.isdir(prompt_path):
                raise ValueError(f"Expected {prompt_path} to be a directory")
            
            prompts = parse_raw_prompt(type, prompt_name, prompt_path, 
                function_suffix=args.function_suffix, add_imports=args.add_imports)
            all_prompts.extend(prompts)
    

    # output
    if args.output:
        with open(args.output, "w") as fp:
            json.dump(all_prompts, fp, indent=4)
    else:
        print(json.dumps(all_prompts, indent=4))



if __name__ == '__main__':
    main()