""" Create a set of translation prompts.
    author: Daniel Nichols
    date: December 2023
"""
# std imports
from argparse import ArgumentParser
from glob import glob
import json
from os import PathLike
import re
from typing import List, Mapping, Optional

# tpl imports
from alive_progress import alive_it

# (dst, src)
TRANSLATION_TASKS = [
    ('serial', 'omp'),
    ('serial', 'mpi'),
    ('cuda', 'kokkos')
]

# mapping from execution model to a clean name
EXECUTION_MODEL_CLEAN_NAME_MAP = {
    "serial": "Serial",
    "omp": "OpenMP",
    "mpi": "MPI",
    "mpi+omp": "MPI+OpenMP",
    "kokkos": "Kokkos",
    "cuda": "CUDA",
    "hip": "HIP"
}

# prompt format for translation prompt
TRANSLATION_PROMPT_FORMAT = """// {src_model} implementation of {function_name}
{src_model_example}

// {dst_model} implementation of {function_name}
{dst_model_prompt}
"""

def get_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--generation-prompts", type=str, default="generation-prompts.json", help="Path to generation prompts.")
    parser.add_argument("--results-root", type=str, required=True, help="Path to results root directory.")
    parser.add_argument("-o", "--output", type=str, help="Path to output json.")
    return parser.parse_args()

def read_json(fpath: PathLike) -> dict:
    with open(fpath, "r") as fp:
        return json.load(fp)

def find_correct_implementation(prompt: dict, all_results: List[list], model: str) -> Optional[str]:
    """ Find a correct implementation from the results list to use as an example. """
    for results in all_results:
        subset = list(filter(lambda x: x['name'] == prompt['name'] and x['parallelism_model'] == model, results))
        assert len(subset) == 1, f"There should only be 1 matching result for each prompt and model. Found: {len(subset)}"

        outputs = subset[0]['outputs']
        assert len(outputs) > 0, f"There should be at least one output for each prompt and model."

        correct_outputs = list(filter(lambda x: x['are_all_valid'] == True, outputs))
        if len(correct_outputs) == 0:
            continue
        else:
            return correct_outputs[0]['generated_output']

    return None


GPU_FUNCTION_NAME_PATTERN = re.compile(r"__global__ void ([a-zA-Z0-9_]+)\(")
CPU_FUNCTION_NAME_PATTERN = re.compile(r"\s*[a-zA-Z_]+ ([a-zA-Z0-9_]+)\(")
def get_function_name(prompt: str, execution_model: str) -> str:
    if execution_model in ['cuda', 'hip']:
        match = GPU_FUNCTION_NAME_PATTERN.match(prompt.splitlines()[-1])
    else:
        match = CPU_FUNCTION_NAME_PATTERN.match(prompt.splitlines()[-1])
    if match is None:
        raise ValueError(f"Could not find function name in prompt: {prompt}")
    return match.group(1)


def prepend_to_every_line(s: str, prefix: str) -> str:
    return "\n".join(map(lambda x: prefix + x, s.splitlines()))


def main():
    args = get_args()

    # load generation prompts
    generation_prompts = read_json(args.generation_prompts)

    # load all the results
    results_jsons = glob(f"{args.results_root}/*/results.json")
    results = []
    for fpath in results_jsons:
        results.append(read_json(fpath))

    # create translation prompts
    translation_prompts = []
    for src_model, dst_model in TRANSLATION_TASKS:

        generation_prompts_for_src_model = list(filter(lambda x: x['parallelism_model'] == src_model, generation_prompts))
        generation_prompts_for_dst_model = list(filter(lambda x: x['parallelism_model'] == dst_model, generation_prompts))

        for generation_prompt in alive_it(generation_prompts_for_dst_model, title=f"{src_model} -> {dst_model}"):
            correct_impl = find_correct_implementation(generation_prompt, results, src_model)

            if correct_impl is None:
                print(f"Could not find correct implementation for {generation_prompt['name']} {src_model}")
                continue

            src_model_clean_name = EXECUTION_MODEL_CLEAN_NAME_MAP[src_model]
            dst_model_clean_name = EXECUTION_MODEL_CLEAN_NAME_MAP[dst_model]
            dst_model_prompt = generation_prompt['prompt']
            function_name = get_function_name(dst_model_prompt, dst_model)
            

            src_model_prompts = list(filter(lambda x: x['name'] == generation_prompt['name'], generation_prompts_for_src_model))
            assert len(src_model_prompts) == 1, f"There should only be one prompt with the name {generation_prompt['name']}. Found: {len(src_model_prompts)}"
            src_model_prompt = src_model_prompts[0]['prompt']
            src_model_example = prepend_to_every_line(src_model_prompt + "\n" + correct_impl, "// ")
        
            prompt = TRANSLATION_PROMPT_FORMAT.format(
                src_model=src_model_clean_name,
                src_model_example=src_model_example,
                dst_model=dst_model_clean_name,
                function_name=function_name,
                dst_model_prompt=dst_model_prompt
            )

            # create copy of generation prompt and update it
            translation_prompt = generation_prompt.copy()
            translation_prompt['translation_prompt'] = prompt
            translation_prompt['translation_src_model'] = src_model
            translation_prompt['translation_dst_model'] = dst_model
            translation_prompt['translation_src_example'] = src_model_prompt + "\n" + correct_impl
            translation_prompt['translation_function_name'] = function_name
            translation_prompts.append(translation_prompt)

    # write out translation prompts
    if args.output:
        with open(args.output, 'w') as output_json:
            json.dump(translation_prompts, output_json, indent=2)
        
        print(f"Wrote {len(translation_prompts)} translation prompts to {args.output}")
    else:
        print(json.dumps(translation_prompts, indent=2))


if __name__ == "__main__":
    main()