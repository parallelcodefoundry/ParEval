""" Get the model outputs from Google's AI api.
    author: Daniel Nichols
    date: February 2024
"""
# std imports
from argparse import ArgumentParser
import json
import os
import re
import time
from typing import Optional

# tpl imports
from alive_progress import alive_bar
import google.generativeai as genai

""" Prompt template: """
SYSTEM_TEMPLATE = """You are a helpful coding assistant.
You are helping a programmer write a C++ function. Write the body of the function and put it in a markdown code block.
Do not write any other code or explanations.
"""

PROMPT_TEMPLATE = """Complete the C++ function {function_name}. Only write the body of the function {function_name}.

```cpp
{prompt}
```
"""


def get_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--model", choices=["gemini-1.0-pro"], required=True, help="The model to use.")
    parser.add_argument("-p", "--prompts", type=str, required=True, help="Path to prompts json")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to output json")
    parser.add_argument("--api-key", type=str, help="Google AI API key. " +
        "If not provided, then uses environment variable GOOGLE_API_KEY.")
    parser.add_argument("--max-requests", type=int, help="If provided, then only makes this many requests.")
    parser.add_argument("--max-tokens-per-second", help="Limit the rate of token generation.")
    parser.add_argument("--max-requests-per-second", help="Limit the rate of request generation.")
    parser.add_argument("--dry", action="store_true", help="If provided, then don't make any requests.")
    parser.add_argument("--overwrite", action="store_true", help="If provided, then overwrite outputs already in file.")
    parser.add_argument("--temperature", type=float, default=0.2, help="The temperature to use for sampling.")
    parser.add_argument("--top-p", type=float, default=0.95, help="The top p to use for sampling.")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="The maximum number of tokens to generate.")
    parser.add_argument("--num-samples-per-prompt", type=int, default=20, help="The number of samples to generate " +
        "per prompt.")
    return parser.parse_args()


def get_env_var(name: str) -> str:
    """ Get an environment variable. """
    if name not in os.environ:
        raise ValueError(f"Environment variable {name} not set.")
    return os.environ[name]

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

def get_max_tokens_per_second(model: str) -> Optional[int]:
    """ rates limites as of January 2024 """
    if model == "gemini-1.0-pro":
        tokens_per_minute = 2048 * 60
        return tokens_per_minute / 60
    else:
        return None
    
def get_max_requests_per_second(model: str) -> Optional[int]:
    """ rates limites as of January 2024 """
    if model == "gemini-1.0-pro":
        requests_per_minute = 60
        return requests_per_minute / 60
    else:
        return None

def get_max_requests_per_day(model: str) -> Optional[int]:
    """ rates limites as of January 2024 """
    if model == "gemini-1.0-pro":
        return 60 * 60 * 24
    else:
        return None

def postprocess(prompt: str, output: str) -> str:
    """ Postprocess the output. """
    # remove leading ```, ```cpp, and trailing ```
    output = output.strip().removeprefix("```cpp").removeprefix("```").removesuffix("```")

    # remove prompt if it included it
    if output.startswith(prompt):
        output = output[len(prompt):]

    return output

def main():
    args = get_args()

    # get the prompts
    with open(args.prompts, 'r') as prompts_json:
        prompts = json.load(prompts_json)

    # read in outputs
    if not args.overwrite and os.path.exists(args.output):
        with open(args.output, 'r') as output_json:
            outputs = json.load(output_json)

        # copy existing outputs into prompts
        copy_count = 0
        for prompt in prompts:
            for o in outputs:
                if o["prompt"] == prompt["prompt"] and \
                   o["name"] == prompt["name"] and \
                   o["parallelism_model"] == prompt["parallelism_model"] and \
                   "outputs" in o and \
                   len(o["outputs"]) == args.num_samples_per_prompt and \
                   o["temperature"] == args.temperature and \
                   o["top_p"] == args.top_p:
                    for col in ["temperature", "top_p", "do_sample", "max_new_tokens", "outputs"]:
                        prompt[col] = o[col]
                    copy_count += 1
                    break
        print(f"Copied {copy_count} existing outputs.")

    # get the keys
    api_key = args.api_key or get_env_var("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)

    # create the client
    config = genai.types.GenerationConfig(
        candidate_count=1,
        max_output_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )
    safety_settings = [
        {
            "category": "HARM_CATEGORY_DANGEROUS",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE",
        },
    ]
    model = genai.GenerativeModel(args.model, generation_config=config, safety_settings=safety_settings)

    # generation metadata
    MAX_TOKENS_PER_SECOND = args.max_tokens_per_second or get_max_tokens_per_second(args.model)
    MAX_REQUESTS_PER_SECOND = args.max_requests_per_second or get_max_requests_per_second(args.model)
    MAX_REQUESTS = args.max_requests or get_max_requests_per_day(args.model)

    # generate outputs
    request_counter = 0
    request_rate_counter = 0
    request_timer = time.time()
    with alive_bar(len(prompts), title="Generating outputs", dual_line=True) as bar:
        for prompt in prompts:
            # see if we can skip this
            if not args.overwrite and "outputs" in prompt:
                bar(skipped=True)
                continue

            # get the prompt
            original_prompt = prompt["prompt"]
            function_name = get_function_name(original_prompt, prompt["parallelism_model"])
            prompt_text = PROMPT_TEMPLATE.format(prompt=original_prompt, function_name=function_name)

            # generate the outputs
            if args.dry:
                print("system", SYSTEM_TEMPLATE)
                print("prompt", prompt_text)
                continue

            # set metadata
            prompt["temperature"] = args.temperature
            prompt["top_p"] = args.top_p
            prompt["do_sample"] = True
            prompt["max_new_tokens"] = args.max_new_tokens

            # generate the outputs
            completions = []
            while len(completions) < args.num_samples_per_prompt:
                completion = model.generate_content(SYSTEM_TEMPLATE + "\n" + prompt_text)
                if completion.candidates[0].finish_reason == 1: # STOP
                    completions.append(completion)
                    bar.text(f"~> Received output {len(completions)} of {args.num_samples_per_prompt}.")
                else:
                    print(f"Got a completion with finish_reason={completion.candidates[0].finish_reason}.")
                    time.sleep(5)

            outputs = [c.text for c in completions]
            outputs = [postprocess(original_prompt, o) for o in outputs]
            prompt["outputs"] = outputs
            bar()

            # update counters
            request_counter += 1
            request_rate_counter += 1

            # check if we should stop
            if MAX_REQUESTS is not None and request_counter >= MAX_REQUESTS:
                print(f"Stopping after {request_counter} requests.")
                break
        
            # check if we should sleep
            requests_per_second = request_rate_counter / (time.time() - request_timer)
            if MAX_REQUESTS_PER_SECOND is not None and requests_per_second > (MAX_REQUESTS_PER_SECOND*0.95):
                sleep_time = 5
                print(f"Sleeping for {sleep_time} seconds.")
                time.sleep(sleep_time)
                request_timer = time.time()
                request_rate_counter = 0

            # write intermediate outputs
            with open(args.output, 'w') as output_json:
                json.dump(prompts, output_json, indent=2)

    # summary stats
    print(f"Submitted {request_counter} requests.")

    # write outputs
    with open(args.output, 'w') as output_json:
        json.dump(prompts, output_json, indent=2)
    

if __name__ == "__main__":
    main()