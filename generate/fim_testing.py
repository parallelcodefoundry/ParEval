# std imports
import argparse
import json
import time
from tqdm import tqdm
import sys
import random
import re

# tpl imports
import torch
from transformers import pipeline

# local imports
from utils import BalancedBracketsCriteria, PromptDataset, clean_output, get_inference_config


""" Parse command line arguments """
parser = argparse.ArgumentParser(description='Generate code')
parser.add_argument('--prompts', required=True, help='Path to the prompt JSON file')
parser.add_argument('--model', required=True, help='Path to the language model')
parser.add_argument('--output', required=True, help='Path to the output JSON file')
parser.add_argument('--max_new_tokens', type=int, default=1024, help='Maximum number of new tokens to generate (default: 1024)')
parser.add_argument('--num_samples_per_prompt', type=int, default=50, help='Number of code samples to generate (default: 50)')
parser.add_argument('--temperature', type=float, default=0.2, help='Temperature for controlling randomness (default: 0.2)')
parser.add_argument('--top_p', type=float, default=0.95, help='Top p value for nucleus sampling (default: 0.95)')
parser.add_argument('--do_sample', action='store_true', help='Enable sampling (default: False)')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for generation (default: 8)')
parser.add_argument('--prompted', action='store_true', help='Use prompted generation. See StarCoder paper (default: False)')
args = parser.parse_args()

""" Load prompts """
with open(args.prompts, 'r') as json_file:
    prompts = json.load(json_file)

    

for p in prompts:
    manipulated_prompt = p["prompt"]
    prompt_length = len(manipulated_prompt)

    #MPI PROCESSING
    if manipulated_prompt.find("MPI_") != -1:
        index_arr = []
        current_index = manipulated_prompt.find("MPI_")
        while current_index != -1:
            index_arr.append(current_index)
            current_index = manipulated_prompt.find("MPI_", current_index + 1)

        target = ""
        random_index = random.choice(index_arr)
        while random_index < prompt_length:
            if manipulated_prompt[random_index] != ";":
                target += manipulated_prompt[random_index]
            else:
                break
            random_index += 1
        target = target[4:]
        manipulated_prompt = manipulated_prompt.replace(target, "<fim_suffix>")

    #Kokkos PROCESSING
    elif manipulated_prompt.find("Kokkos::") != -1:
        index_arr = []
        current_index = manipulated_prompt.find("Kokkos::")
        while current_index != -1:
            index_arr.append(current_index)
            current_index = manipulated_prompt.find("Kokkos::", current_index + 1)
            
        target = ""
        random_index = random.choice(index_arr)
        while random_index < prompt_length:
            if manipulated_prompt[random_index] != ";":
                target += manipulated_prompt[random_index]
            else:
                break
            random_index += 1
        target = target[8:]
        manipulated_prompt = manipulated_prompt.replace(target, "<fim_suffix>")

    # OpenMP PROCESSING
    elif manipulated_prompt.find("#pragma omp") != -1:
        index_arr = []
        current_index = manipulated_prompt.find("#pragma omp")
        while current_index != -1:
            index_arr.append(current_index)
            current_index = manipulated_prompt.find("#pragma omp", current_index + 1)
            
        target = ""
        random_index = random.choice(index_arr)
        while random_index < prompt_length:
            if manipulated_prompt[random_index] != ")":
                target += manipulated_prompt[random_index]
            else:
                target += manipulated_prompt[random_index]
                break
            random_index += 1
        target = target[11:]
        manipulated_prompt = manipulated_prompt.replace(target, "<fim_suffix>")

    
    #RANDOM STRING PROCESSING
    else:
        suffix_length = random.randint(1, prompt_length - 1) 
        suffix_start = random.randint(0, prompt_length - suffix_length) 
        manipulated_prompt = manipulated_prompt[:suffix_start] + "<fim_suffix>" + manipulated_prompt[suffix_start + suffix_length:]

    manipulated_prompt = "<fim_prefix>" + manipulated_prompt + "<fim_middle>" 
    p["prompt"] = manipulated_prompt


""" Initialize inference config """
inference_config = get_inference_config(args.model, prompted=args.prompted)

# to use a torch.utils.data.DataSet with the HuggingFace pipeline, we need to flatten out the prompts
# and repeat them for however many samples we want to generate per prompt
prompts_repeated = [p for p in prompts for _ in range(args.num_samples_per_prompt)]

""" Initialize HuggingFace pipeline for generation """
generator = pipeline(model=args.model, torch_dtype=inference_config.get_dtype(), device_map="auto")
inference_config.init_padding(generator.tokenizer)

#Doing something here to randomly remove part of the program. and replace with <fim_prefix> or <fim_suffix> or <fim_middle>

""" Create a prompt data set to pass to generate method """

prompt_dataset = PromptDataset([inference_config.format_prompt(p["prompt"]) for p in prompts_repeated])

generated_outputs = generator(
    prompt_dataset,
    max_new_tokens=args.max_new_tokens,
    do_sample=args.do_sample,
    temperature=args.temperature,
    top_p=args.top_p,
    pad_token_id=inference_config.get_pad_token_id(generator.tokenizer),
    eos_token_id=inference_config.get_eos_token_id(generator.tokenizer),
    batch_size=args.batch_size,
)

""" Iterate over prompts and generate code """
responses = []
cur_prompt = None
start_time = time.time()
total_tokens = 0
for idx, (prompt, output) in tqdm(enumerate(zip(prompts_repeated, generated_outputs)), total=len(prompts_repeated), desc="Generating code", file=sys.stdout):
    if idx % args.num_samples_per_prompt == 0:
        cur_prompt = prompt.copy()
        cur_prompt.update({"temperature": args.temperature, "top_p": args.top_p, "do_sample": args.do_sample, "max_new_tokens": args.max_new_tokens, "prompted": args.prompted})
        cur_prompt["outputs"] = []
        prompt_str = cur_prompt["prompt"]

    total_tokens += len(generator.tokenizer.encode(output[0]["generated_text"]))
    cleaned_output = clean_output(output[0]["generated_text"], prompt_str)
    
    # Find position of <fim_suffix>
    suffix_pos = prompt_str.find('<fim_suffix>')
    if suffix_pos != -1:
        modified_prompt_str = prompt_str[:suffix_pos] + cleaned_output + prompt_str[suffix_pos:]
        modified_prompt_str = modified_prompt_str.replace("<fim_prefix>", "")
        modified_prompt_str = modified_prompt_str.replace("<fim_suffix>", "")
        modified_prompt_str = modified_prompt_str.replace("<fim_middle>", "")
    else: # if suffix is not found
        modified_prompt_str = prompt_str
    
    cur_prompt["outputs"].append(modified_prompt_str)

    if idx % args.num_samples_per_prompt == args.num_samples_per_prompt - 1:
        responses.append(cur_prompt)

    if idx != 0 and idx % args.num_samples_per_prompt == 0:
        print(f"Tokens per second: {total_tokens / (time.time() - start_time):.2f}")

end_time = time.time()
tokens_per_second = total_tokens / (end_time - start_time)
print(f"Generated {len(responses)} code samples in {end_time - start_time:.2f} seconds ({tokens_per_second:.2f} tokens per second)")

""" Save responses to JSON file """
with open(args.output, 'w') as output_file:
    json.dump(responses, output_file, indent=4)

