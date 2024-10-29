# std imports
import argparse
import json
import os
import sys
import time
from tqdm import tqdm

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
parser.add_argument('--restart', action='store_true', help='Restart generation from scratch (default: False)')
parser.add_argument('--cache', help='JSONL file to cache intermediate results in. Will be restored from if it ' +
    'already exists and --restart is not specified')
parser.add_argument('--restore_from', help='JSON file to restore old results from. Will be restored from ' +
    'if it already exists and --restart is not specified. Is different from --cache in that it is a JSON file, not a ' +
    'JSONL file, and it is only used to restore old results where the prompt is equivalent. Cached results are ' +
    'prioritized over restored results.')
parser.add_argument('--max_new_tokens', type=int, default=1024, help='Maximum number of new tokens to generate (default: 1024)')
parser.add_argument('--num_samples_per_prompt', type=int, default=50, help='Number of code samples to generate (default: 50)')
parser.add_argument('--temperature', type=float, default=0.2, help='Temperature for controlling randomness (default: 0.2)')
parser.add_argument('--top_p', type=float, default=0.95, help='Top p value for nucleus sampling (default: 0.95)')
parser.add_argument('--do_sample', action='store_true', help='Enable sampling (default: False)')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for generation (default: 8)')
parser.add_argument('--prompted', action='store_true', help='Use prompted generation. See StarCoder paper (default: False)')
parser.add_argument('--hf_token', type=str, help='HuggingFace API token for loading models')
args = parser.parse_args()

""" Load prompts """
with open(args.prompts, 'r') as json_file:
    prompts = json.load(json_file)

""" Load existing responses if they exist """
if not args.restart and os.path.exists(args.cache):
    with open(args.cache, 'r') as jsonl_file:
        responses = [json.loads(line) for line in jsonl_file]
    
    # remove prompt from prompts if it is in responses and has an 'output' value with at least 1 entry
    original_len = len(prompts)
    prompts = [p for p in prompts if 
                not any(p["name"] == r["name"] and 
                        p["parallelism_model"] == r["parallelism_model"] and
                        p["prompt"] == r["prompt"] and 
                        args.temperature == r["temperature"] and 
                        args.prompted == r["prompted"] and
                        args.num_samples_per_prompt == len(r["outputs"])
                        for r in responses)]
    print(f"[cache] Skipping {original_len - len(prompts)} prompts that already have responses")

""" Load existing responses if they exist """
if not args.restart and args.restore_from and os.path.exists(args.restore_from):
    with open(args.restore_from, 'r') as json_file:
        restored_responses = json.load(json_file)
    
    # remove prompt from prompts if it is in responses and has an 'output' value with at least 1 entry
    original_len = len(prompts)
    responses_to_keep = []
    prompts_without_existing_responses = []
    for p in prompts:
        for r in restored_responses:
            if p["name"] == r["name"] and \
                p["parallelism_model"] == r["parallelism_model"] and \
                p["prompt"] == r["prompt"] and \
                args.temperature == r["temperature"] and \
                args.prompted == r["prompted"] and \
                args.num_samples_per_prompt == len(r["outputs"]):
                responses_to_keep.append(r)
                break
        else:
            prompts_without_existing_responses.append(p)
    prompts = prompts_without_existing_responses
    print(f"[restore_from] Skipping {original_len - len(prompts)} prompts that already have responses. " +
        f"{len(prompts)} prompts left.")

    # write restored responses to cache
    if args.cache is not None:
        with open(args.cache, 'a') as jsonl_file:
            for response in responses_to_keep:
                jsonl_file.write(json.dumps(response) + "\n")
            print(f"[restore_from] Wrote {len(responses_to_keep)} restored responses to cache")


""" Initialize inference config """
inference_config = get_inference_config(args.model, prompted=args.prompted)

# to use a torch.utils.data.DataSet with the HuggingFace pipeline, we need to flatten out the prompts
# and repeat them for however many samples we want to generate per prompt
prompts_repeated = [p for p in prompts for _ in range(args.num_samples_per_prompt)]

""" Initialize HuggingFace pipeline for generation """
generator = pipeline(task="text-generation", model=args.model, torch_dtype=inference_config.get_dtype(), device=0, token=args.hf_token)
inference_config.init_padding(generator.tokenizer)

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
if not args.restart and args.cache is not None and os.path.exists(args.cache):
    with open(args.cache, 'r') as jsonl_file:
        responses = [json.loads(line) for line in jsonl_file]
        responses = [r for r in responses if r["temperature"] == args.temperature and r["prompted"] == args.prompted
                        and args.num_samples_per_prompt == len(r["outputs"])
                        and any(p["name"] == r["name"] and p["prompt"] == r["prompt"] and p["parallelism_model"] == r["parallelism_model"] for p in prompts)]
else:
    responses = []
cur_prompt = None
start_time = time.time()
total_tokens = 0
for idx, (prompt, output) in tqdm(enumerate(zip(prompts_repeated, generated_outputs)), total=len(prompts_repeated), desc="Generating code", file=sys.stdout):
    if idx % args.num_samples_per_prompt == 0:
        cur_prompt = prompt.copy()
        cur_prompt.update({"temperature": args.temperature, "top_p": args.top_p, "do_sample": args.do_sample, "max_new_tokens": args.max_new_tokens, "prompted": args.prompted})
        cur_prompt["outputs"] = []
        cur_prompt["raw_outputs"] = []
        prompt_str = cur_prompt["prompt"]

    total_tokens += len(generator.tokenizer.encode(output[0]["generated_text"]))
    cleaned_output = inference_config.clean_output(output[0]["generated_text"], prompt_str)
    cur_prompt["outputs"].append(cleaned_output)
    cur_prompt["raw_outputs"].append(output[0]["generated_text"])

    if idx % args.num_samples_per_prompt == args.num_samples_per_prompt - 1:
        responses.append(cur_prompt)

        if not args.restart and args.cache is not None:
            with open(args.cache, 'a') as jsonl_file:
                jsonl_file.write(json.dumps(cur_prompt) + "\n")

    if idx != 0 and idx % args.num_samples_per_prompt == 0:
        print(f"Tokens per second: {total_tokens / (time.time() - start_time):.2f}")

end_time = time.time()
tokens_per_second = total_tokens / (end_time - start_time)
print(f"Generated {len(responses)} code samples in {end_time - start_time:.2f} seconds ({tokens_per_second:.2f} tokens per second)")

""" Save responses to JSON file """
with open(args.output, 'w') as output_file:
    json.dump(responses, output_file, indent=4)
