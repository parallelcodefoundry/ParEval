""" Script for benchmarking the throughput of the models
    author: Daniel Nichols
    date: December 2023
"""
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
parser.add_argument('--max_new_tokens', type=int, default=1024, help='Maximum number of new tokens to generate (default: 1024)')
parser.add_argument('--num_samples_per_prompt', type=int, default=50, help='Number of code samples to generate (default: 50)')
parser.add_argument('--temperature', type=float, default=0.2, help='Temperature for controlling randomness (default: 0.2)')
parser.add_argument('--top_p', type=float, default=0.95, help='Top p value for nucleus sampling (default: 0.95)')
parser.add_argument('--do_sample', action='store_true', help='Enable sampling (default: False)')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for generation (default: 8)')
parser.add_argument('--num_batches', type=int, default=10, help='How many batches to run (default: 10)')
parser.add_argument('--warmup', type=int , default=5, help='How many batches to run before timing (default: 5)')
args = parser.parse_args()

""" Load prompts """
with open(args.prompts, 'r') as json_file:
    prompts = json.load(json_file)

""" Initialize inference config """
inference_config = get_inference_config(args.model, prompted=True)

# to use a torch.utils.data.DataSet with the HuggingFace pipeline, we need to flatten out the prompts
# and repeat them for however many samples we want to generate per prompt
prompts_repeated = [p for p in prompts for _ in range(args.num_samples_per_prompt)]
sample_cutoff = args.batch_size * (args.num_batches + args.warmup)
prompts_repeated = prompts_repeated[:sample_cutoff]

""" Initialize HuggingFace pipeline for generation """
generator = pipeline(model=args.model, torch_dtype=inference_config.get_dtype(), device=0, trust_remote_code=inference_config.trust_remote_code())
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

outputs = []
past_warmup = False
for idx, output in tqdm(enumerate(generated_outputs), total=len(prompts_repeated), desc="Generating code", file=sys.stdout):
    if idx == args.warmup * args.batch_size:
        past_warmup = True
        start_time = time.time()

    if past_warmup:
        outputs.append(output[0]["generated_text"])
    

duration = time.time() - start_time
total_tokens = sum(len(generator.tokenizer.encode(o)) for o in outputs)
print(f"Throughput: {total_tokens / duration} tokens/sec")
