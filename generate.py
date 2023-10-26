import torch
import json
import argparse
from transformers import pipeline

parser = argparse.ArgumentParser(description='Generate code')

parser.add_argument('--prompts', help='Path to the prompt JSON file')
parser.add_argument('--model', help='Path to the language model')
parser.add_argument('--output', help='Path to the output JSON file')
parser.add_argument('--max_new_tokens', type=int, default=500, help='Maximum number of new tokens to generate (default: 500)')
parser.add_argument('--max_length', type=int, default=100, help='Maximum length (default: 500)')
parser.add_argument('--num-samples-per-prompt', type=int, default=100, help='Number of code samples to generate (default: 100)')
parser.add_argument('--temperature', type=float, default=0.9, help='Temperature for controlling randomness (default: 0.9)')
parser.add_argument('--top_p', type=float, default=0.9, help='Top p value for nucleus sampling (default: 0.9)')
parser.add_argument('--do_sample', action='store_true', help='Enable sampling (default: False)')
args = parser.parse_args()

with open(args.prompts, 'r') as json_file:
    prompts = json.load(json_file)

generator = pipeline(model=args.model, torch_dtype=torch.float16, device_map="auto")

responses = {}

for item in prompts:
    name = item["name"]
    prompt = item["prompt"]

    print(f"Generating code for: {name}")
    
    response = generator(prompt, max_length=args.max_length, max_new_tokens=args.max_new_tokens, do_sample=args.do_sample, temperature=args.temperature, top_p=args.top_p)

    responses[name] = response[0]['generated_text']

if args.output:
    with open(args.output, 'w') as output_file:
        json.dump(responses, output_file, indent=4)
