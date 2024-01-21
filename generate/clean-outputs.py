from argparse import ArgumentParser
import json

parser = ArgumentParser(description='clean model outputs')
parser.add_argument('--input', required=True, help='Path to the input JSON file')
parser.add_argument('--output', required=True, help='Path to the output JSON file')
args = parser.parse_args()

def has_balanced_brackets(s: str) -> bool:
    stack = []
    for c in s:
        if c == '{':
            stack.append(c)
        elif c == '}':
            if len(stack) == 0:
                return False
            stack.pop()
    return len(stack) == 0

def clean_output(prompt: str, output: str) -> str:
    last_line_of_prompt = prompt.split('\n')[-1].strip()
    if output.strip().startswith(last_line_of_prompt):
        output = output.replace(last_line_of_prompt, '', 1)

    if has_balanced_brackets("{" + output + "}"):
        output = output + '}'

    return output

with open(args.input, 'r') as json_file:
    responses = json.load(json_file)

for r in responses:
    r["outputs"] = [clean_output(r["prompt"], o) for o in r["outputs"]]

with open(args.output, 'w') as json_file:
    json.dump(responses, json_file, indent=2)