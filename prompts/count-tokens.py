""" Utility for counting (or estimating) number of tokens in the prompts.
    author: Daniel Nichols
    date: November 2023
"""
# std imports
from argparse import ArgumentParser
import json

# tpl imports
import numpy as np


# AVAILABLE MODELS
OPENAI_MODELS = ["gpt-3.5-turbo", "gpt-4", "code-davinci-002", "code-cushman-001"]


class OpenAITokenCounter:

    def __init__(self, model_name: str):
        import tiktoken
        self.encoding = tiktoken.encoding_for_model(model_name)
    
    def __call__(self, prompt: str) -> int:
        """ Count the number of tokens in the prompt. """
        return len(self.encoding.encode(prompt))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("prompts", help="Path to prompts file")
    parser.add_argument("-m", "--model-name", required=True, choices=[*OPENAI_MODELS], help="What LLM to count tokens for")
    args = parser.parse_args()

    if args.model_name in OPENAI_MODELS:
        counter = OpenAITokenCounter(args.model_name)
    else:
        raise NotImplementedError(f"Model {args.model_name} not implemented")
    
    # read the prompts
    with open(args.prompts, "r") as f:
        prompts = json.load(f)

    # count the tokens
    num_tokens = []
    for prompt in prompts:
        num_tokens.append(counter(prompt["prompt"]))

    # print the results
    print(f"Mean: {np.mean(num_tokens)}")
    print(f"Std: {np.std(num_tokens)}")
    print(f"Min: {np.min(num_tokens)}")
    print(f"Max: {np.max(num_tokens)}")
    print(f"Median: {np.median(num_tokens)}")
    print(f"Total: {np.sum(num_tokens)}")


if __name__ == "__main__":
    main()