# std imports
from abc import ABC, abstractmethod
import re

# tpl imports
import torch
from torch.utils.data import Dataset
from transformers import StoppingCriteria


def clean_output(output : str, prompt : str) -> str:
    """ Remove `prompt` from the begging of `output`.
        Also truncate at the end of the function definition (i.e. matching closing brace).
    """
    # replace up to the end of the first instance of prompt
    prompt_loc = output.find(prompt)
    if prompt_loc == -1:
        raise ValueError(f"Prompt not found in output: {prompt}")
    output = output[prompt_loc + len(prompt):].strip()

    # temporarily add opening brace to the beginning
    output = '{' + output

    # find the matching brace to output[0]
    stack = []
    index = 0
    while index < len(output):
        token = output[index]
        if token == '{':
            stack.append(token)
        elif token == '}':
            stack.pop()
            if len(stack) == 0:
                break

        index += 1

    # truncate at the matching brace
    output = output[1:index+1]
    return output

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


def find_matching_brace_index(code: str, open_brace_index: int) -> int:
    """Finds the index of the closing brace that matches the opening brace at the given index."""

    brace_count = 1
    for i in range(open_brace_index + 1, len(code)):
        if code[i] == "{":
            brace_count += 1
        elif code[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                return i

    raise ValueError("Unmatched opening brace")


def clean_instruct_output(output: str, prompt: str, response_tag: str) -> str:
    """ Clean LLM output to find code solution. The output should be in a ```c++ ``` code block. If there are
        multiple, then it tries to find the block with the function definition (as contained in the prompt).
        The code block itself may include the function definition and body OR just the body. This will try
        to parse both.
    """
    # 0. replace up to the end of the first instance of prompt
    prompt_loc = output.find(response_tag)
    if prompt_loc == -1:
        raise ValueError(f"Prompt not found in output: {prompt}")
    output = output[prompt_loc + len(response_tag):].strip()

    # 1. Find all code blocks enclosed in triple backticks with "c++" language tag
    code_blocks = re.findall(r"```\n(.*?)\n```", output, flags=re.DOTALL)
    code_blocks = [block.removeprefix("```").removeprefix("cpp").removeprefix('c++').removesuffix('```') for block in code_blocks]

    # 2. Prioritize code blocks containing the function definition from the prompt
    sub_prompt = prompt.rstrip().removesuffix(response_tag).rstrip().removesuffix("```").split("```")[-1]
    function_name = get_function_name(sub_prompt, "cuda" if "__global__" in sub_prompt else "serial")
    prioritized_blocks = [block for block in code_blocks if function_name in block]

    # 3. Choose the first block if multiple match, or any block if none match
    if len(code_blocks) > 0:
        selected_block = prioritized_blocks[0] if prioritized_blocks else code_blocks[0]
    else:
        if '```' in output: # starts with ```c++ but it didn't finish
            code_idx = output.find('```')
            selected_block = output[code_idx:].removeprefix('```')
        else:
            selected_block = output

    # 4. Handle cases where the block contains only the function body
    if function_name not in selected_block:
        return selected_block
    else:
        function_start_index = selected_block.index(function_name)
        open_brace_index = selected_block.find("{", function_start_index)
        try:
            close_brace_index = find_matching_brace_index(selected_block, open_brace_index)
        except ValueError:
            close_brace_index = len(selected_block)

        function_body = selected_block[open_brace_index + 1 : close_brace_index]
        return function_body + "}"


class InferenceConfig(ABC):

    def __init__(self, prompted : bool = False):
        self.prompted = prompted
    
    @abstractmethod
    def get_dtype(self):
        pass
    
    @abstractmethod
    def init_padding(self, tokenizer):
        pass

    @abstractmethod
    def get_pad_token_id(self, tokenizer) -> int:
        pass

    @abstractmethod
    def get_eos_token_id(self, tokenizer) -> int:
        pass

    @abstractmethod
    def trust_remote_code(self) -> bool:
        pass

    @abstractmethod
    def format_prompt(self, prompt : str) -> str:
        pass

    @abstractmethod
    def clean_output(self, output: str, prompt: str) -> str:
        pass


class StarCoderConfig(InferenceConfig):

    def __init__(self, prompted : bool = False):
        super().__init__(prompted=prompted)

    def get_dtype(self):
        return torch.float16

    def init_padding(self, tokenizer):
        tokenizer.pad_token_id = tokenizer.eos_token_id  # for batching
        tokenizer.padding_side = "left"   # for decoder-only models

    def get_pad_token_id(self, tokenizer) -> int:
        return tokenizer.eos_token_id

    def get_eos_token_id(self, tokenizer) -> int:
        return None

    def trust_remote_code(self) -> bool:
        return False

    def format_prompt(self, prompt : str) -> str:
        if self.prompted:
            return f"<filename>solutions/solution_1.cpp\n// here is the correct implementation of the coding exercise\n\n{prompt}"
        return prompt.strip()

    def clean_output(self, output: str, prompt: str) -> str:
        return clean_output(output, prompt)

class CodeLlamaConfig(InferenceConfig):

    def __init__(self, prompted : bool = False):
        super().__init__(prompted=prompted)

    def get_dtype(self):
        return torch.float16

    def init_padding(self, tokenizer):
        tokenizer.pad_token_id = tokenizer.eos_token_id  # for batching
        tokenizer.padding_side = "left"   # for decoder-only models
        pass

    def get_pad_token_id(self, tokenizer) -> int:
        return tokenizer.pad_token_id

    def get_eos_token_id(self, tokenizer) -> int:
        return tokenizer.eos_token_id
    
    def trust_remote_code(self) -> bool:
        return False

    def format_prompt(self, prompt : str) -> str:
        if self.prompted:
            return f"// filename: solutions/solution_1.cpp\n// here is the correct implementation of the coding exercise\n\n{prompt}"
        return prompt.strip()

    def clean_output(self, output: str, prompt: str) -> str:
        return clean_output(output, prompt)

class PolyCoderConfig(InferenceConfig):

    def __init__(self, prompted : bool = False):
        super().__init__(prompted=prompted)

    def get_dtype(self):
        return torch.float16

    def init_padding(self, tokenizer):
        tokenizer.pad_token_id = tokenizer.eos_token_id  # for batching
        tokenizer.padding_side = "left"   # for decoder-only models

    def get_pad_token_id(self, tokenizer) -> int:
        return tokenizer.eos_token_id

    def get_eos_token_id(self, tokenizer) -> int:
        return tokenizer.eos_token_id

    def trust_remote_code(self) -> bool:
        return False

    def format_prompt(self, prompt : str) -> str:
        if self.prompted:
            return f"// filename: solutions/solution_1.cpp\n// here is the correct implementation of the coding exercise\n\n{prompt}"
        return prompt.strip()
    
    def clean_output(self, output: str, prompt: str) -> str:
        return clean_output(output, prompt)


class PhindConfig(InferenceConfig):

    def __init__(self, prompted: bool = False):
        super().__init__(prompted=prompted)

    def get_dtype(self):
        return torch.float16

    def init_padding(self, tokenizer):
        tokenizer.pad_token_id = tokenizer.eos_token_id  # for batching
        tokenizer.padding_side = "left"   # for decoder-only models

    def get_pad_token_id(self, tokenizer) -> int:
        return tokenizer.eos_token_id

    def get_eos_token_id(self, tokenizer) -> int:
        return tokenizer.eos_token_id
    
    def trust_remote_code(self) -> bool:
        return False

    def format_prompt(self, prompt : str) -> str:
        if self.prompted:
            return f"// filename: solutions/solution_1.cpp\n// here is the correct implementation of the coding exercise\n\n{prompt}"
        return prompt.strip()

    def clean_output(self, output: str, prompt: str) -> str:
        return clean_output(output, prompt)


class ReplitConfig(InferenceConfig):

    def __init__(self, prompted: bool = False):
        super().__init__(prompted=prompted)

    def get_dtype(self):
        return torch.float16

    def init_padding(self, tokenizer):
        pass
        #tokenizer.pad_token_id = tokenizer.eos_token_id  # for batching
        #tokenizer.padding_side = "left"   # for decoder-only models

    def get_pad_token_id(self, tokenizer) -> int:
        return None
        #return tokenizer.eos_token_id

    def get_eos_token_id(self, tokenizer) -> int:
        return None
        #return tokenizer.eos_token_id
    
    def trust_remote_code(self) -> bool:
        return True

    def format_prompt(self, prompt : str) -> str:
        if self.prompted:
            return f"// filename: solutions/solution_1.cpp\n// here is the correct implementation of the coding exercise\n\n{prompt}"
        return prompt.strip()

    def clean_output(self, output: str, prompt: str) -> str:
        return clean_output(output, prompt)


class MagicoderConfig(InferenceConfig):

    PROMPT_TEMPLATE = """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.

@@ Instruction
{instruction}

@@ Response
"""

    def __init__(self, prompted : bool = False):
        super().__init__(prompted=prompted)

    def get_dtype(self):
        return torch.bfloat16

    def init_padding(self, tokenizer):
        tokenizer.pad_token_id = tokenizer.eos_token_id  # for batching
        tokenizer.padding_side = "left"   # for decoder-only models
        pass

    def get_pad_token_id(self, tokenizer) -> int:
        return tokenizer.pad_token_id

    def get_eos_token_id(self, tokenizer) -> int:
        return tokenizer.eos_token_id
    
    def trust_remote_code(self) -> bool:
        return False

    def format_prompt(self, prompt : str) -> str:
        if self.prompted:
            function_name = get_function_name(prompt, "cuda" if "__global__" in prompt else "serial")
            prompt = f"Complete the following c++ function.\n```c++{prompt.strip()}```\nWrite only the function {function_name} and no other code. Enclose your solution in ```c++ and ```."
            return self.PROMPT_TEMPLATE.format(instruction=prompt)
        return prompt.strip()

    def clean_output(self, output: str, prompt: str) -> str:
        return clean_instruct_output(output, prompt, "@@ Response")


class DeepSeekBaseConfig(InferenceConfig):

    def __init__(self, prompted : bool = False):
        super().__init__(prompted=prompted)

    def get_dtype(self):
        return torch.bfloat16

    def init_padding(self, tokenizer):
        tokenizer.pad_token_id = tokenizer.eos_token_id  # for batching
        tokenizer.padding_side = "left"   # for decoder-only models

    def get_pad_token_id(self, tokenizer) -> int:
        return tokenizer.pad_token_id

    def get_eos_token_id(self, tokenizer) -> int:
        return tokenizer.eos_token_id
    
    def trust_remote_code(self) -> bool:
        return False

    def format_prompt(self, prompt : str) -> str:
        if self.prompted:
            return f"// filename: solutions/solution_1.cpp\n// here is the correct implementation of the coding exercise\n\n{prompt}"
        return prompt.strip()

    def clean_output(self, output: str, prompt: str) -> str:
        return clean_output(output, prompt)


class InstructConfig(InferenceConfig):

    def __init__(self, prompted : bool = False, instruction_tag : str = "### Instruction", response_tag : str = "### Response"):
        super().__init__(prompted=prompted)
        self.instruction_tag = instruction_tag
        self.response_tag = response_tag

    def get_dtype(self):
        return torch.bfloat16

    def init_padding(self, tokenizer):
        tokenizer.pad_token_id = tokenizer.eos_token_id  # for batching
        tokenizer.padding_side = "left"   # for decoder-only models

    def get_pad_token_id(self, tokenizer) -> int:
        return tokenizer.pad_token_id

    def get_eos_token_id(self, tokenizer) -> int:
        return tokenizer.eos_token_id
    
    def trust_remote_code(self) -> bool:
        return False

    def format_prompt(self, prompt : str) -> str:
        function_name = get_function_name(prompt, "cuda" if "__global__" in prompt else "serial")
        prompt = f"Complete the following c++ function.\n```c++{prompt.strip()}```\nWrite only the function {function_name} and no other code. Enclose your solution in ```c++ and ```."
        prompt = f"{self.instruction_tag}\n{prompt}\n{self.response_tag}\n"
        return prompt.strip()

    def clean_output(self, output: str, prompt: str) -> str:
        return clean_instruct_output(output, prompt, self.response_tag)

def get_inference_config(model_name : str, **kwargs) -> InferenceConfig:
    if model_name == "bigcode/starcoderbase":
        return StarCoderConfig(**kwargs)
    elif model_name in ["bigcode/starcoder2-3b", "bigcode/starcoder2-7b", "bigcode/starcoder2-15b"]:
        return StarCoderConfig(**kwargs)
    elif model_name.startswith("codellama/CodeLlama-") and 'Instruct' not in model_name:
        return CodeLlamaConfig(**kwargs)
    elif model_name == "NinedayWang/PolyCoder-2.7B":
        return PolyCoderConfig(**kwargs)
    elif model_name == 'Phind/Phind-CodeLlama-34B-v2':
        return PhindConfig(**kwargs)
    elif model_name == 'replit/replit-code-v1_5-3b':
        return ReplitConfig(**kwargs)
    elif model_name.startswith('ise-uiuc/Magicoder'):
        return MagicoderConfig(**kwargs)
    elif model_name in ['deepseek-ai/deepseek-coder-6.7b-base', 'deepseek-ai/deepseek-coder-7b-base-v1.5']:
        return DeepSeekBaseConfig(**kwargs)
    elif model_name.startswith('hpcgroup/hpc-coder-v2'):
        return InstructConfig(instruction_tag='Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:', response_tag='### Response:', **kwargs)
    elif model_name.startswith('hpcgroup/rlpf'):
        return InstructConfig(instruction_tag='### Instruction', response_tag='### Response', **kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


class PromptDataset(Dataset):
    ''' PyTorch dataset that simply wraps a list of strings. They do not have to have the same length.
    '''

    def __init__(self, prompts):
        super().__init__()
        self.prompts_ = prompts
    
    def __len__(self):
        return len(self.prompts_)
    
    def __getitem__(self, idx): 
        return self.prompts_[idx]


def has_balanced_brackets(text : str, left_bracket : str = '{', right_bracket : str = '}') -> bool:
    ''' Check if string has balanced brackets.
        modified from: https://stackoverflow.com/a/38834249/3769237

        Arguments:
            text: string to check for balanced brackets in.
            left_bracket: left bracket to balance
            right_bracket: right bracket to balance

        Returns:
            true if left_bracket and right_bracket are balanced
    '''
    stack = []
    balanced = True
    index = 0
    while index < len(text) and balanced:
        token = text[index]
        if token == left_bracket:
            stack.append(token)
        elif token == right_bracket:
            if len(stack) == 0:
                balanced = False
            else:
                stack.pop()

        index += 1

    return balanced and len(stack) == 0


class BalancedBracketsCriteria(StoppingCriteria):
    ''' extension of transformers' text-generation stopping criteria.
        Stops either when function is complete (i.e. { and } are balanced) or when max_length is surpassed, whichever
        happens first. 

        _Note:_ This is a slow stopping criteria, but it's much faster than continually running model inference when 
        it does not need to be run anymore.
    '''

    def __init__(self, max_length : int, tokenizer, left_bracket : str = '{', right_bracket : str = '}'):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.left_bracket = left_bracket
        self.right_bracket = right_bracket
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids.shape[-1] > self.max_length:
            # already too long, early stop
            return True

        # return true if {} are balanced i.e. the function is complete
        return all(
            has_balanced_brackets(
                self.tokenizer.decode(t), 
                left_bracket=self.left_bracket, 
                right_bracket=self.right_bracket
            ) for t in input_ids)