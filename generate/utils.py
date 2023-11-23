# std imports
from abc import ABC, abstractmethod

# tpl imports
import torch
from torch.utils.data import Dataset
from transformers import StoppingCriteria


class InferenceConfig(ABC):
    
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
    def format_prompt(self, prompt : str) -> str:
        pass


class StarCoderConfig(InferenceConfig):

    def get_dtype(self):
        return torch.float16

    def init_padding(self, tokenizer):
        tokenizer.pad_token_id = tokenizer.eos_token_id  # for batching
        tokenizer.padding_side = "left"   # for decoder-only models

    def get_pad_token_id(self, tokenizer) -> int:
        return tokenizer.eos_token_id

    def get_eos_token_id(self, tokenizer) -> int:
        return None

    def format_prompt(self, prompt : str) -> str:
        return prompt.strip()

class CodeLlamaConfig(InferenceConfig):

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

    def format_prompt(self, prompt : str) -> str:
        return prompt.strip()


class PolyCoderConfig(InferenceConfig):

    def get_dtype(self):
        return torch.float16

    def init_padding(self, tokenizer):
        tokenizer.pad_token_id = tokenizer.eos_token_id  # for batching
        tokenizer.padding_side = "left"   # for decoder-only models

    def get_pad_token_id(self, tokenizer) -> int:
        return tokenizer.eos_token_id

    def get_eos_token_id(self, tokenizer) -> int:
        return tokenizer.eos_token_id

    def format_prompt(self, prompt : str) -> str:
        return prompt.strip()


def get_inference_config(model_name : str) -> InferenceConfig:
    if model_name == "bigcode/starcoderbase":
        return StarCoderConfig()
    elif model_name.startswith("codellama/CodeLlama-") and 'Instruct' not in model_name:
        return CodeLlamaConfig()
    elif model_name == "NinedayWang/PolyCoder-2.7B":
        return PolyCoderConfig()
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def clean_output(output : str, prompt : str) -> str:
    """ Remove `prompt` from the begging of `output`.
        Also truncate at the end of the function definition (i.e. matching closing brace).
    """
    # remove prompt
    output = output.replace(prompt, "", 1).strip()

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