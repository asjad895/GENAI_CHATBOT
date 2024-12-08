from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from transformers import BitsAndBytesConfig
from typing import Tuple,Union,List,Dict
import subprocess

class ChatCompletion:
    def __init__(self, **kwargs) -> None:
        """
        Initializes the chat completion instance with optional device and model/tokenizer settings.
        """
        if 'device' in kwargs:
            self.device = kwargs['device']
        else:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        if 'model' in kwargs and 'tokenizer' in kwargs:
            self.model = kwargs['model']
            self.tokenizer = kwargs['tokenizer']
        else:
            raise Exception("'model' or 'tokenizer' not found")

    async def create(self, messages: List[Dict], **kwargs) -> str:
        """
        Generates a response to the provided messages using the model, and appends the message to the chat history.
        """
        
        max_new_tokens = kwargs.get('max_new_tokens', 50)
        temperature = kwargs.get('temperature', 0.6)
        top_p = kwargs.get('top_p', 0.9)
        do_sample = kwargs.get('do_sample', False)
        repetition_penalty = kwargs.get('repetition_penalty', 1.0)
        eos_token_id = kwargs.get('eos_token_id', None)
        pad_token_id = kwargs.get('pad_token_id', self.tokenizer.eos_token_id)
        return_full_text = kwargs.get('return_full_text', False)
        return_prompt = kwargs.get('return_prompt', False)
        return_instruction = kwargs.get('return_instruction', False)

        
        encoded_input = self.tokenizer.apply_chat_template(self.chat_history.get_chat_history(), tokenize=True, return_dict=True, return_tensors="pt").to(self.device)

        
        outputs = self.model.generate(
            **encoded_input,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id
        )

        response = self.tokenizer.decode(outputs[0])
        response = response.split('assistant<|end_header_id|>')[-1].strip().split('<|eot_id|>')[0].strip()

        return response

    async def clear_history(self) -> None:
        """
        Clears the chat history.
        """
        self.chat_history.clear_chat_history()