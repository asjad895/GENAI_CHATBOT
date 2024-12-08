import torch
import subprocess
from functools import wraps
from typing import Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from Conversational_ChatBot.config import MODEL_SAVE_DIR
import os
from dotenv import load_dotenv
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

def gpu_memory_decorator(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        before_memory = self._get_gpu_memory()
        print(f"Available GPU Memory Before '{func.__name__}': {before_memory:.2f} MB")

        result = func(self, *args, **kwargs)

        after_memory = self._get_gpu_memory()
        print(f"Available GPU Memory After '{func.__name__}': {after_memory:.2f} MB")

        return result
    return wrapper

class LLM:
    def __init__(self, **kwargs) -> None:
        if 'model_id' in kwargs:
            self.model_id = kwargs['model_id']
        else:
            self.model_id = "meta-llama/Llama-3.2-3b-Instruct"

        if 'device' in kwargs:
            self.device = kwargs['device']
        else:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def __call__(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        model, tokenizer = self.__get_model(self.model_id)
        return model, tokenizer

    @gpu_memory_decorator
    def __get_model(self, model_id: str = 'meta-llama/Llama-3.2-3B-Instruct') -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        if self._get_gpu_memory() <= 60000:
          print("Loading Quantized Model")
          save_dir = MODEL_SAVE_DIR['loaded_model']
          if os.path.exists(save_dir) and os.path.isdir(save_dir):
            print(f"Model found in '{save_dir}' directory. Loading model from there...")
            model = AutoModelForCausalLM.from_pretrained(save_dir)
            tokenizer = AutoTokenizer.from_pretrained(save_dir)
          else:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.bfloat16)
            print(f"Model not found in '{save_dir}' directory. Loading from the model ID: {model_id}...")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                quantization_config=quantization_config,
                use_auth_token=hf_token
              )
            # Load the tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_id,use_auth_token=hf_token)
          
          model.save_pretrained(save_dir, from_pt=True)
          tokenizer.save_pretrained(save_dir)

          return model, tokenizer
        else:
          print("Loading UnQuantized Model")
          model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16,use_auth_token=hf_token)
          tokenizer = AutoTokenizer.from_pretrained(model_id,use_auth_token=hf_token)
          return model, tokenizer

    def _get_gpu_memory(self) -> float:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        memory_free = result.stdout.decode().strip()
        try:
            return float(memory_free)
        except ValueError:
            return 0.0
