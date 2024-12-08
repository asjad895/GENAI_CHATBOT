import torch
import subprocess
from functools import wraps
from typing import Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from Conversational_ChatBot.config import MODEL_SAVE_DIR
import os
from dotenv import load_dotenv
import psutil

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

        self.save_dir = MODEL_SAVE_DIR

    def __call__(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        model, tokenizer = self.__get_model(self.model_id)
        return model, tokenizer

    @gpu_memory_decorator
    def __get_model(self, model_id: str = 'meta-llama/Llama-3.2-3B-Instruct') -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        if  self.device == 'cuda' or self._get_gpu_memory() <= 60000:
          print("Loading Quantized Model")
          if os.path.exists(self.save_dir) and os.path.isdir(self.save_dir):
            print(f"Model found in '{self.save_dir}' directory. Loading model from there...")
            # model = AutoModelForCausalLM.from_pretrained(save_dir)
            # tokenizer = AutoTokenizer.from_pretrained(save_dir)
            self._get_gguf_model()
          else:
            # quantization_config = BitsAndBytesConfig(load_in_8bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.bfloat16)
            print(f"Model not found in '{self.save_dir}' directory. Loading from the model ID: {model_id}...")
            self._get_gguf_model()
            # model = AutoModelForCausalLM.from_pretrained(
            #     model_id,
            #     device_map="auto",
            #     torch_dtype=torch.bfloat16,
            #     quantization_config=quantization_config,
            #     use_auth_token=hf_token
            #   )
            # # Load the tokenizer
            # tokenizer = AutoTokenizer.from_pretrained(model_id,use_auth_token=hf_token)
          
        #   model.save_pretrained(save_dir, from_pt=True)
        #   tokenizer.save_pretrained(save_dir)
          model = None,
          tokenizer = None
          return model, tokenizer
        else:
          print("Loading UnQuantized Model")
          model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16,use_auth_token=hf_token)
          tokenizer = AutoTokenizer.from_pretrained(model_id,use_auth_token=hf_token)
          return model, tokenizer

    def _get_gpu_memory(self) -> float:
        if 'cuda' in self.device:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            memory_free = result.stdout.decode().strip()
        else:
            mem = psutil.virtual_memory()
            memory_free = mem.available / (1024 * 1024) 
        try:
            return float(memory_free)
        except ValueError:
            return 0.0
        
    def _get_gguf_model(self) ->str:
        local_dir = self.save_dir
        from huggingface_hub import hf_hub_download
        repo_id = "bartowski/Llama-3.2-3B-Instruct-GGUF"
        filename = "Llama-3.2-3B-Instruct-IQ3_M.gguf"
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False  
        )
        print("Model Loaded at ",file_path)

        return file_path
