from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Dict
from llama_cpp import Llama

class ChatCompletion:
    def __init__(self, **kwargs) -> None:
        """
        Initializes the chat completion instance with optional device and model/tokenizer settings.
        """
        self.device = kwargs.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # For using transformers-based model
        if 'transformers_model' in kwargs and 'transformers_tokenizer' in kwargs:
            self.model = kwargs['transformers_model']
            self.tokenizer = kwargs['transformers_tokenizer']
        # For using vllm model
        elif 'vllm_model_path' in kwargs and 'vllm_tokenizer_path' in kwargs:
            self.vllm_model_path = kwargs['vllm_model_path']
            self.vllm_tokenizer_path = kwargs['vllm_tokenizer_path']
            # self.llm = LLM(model=self.vllm_model_path, tokenizer=self.vllm_tokenizer_path)
            self.llm = Llama.from_pretrained(
                repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
	            filename="Llama-3.2-3B-Instruct-IQ1_M.gguf",
                )

        else:
            raise Exception("Model or tokenizer not found")

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
        pad_token_id = kwargs.get('pad_token_id', self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else None)
        
        # If device is CPU, use vllm model for response
        if self.device == 'cpu':
            return await self.vllm_response(messages, max_new_tokens, temperature, top_p, do_sample, repetition_penalty, eos_token_id, pad_token_id)
        else:
            # If device is GPU, use transformers-based model
            return await self.transformers_response(messages, max_new_tokens, temperature, top_p, do_sample, repetition_penalty, eos_token_id, pad_token_id)

    async def vllm_response(self, messages: List[Dict], max_new_tokens: int, temperature: float, top_p: float, do_sample: bool, repetition_penalty: float, eos_token_id: int, pad_token_id: int) -> str:
        """
        Generate response using vllm model
        """
        # Prepare conversation for vllm
        conversation = [{'role': msg['role'], 'content': msg['content']} for msg in messages]
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p)

        outputs = self.llm.chat(conversation, sampling_params)
        generated_text = outputs[0].outputs[0].text if outputs else "Error generating response"
        
        return generated_text

    async def transformers_response(self, messages: List[Dict], max_new_tokens: int, temperature: float, top_p: float, do_sample: bool, repetition_penalty: float, eos_token_id: int, pad_token_id: int) -> str:
        """
        Generate response using transformers-based model
        """
        # Prepare conversation for transformers
        conversation = [{'role': msg['role'], 'content': msg['content']} for msg in messages]
        inputs = self.tokenizer(conversation, return_tensors="pt", padding=True, truncation=True).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    async def clear_history(self) -> None:
        """
        Clears the chat history.
        """
        self.chat_history.clear_chat_history()

