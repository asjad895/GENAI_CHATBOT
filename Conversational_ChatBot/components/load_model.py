from .llm import LLM

def get_model():
    model,tokenizer = LLM()
    return model,tokenizer

