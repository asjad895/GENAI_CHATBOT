from .llm import LLM

def get_model():
    model,tokenizer = await LLM()
    return model,tokenizer

