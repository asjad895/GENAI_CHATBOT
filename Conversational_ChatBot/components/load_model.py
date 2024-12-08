from Conversational_ChatBot.components.llm import LLM

def get_model():
    model,tokenizer = LLM()()
    return model,tokenizer
