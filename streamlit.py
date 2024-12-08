import logging
import streamlit as st
from Conversational_ChatBot import config
from Conversational_ChatBot import constant
from Conversational_ChatBot.components import load_model, intent, prompt_template, chat_completion, chat_history
import asyncio

# model, tokenizer = load_model.get_model()
intent_classifier = intent.get_intent
chat_history = chat_history.ChatHistory()
chat = chat_completion.ChatCompletion(vllm_model_path ='../loaded_model/Llama-3.2-3B-Instruct-IQ3_M.gguf' , vllm_tokenizer_path='meta-llama/Llama-3.2-3B-Instruct')
prompt_template = prompt_template.PromptTemplate()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.title("Chatbot Interface")
st.write("Ask me anything!")

if 'session_id' not in st.session_state:
    st.session_state.session_id = "12345"


async def generate_response(user_query: str, session_id: str):
    try:
        kwargs = {"keys": ['intents'], "values": [constant.intents_des]}
        system_message = await prompt_template.format(chatbot_prompt=False, intent_classifier_prompt=True,kwargs=kwargs)
        print(system_message)
        intent_result = await intent_classifier(chat,system_message, user_query, 'cpu')

        if intent_result not in constant.expected_intents:
            intent_result = 'other'

        await chat_history.add_message("user", user_query)

        system_chat = await prompt_template.format(chatbot_prompt=True, intent_classifier_prompt=False)
        messages = [system_chat] + chat_history.chat_history
        genration_config = config.generation_config.GENERATION_PARAMS

        chat_response = await chat.create(messages, kwargs=genration_config)

        await chat_history.add_message("assistant", chat_response)
        chat_response += f"\nDomain: {intent_result}"

        return {"session_id": session_id, "response": chat_response}
    
    except Exception as e:
        logger.error("Error processing request: %s", e)
        return {"response": "An error occurred while processing the request"}

user_query = st.text_input("Your question:")

if user_query:
    with st.spinner("Getting response from bot..."):
        response_data = asyncio.run(generate_response(user_query, st.session_state.session_id))

    st.write(f"**Bot**: {response_data['response']}")


