import logging
import streamlit as st
from Conversational_ChatBot import config
from Conversational_ChatBot import constant
from Conversational_ChatBot.components import load_model, intent, prompt_template, chat_completion, chat_history
import asyncio

# load_model.get_model()
# Load necessary components
intent_classifier = intent.get_intent
chat = chat_completion.ChatCompletion(vllm_model_path='../loaded_model/Llama-3.2-3B-Instruct-IQ3_M.gguf', vllm_tokenizer_path='meta-llama/Llama-3.2-3B-Instruct')
prompt_template = prompt_template.PromptTemplate(fill_keys=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit app setup
st.title("Chatbot Interface")
st.write("Ask me anything!")

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = "12345"

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Async function to generate response
async def generate_response(user_query: str, session_id: str):
    try:
        # Prepare system message
        kwargs = {"keys": ['intents'], "values": [constant.intents_des]}
        system_message = await prompt_template.format(chatbot_prompt=False, intent_classifier_prompt=True, values=kwargs['values'])

        # Classify intent
        intent_result = await intent_classifier(chat, system_message, user_query, 'cpu')
        if intent_result not in constant.expected_intents:
            intent_result = 'other'

        # Add user message to session chat history
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        # Prepare chat history and system prompt
        system_chat = await prompt_template.format(chatbot_prompt=True, intent_classifier_prompt=False)
        messages = [{"role": "system", "content": system_chat}] + st.session_state.chat_history

        # Generate response
        generation_config = config.generation_config.GENERATION_PARAMS
        chat_response = await chat.create(messages, **generation_config)

        # Add assistant message to session chat history
        st.session_state.chat_history.append({"role": "assistant", "content": chat_response})

        # Append domain information
        chat_response += f"\n\nDomain: {intent_result}"
        return {"session_id": session_id, "response": chat_response}
    
    except Exception as e:
        logger.error("Error processing request: %s", e)
        return {"response": "An error occurred while processing the request"}


st.write("### Conversation")
conversation_container = st.container()


with conversation_container:
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.chat_message("User").markdown(f"**You:** {message['content']}")
        elif message["role"] == "assistant":
            st.chat_message("Assistant").markdown(f"**Bot:** {message['content']}")


user_query = st.text_input("Your question:")

if user_query and st.button("Send"):
    with st.spinner("Getting response from bot..."):
        response_data = asyncio.run(generate_response(user_query, st.session_state.session_id))
        st.session_state.user_query = ""  
