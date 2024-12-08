import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from .Conversational_ChatBot import config
from .Conversational_ChatBot import constant
from .Conversational_ChatBot.components import load_model, intent, prompt_template, chat_completion, chat_history
from typing import Dict

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model, tokenizer = load_model()

intent_classifier = intent.get_intent
chat_history = chat_history.ChatHistory()
chat = chat_completion.ChatCompletion(model=model, tokenizer=tokenizer)

@app.post("/generate-response/")
async def generate_response(request: Request) -> JSONResponse:
    """
    Endpoint to get a generative response based on user query and session_id.
    """
    try:
        request_data = await request.json()
        user_query = request_data.get("user_query")
        session_id = request_data.get("session_id")
        logger.info("Received user query: %s with session_id: %s", user_query, session_id)

        prompt_template = prompt_template.PromptTemplate()
        kwargs = {"keys": ['intents'], "values": [constant.intents_des]}
        system_message = prompt_template.format(chatbot_prompt=False, intent_classifier_prompt=True)
        logger.info("System Message: %s", system_message)
        intent_result = intent_classifier(model, tokenizer, system_message, user_query, model.device)
        logger.info("Extracted Intent: %s", intent_result)

        if intent_result not in constant.expected_intents:
            intent_result = 'other' 

        chat_history.add_message("user", user_query)

        # 
        system_chat = prompt_template.format(chatbot_prompt=True, intent_classifier_prompt=False)

        messages = [system_chat]+chat_history.chat_history

        genration_config = config.generation_config.GENERATION_PARAMS

        chat_response = await chat.create(messages,kwargs=genration_config)
        logger.info("Generated Chat Response: %s", chat_response)

        chat_history.add_message('user',user_query)
        chat_history.add_message("assistant", chat_response)

        response_data = {
            "session_id": session_id,
            "intent": intent_result,
            "response": chat_response
        }

        return JSONResponse(content=response_data, status_code=200)

    except Exception as e:
        logger.error("Error processing request: %s", str(e))
        return JSONResponse(
            content={"error": "An error occurred while processing the request"},
            status_code=500
        )

