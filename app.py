import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse,HTMLResponse,FileResponse
from fastapi.staticfiles import StaticFiles

from Conversational_ChatBot import config
from Conversational_ChatBot import constant
from Conversational_ChatBot.components import load_model, intent, prompt_template, chat_completion, chat_history
from typing import Dict
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "https://silver-adventure-x7v5j9jr5q629q66-8000.app.github.dev", 
    "http://localhost",  
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, 
    allow_credentials=True,
    allow_methods=["GET","POST"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model, tokenizer = load_model.get_model()

intent_classifier = intent.get_intent
chat_history = chat_history.ChatHistory()
chat = chat_completion.ChatCompletion(model=model, tokenizer=tokenizer)

prompt_template = prompt_template.PromptTemplate()


app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    with open("static/index.html", "r") as file:
        content = file.read()
    return HTMLResponse(content)


@app.post("/generative_response/")
async def generate_response(request: Request) -> JSONResponse:
    """
    Endpoint to get a generative response based on user query and session_id.
    """
    try:
        request_data = await request.json()
        user_query = request_data.get("user_query")
        session_id = request_data.get("session_id",'session_id')
        logger.info("Received user query: %s with session_id: %s", user_query, session_id)

        kwargs = {"keys": ['intents'], "values": [constant.intents_des]}
        system_message = await prompt_template.format(chatbot_prompt=False, intent_classifier_prompt=True)
        logger.info("System Message: %s", system_message)
        intent_result = intent_classifier(model, tokenizer, system_message, user_query, model.device)
        logger.info("Extracted Intent: %s", intent_result)

        if intent_result not in constant.expected_intents:
            intent_result = 'other' 

        await chat_history.add_message("user", user_query)

        # 
        system_chat = await prompt_template.format(chatbot_prompt=True, intent_classifier_prompt=False)

        messages = [system_chat]+chat_history.chat_history

        genration_config = config.generation_config.GENERATION_PARAMS

        chat_response = await chat.create(messages,kwargs=genration_config)
        logger.info("Generated Chat Response: %s", chat_response)

        await chat_history.add_message("assistant", chat_response)
        chat_response = chat_response+f"\nDomain:{intent_result}"
        response_data = {
            "session_id": session_id,
            "response": chat_response
        }

        return {response_data}

    except Exception as e:
        logger.error("Error processing request: %s", str(e))
        return {"response": "An error occurred while processing the request"},

