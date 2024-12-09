# GENAI Chatbot

Objective:
Create a domain-specific chatbot capable of answering queries in Healthcare, Insurance, Finance, and Retail domains using the Mixtral Model / LLAMA. The chatbot should also gracefully handle out-of-domain queries by responding:
"I can only assist with queries related to Healthcare, Insurance, Finance, or Retail."
Additionally, build a web application using Streamlit or FastAPI to interact with the chatbot.

Instructions:
1. Core Functionality
Domain-Specific Queries:
The chatbot must provide accurate and contextually relevant answers for queries from the specified domains:
Healthcare: Example - Symptoms of common diseases, appointment scheduling, etc.
Insurance: Example - Policy details, claim processes, coverage-related queries, etc.
Finance: Example - Investment options, loan inquiries, credit card issues, etc.
Retail: Example - Product availability, pricing, order tracking, etc.
Out-of-Domain Handling:
For queries outside the specified domains, the chatbot must respond with:
"I can only assist with queries related to Healthcare, Insurance, Finance, or Retail.

## Prerequisites

Make sure you have the following installed:
- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. **Set Up Environment Variables**
   - Create a `.env` file in the project root.
   - Add the necessary environment variables. For example:
     ```env
     HF_TOKEN="token"
     ```

## Running the Application

1. **Start the Application**
   - Run the following command to start the app using `uvicorn`:
     ```bash
     uvicorn app:app --reload
     ```
   - Replace `app:app` with the module and application instance path if different.

2. **Access the Application**
   - Open your browser and go to [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Additional Commands

- **Running Tests** (if applicable):
  ```bash
  pytest
  ```

- **Formatting Code**:
  ```bash
  black .
  ```

- **Linting Code**:
  ```bash
  flake8
  ```

## License

This project is licensed under the MIT License. See the `LICENSE` file for more information.

---

## Experiment (Quick)
  ```bash
  check Research/llama_experiment.ipynb
  ```



## Features
- **Modular Design**: Easy-to-understand components for chat generation, intent detection, and history management.
- **Customizable**: Fine-tune parameters for response generation like temperature, max tokens, and repetition penalties.
- **Scalable**: Designed to work with large language models efficiently.

---

## Components

1. **`chat_completion.py`**:
   - Handles the generation of responses using pre-trained models.
   - Supports customization of parameters like temperature, token limits, and sampling techniques.

2. **`chat_history.py`**:
   - Manages chat history to ensure contextual replies.

3. **`intent.py`**:
   - Identifies user intent for more tailored responses.

4. **`llm.py`**:
   - Abstracts the interaction with large language models.

5. **`load_model.py`**:
   - Loads and initializes the language model and tokenizer.

6. **`prompt_template.py`**:
   - Contains customizable templates for structuring input prompts.

---

# Component: Intent Extraction (`get_intent`)

## Overview
The `get_intent` function is designed to extract user intent from a model's response in a structured format. It utilizes a pre-trained language model and tokenizer to process user queries and ensure the output is returned as a JSON object. If the initial output is not in the expected format, the function attempts to rectify and reprocess it.

---

## Functionality

### **Key Responsibilities**
1. Processes user queries and system messages.
2. Generates responses using a pre-trained language model.
3. Extracts and validates the intent as a JSON object.
4. Handles errors in the initial response and retries with additional clarification to the model.

---

## Arguments

- **`model`**:
  The pre-trained language model used for generating responses.

- **`tokenizer`**:
  The tokenizer associated with the pre-trained model for encoding inputs and decoding outputs.

- **`system_message`**:
  A system-level message to provide context for the conversation.

- **`query`**:
  The user query for which intent is to be extracted.

- **`device`**:
  The device to execute the model on (e.g., `cuda` for GPU or `cpu`).

---

## Process Flow

1. **Message Preparation**:
   - Constructs a list of messages including the system message and the user query.
   - Encodes the messages using the tokenizer with chat-specific templates.

2. **Response Generation**:
   - Generates the model's output using `model.generate` with parameters like:
     - `max_new_tokens`: Limits the length of the response.
     - `temperature`: Controls randomness in the output.
     - `pad_token_id` and `eos_token_id`: Handle tokenization.
   - Decodes the output to extract the intent as text.

3. **JSON Parsing**:
   - Attempts to parse the intent as a JSON object.
   - Handles parsing errors by sending a corrective message back to the model, explicitly requesting a valid JSON response.

4. **Error Handling**:
   - Logs and retries generating the intent if parsing fails.
   - Defaults to a generic intent (`{'intent':'other'}`) if repeated parsing attempts fail.

---

## Notes
- Ensure the model is fine-tuned for intent extraction to improve accuracy.
- Handle scenarios where the output is not JSON by providing robust corrective prompts.
- Performance may vary depending on the model size and quantization.

# Component: ChatCompletion

## Overview
The `ChatCompletion` class is designed to handle conversational AI tasks, providing capabilities for generating context-aware responses based on user inputs and maintaining chat history. It uses pre-trained language models and tokenizers for efficient and intelligent response generation.

---

## Class Details

### **Initialization (`__init__`):**
- Initializes the `ChatCompletion` instance with configurable settings.
- **Parameters:**
  - `device`: Optional parameter to specify the computation device (e.g., `cuda:0` for GPU or `cpu`). Defaults to GPU if available, otherwise CPU.
  - `model`: Pre-trained language model instance.
  - `tokenizer`: Tokenizer associated with the model.
- **Error Handling:**
  - Raises an exception if `model` or `tokenizer` is not provided.

---

### **Generate a Response (`create`):**
- Generates a conversational response based on input messages.

#### **Input Arguments:**
- `messages`: A list of dictionaries representing the chat context.
- `kwargs`: Optional keyword arguments to customize the response generation:
  - `max_new_tokens`: Maximum number of tokens to generate (default: 50).
  - `temperature`: Controls randomness in output (default: 0.6).
  - `top_p`: Nucleus sampling parameter for diverse outputs (default: 0.9).
  - `do_sample`: Enables sampling over deterministic decoding (default: False).
  - `repetition_penalty`: Penalizes repetitive text (default: 1.0).
  - `eos_token_id`: End-of-sequence token ID.
  - `pad_token_id`: Padding token ID (default: `tokenizer.eos_token_id`).

#### **Process Flow:**
1. Encodes the input messages using a chat-specific template provided by the tokenizer.
2. Converts the input into tensors compatible with the specified device (GPU/CPU).
3. Passes the encoded input to the model's `generate` method with the specified parameters.
4. Decodes the output to extract a human-readable response.
5. Cleans the response by removing unnecessary tokens or formatting artifacts.

#### **Output:**
- Returns a clean, contextually appropriate response as a string.

---

### **Clear Chat History (`clear_history`):**
- Resets the chat history to ensure no residual context affects subsequent interactions.

---
---

## Notes
- Ensure the `model` and `tokenizer` are compatible for optimal performance.
- Adjust parameters like `temperature` and `top_p` for specific conversational needs.
- Use `clear_history` to reset the context when starting a new conversation.


# ChatHistory Class

The `ChatHistory` class manages a chat message history, providing functionality to add, retrieve, clear, and check the length of the chat history. It stores messages as a list of dictionaries, where each dictionary includes the role of the sender (e.g., 'user' or 'assistant') and the message content.

## Features:
- **Add messages**: Store chat messages with a specified role and content.
- **Retrieve history**: Get the entire chat history as a list of dictionaries.
- **Clear history**: Clear all stored chat messages.
- **Length of history**: Get the number of messages stored in the chat history.

## Methods:

### `add_message(role: str, content: str) -> None`
Adds a new message to the chat history.
- **role**: The sender's role (e.g., 'user' or 'assistant').
- **content**: The content of the message.

### `get_chat_history() -> List[Dict]`
Returns the entire chat history as a list of dictionaries.
- **Returns**: A list of dictionaries, each containing `role` and `content` of the message.

### `clear_chat_history() -> None`
Clears all messages from the chat history.
- **Returns**: None.

### `__len__() -> int`
Returns the number of messages in the chat history.
- **Returns**: The total count of messages stored in the chat history.

# LLM Class

The `LLM` class provides functionality to load a pre-trained language model (e.g., LLaMA) with optional quantization, track GPU memory usage, and manage model loading from both local storage and the Hugging Face Hub.

## Features:
- **Quantized Model Loading**: Loads a quantized model when GPU memory is low (less than 60 GB).
- **GPU Memory Tracking**: Monitors and prints GPU memory before and after loading the model.
- **Automatic Device Selection**: Automatically selects the device (`cuda:0` or `cpu`) based on GPU availability.
- **Model Caching**: Saves the model and tokenizer locally to avoid reloading from the Hugging Face Hub.
- **Environment Configuration**: Loads environment variables like Hugging Face authentication token (`HF_TOKEN`) from `.env` file.

## Dependencies:
- `torch`: For loading models and managing device selection.
- `subprocess`: For querying GPU memory.
- `transformers`: For model and tokenizer handling.
- `dotenv`: For loading environment variables.

## Methods:

### `__init__(self, **kwargs) -> None`
Initializes the `LLM` instance with optional parameters:
- **model_id**: The model's Hugging Face identifier (defaults to `meta-llama/Llama-3.2-3b-Instruct`).
- **device**: The device to use for loading the model (`cuda:0` or `cpu`).

### `__call__(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]`
Loads and returns the model and tokenizer based on the specified `model_id`.

### `__get_model(self, model_id: str = 'meta-llama/Llama-3.2-3B-Instruct') -> Tuple[AutoModelForCausalLM, AutoTokenizer]`
- **Decorator**: Uses the `gpu_memory_decorator` to track GPU memory usage.
- Loads the model either in quantized or unquantized format depending on available GPU memory.
- **Quantized Model**: If GPU memory is less than or equal to 60 GB, a quantized model (8-bit or 4-bit) is loaded.
- **Unquantized Model**: If GPU memory is more than 60 GB, the full unquantized model is loaded.

### `_get_gpu_memory(self) -> float`
Queries the available GPU memory using `nvidia-smi` and returns the free memory in MB as a float.

### `gpu_memory_decorator(func)`
A decorator function that tracks the GPU memory before and after the execution of a model loading function.

## Environment Variables:
- **HF_TOKEN**: A Hugging Face authentication token required for downloading models.

## Usage Example:

```python
# Instantiate the LLM class
llm = LLM(model_id="meta-llama/Llama-3.2-3b-Instruct", device="cuda:0")

# PromptTemplate Class

The `PromptTemplate` class is designed to manage and format prompts for conversational chatbots and intent classifiers. It allows filling dynamic values into predefined prompts using Jinja2 templating.

## Features:
- **Customizable Prompts**: Supports setting custom prompts for chatbot and intent classification.
- **Dynamic Key-Value Filling**: Automatically fills keys with provided values into prompts using Jinja2 templating.
- **Validation**: Ensures the `fill_keys` parameter is a list of strings for valid key-value pair formatting.

## Dependencies:
- `jinja2`: For template rendering.

## Methods:

### `__init__(self, chatbot_prompt: str = chatbot_prompt, intent_classifier_prompt: str = intents_prompt, fill_keys: bool = True) -> None`
Initializes the `PromptTemplate` instance with:
- **chatbot_prompt**: The prompt template for the chatbot (defaults to `chatbot_prompt` constant).
- **intent_classifier_prompt**: The prompt template for the intent classifier (defaults to `intents_prompt` constant).
- **fill_keys**: A boolean to specify whether keys should be filled with values in the template (defaults to `True`).

### `format(self, chatbot_prompt=True, **kwargs: Dict) -> Dict`
Formats the prompt by filling in dynamic values and returns the formatted prompt as a dictionary.
- **chatbot_prompt**: A boolean to choose which prompt to format (defaults to `True`, for chatbot prompt).
- **kwargs**: Dictionary containing dynamic values to replace placeholders in the prompt.
  - `keys`: List of keys to replace in the prompt template.
  - `values`: Corresponding list of values to fill for the placeholders.

### Example Usage:

```python
# Initialize the PromptTemplate class
prompt_template = PromptTemplate()

# Format the chatbot prompt with dynamic values
formatted_prompt = prompt_template.format(
    chatbot_prompt=True, 
    keys=['user_name', 'user_intent'], 
    values=['John Doe', 'greeting']
)


# FastAPI Chatbot API

This FastAPI application provides an endpoint for generating responses from a conversational chatbot. It integrates several components including model loading, intent classification, prompt template formatting, and chat history management.

## Features:
- **Intent Classification**: Identifies the user's intent based on the input query.
- **Chat History**: Keeps track of the ongoing conversation to provide context to the model.
- **Generative Response**: Uses a language model to generate contextually relevant responses.

## Dependencies:
- `fastapi`: For creating the web API.
- `logging`: For logging the application's activity.
- `Conversational_ChatBot`: Custom modules for model loading, intent classification, prompt templating, and chat history.
- `typing`: For type hinting.

## API Endpoint:

### `POST /generate-response/`
Generates a response based on the user query and session ID.

#### Request Body:
- **user_query**: The query sent by the user (string).
- **session_id**: A unique identifier for the user's session (string).

Example:
```json
{
    "user_query": "What's the weather like today?",
    "session_id": "12345"
}

```


[Healthcare](https://www.loom.com/share/7dcb09249de64566af6dd565cb3932f0?sid=5a8ef02a-c758-4b3d-a2e7-ffc16d522ffb)
[Insurance](https://www.loom.com/share/d8f4070c5d5d4a909b9fd9f680729457?sid=4edf0a97-34ce-4923-8abf-6a3e89d5f6d7)
[Retail](https://www.loom.com/share/dff649e3149c404f96334b8e0b48396a?sid=0337e364-ed2d-4a35-9418-37ffd22ebdcf)
[Finance](https://www.loom.com/share/7dcb09249de64566af6dd565cb3932f0?sid=5a8ef02a-c758-4b3d-a2e7-ffc16d522ffb)

<iframe width="640" height="360" src="https://www.loom.com/embed/d8f4070c5d5d4a909b9fd9f680729457?sid=c2416734-8a21-4ab8-82d8-a64f191946eb" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>