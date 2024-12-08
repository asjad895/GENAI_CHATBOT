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
  check dir Research/llama
  ```


