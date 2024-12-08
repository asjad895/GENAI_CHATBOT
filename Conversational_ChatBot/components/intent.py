import json

async def get_intent(chat, system_message, query, device):
    """
    Extracts intent from the model's response.

    Args:
        chat: The chat object for model interaction.
        system_message: The system message for the chat.
        query: The user query to extract intent.
        device: The device to run the model on (e.g., 'cuda' or 'cpu').

    Returns:
        A dictionary representing the extracted intent.
    """
    messages = [
        system_message,
        {"role": "user", "content": query},
    ]
    try:
        # Generate response using the chat object
        response = await chat.create(messages, max_new_tokens=50, temperature=0.6)
        intent_pred = response.strip()

        try:
            # Parse the response as JSON
            intent_pred = json.loads(intent_pred)
        except Exception as e:
            print(f"Error parsing initial response: {e}\n{intent_pred}")
            # Request clarification and correction from the model
            correction_messages = [
                {"role": "assistant", "content": intent_pred},
                {
                    "role": "user",
                    "content": f"In your previous response {intent_pred} got this error {str(e)}. "
                               "Ensure it is a JSON response that can be consumed by an automated system. "
                               "Do not generate anything apart from JSON output with keys and values as "
                               f"expected based on the input: {intent_pred}."
                },
            ]
            correction_response = await chat.create(correction_messages, max_new_tokens=50, temperature=0.6)
            intent_pred = correction_response.strip()

            print(f"Corrected response: {intent_pred}")
            try:
                # Parse the corrected response as JSON
                intent_pred = json.loads(intent_pred)
            except:
                intent_pred = {'intent': 'other'}

    except Exception as e:
        print(f"Error during chat creation: {e}")
        intent_pred = {'intent': 'other'}

    return intent_pred
