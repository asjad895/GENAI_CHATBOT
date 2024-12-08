import json
def get_intent(model, tokenizer, system_message, query,device):
  """
  Extracts intent from the model's response.

  Args:
    model: The pre-trained language model.
    tokenizer: The tokenizer associated with the model.
    system_message: The system message for the chat.
    device: The device to run the model on (e.g., 'cuda' or 'cpu').

  Returns:
    A dictionary representing the extracted intent.
  """
  messages = [
      system_message,
      {"role": "user", "content": query},
  ]
  encoded_input = tokenizer.apply_chat_template(messages, tokenize = True,return_dict=True,return_tensors="pt").to(device)
  outputs = model.generate(
      **encoded_input,
      max_new_tokens=50,
      temperature=0.6,
      pad_token_id=tokenizer.eos_token_id,
      eos_token_id=tokenizer.eos_token_id,
      )
  output = tokenizer.decode(outputs[0])
  intent_pred = output.split('assistant<|end_header_id|>')[-1].strip().split('<|eot_id|>')[0].strip()

  try:
    intent_pred = json.loads(intent_pred)
  except Exception as e:
    print(f"Error parsing initial response: {e}\n{intent_pred}")
    messages = [
        {"role":'assistant','content':intent_pred},
        {"role": "user", "content": f"In your previous response {intent_pred} got this error {str(e)} \n make sure it is json response so that is should consume by automated system.Do not generate anything apart from json output as keys and value as same as {intent_pred}"},
    ]
    encoded_input = tokenizer.apply_chat_template(messages, tokenize = True,return_dict=True,return_tensors="pt").to(device)
    outputs = model.generate(
        **encoded_input,
        max_new_tokens=50,
        temperature=0.6,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        )
    output = tokenizer.decode(outputs[0])
    intent_pred = output.split('assistant<|end_header_id|>')[-1].strip().split('<|eot_id|>')[0].strip()
    print(f"Corrected response: {intent_pred}")

    try:
      intent_pred = json.loads(intent_pred)
    except:
      intent_pred = {'intent':'other'}

  return intent_pred
