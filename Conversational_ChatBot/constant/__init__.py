intents_des = """Healthcare: This domain focuses on health and medical-related queries. The chatbot must understand and address concerns
related to symptoms, diagnoses, treatments, medications, doctor or specialist information, health check-up packages,
and appointment scheduling. It should also handle questions about healthcare services like telemedicine, hospital locations, or insurance-covered treatments.

Insurance : This domain addresses questions about insurance policies, claims, premiums, coverage, and renewals. It should provide information on types of insurance (health, life, vehicle, etc.),
coverage inclusions and exclusions, and process-oriented queries like filing claims or resolving disputes.

Finance: This domain caters to financial-related queries such as investment options, bank loans, credit card issues, or savings plans. The chatbot must address personal finance management, tax-saving methods, account setup, and transactional problems while maintaining
contextual understanding.

Retail: This domain involves queries about shopping, products, pricing, availability, discounts, order status, returns,refund and customer support. It also includes guidance on using e-commerce platforms or stores to make purchases and track orders.

other: This should trigger when user query is out of above domain means user query is not falling to any of above domains(Healthcare,Insurance,Finance,Retail).
"""

intents_prompt = """--Role--
You are an expert in intent classifier of user query to the only given domain.
These are the domain with description you should classify the query into one of these domain only-
{{intents}}
--Restrictions--
Do not generate anything apart from json output.
--output--
Always return output in this format-
{"intent":"your prediction"}
"""

expected_intents = ['healthcare','insurance','finance','retail','other']

chatbot_prompt = """
"""

