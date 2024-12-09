from Conversational_ChatBot.constant import intents_prompt,chatbot_prompt
from typing import List,Dict,Tuple,Union
from jinja2 import Template

class PromptTemplate:
    def __init__(
        self,
        chatbot_prompt :str = chatbot_prompt, 
        intent_classifier_prompt :str = intents_prompt,
        fill_keys :bool = True,
        ) ->None:
       self.chatbot_prompt = """
""" if chatbot_prompt is None else chatbot_prompt

       self.intent_classifier_prompt = """
""" if intent_classifier_prompt is None else intent_classifier_prompt
       self.fill_keys = fill_keys
       if isinstance(self.fill_keys,list):
           raise(f"'fill_keys' should be List of string which have to format in prompt.")

    async def format(self,chatbot_prompt = True,**kwargs :Dict) ->Dict:
        # keys : List[str] = ['intents'],
        # values :str = [intents_des]
        if chatbot_prompt:
            # knowledge_source = kwargs['knowledge_source']
            # del kwargs['knowledge_source']
             
             system = self.chatbot_prompt
        else:
            system = self.intent_classifier_prompt
            template = Template(system)
            system = template.render(intents = kwargs['values'][0])
        

        
        if self.fill_keys and 'keys' in kwargs:
            print("yes")
            context = dict(zip(kwargs['keys'], kwargs['values']))
            template = Template(system)
            system = template.render(**context)

    

        return {"role":"system","content":f"{system}"}

