import os
import time
import threading
from typing import *

from transformers import AutoTokenizer
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

from src.constants import *
from src.utils import log_prompt, filter_old_messages, post_process_output, create_logger

logger = create_logger(LOGFILE_PATH)

class MistralAPIClient:
    
    def __init__(self,
                 model: str = API_ENDPOINT_NAME,
                 temperature: float = TEMPERATURE,
                 api_key:Optional[str]=None
                 ):
        """_summary_

        Args:
            model (str, optional): _description_. Defaults to API_ENDPOINT_NAME.
            temperature (float, optional): _description_. Defaults to TEMPERATURE.
            api_key (Optional[str], optional): _description_. Defaults to None.
        """
        
        self.model_name = model
        self.client = self._get_client(api_key)
        self.temperature = temperature
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        try:
            self.comet_project_name = f"{os.environ['COMET_PROJECT_NAME']}-monitor-prompts" 
        except KeyError:
            raise RuntimeError(
                "Please set the COMET_PROJECT_NAME environment variable."
            )
            
        logger.info("Successfully Initiated the LLM API Client")
        
            
    def _get_client(self,
                    api_key:Optional[str]=None
                    )->MistralClient:
        """_summary_

        Args:
            api_key (Optional[str], optional): _description_. Defaults to None.

        Returns:
            MistralClient: _description_
        """
        
        logger.info(f"Instantiating a {self.model_name} API client.")
        
        if api_key is None:
            try:
                api_key = os.environ['MISTRAL_API_KEY']
            except:
                raise ValueError("Either pass the mistral API key while instantiating this class or set 'MISTRAL_API_KEY' environment variable.")

        return MistralClient(api_key=api_key)

    
    def _get_templated_query(self, question):
        return f"""<<<\nQUESTION: {question} >>>.\n\n\n\n\n"""


    def _get_inference_prompt(self, question:str, chat_history:List[Tuple[str,str]]) -> List[ChatMessage]:
        
        logger.info(f"Preparing prompt for response generation")
        
        messages = []
        
        for index,(past_question, response_text) in enumerate(chat_history):
            
            if not index:
                prompt = INSTRUCTION_TEMPLATE + self._get_templated_query(past_question)
                messages.append(ChatMessage(role='user',content=prompt))
            else:
                prompt = self._get_templated_query(past_question)
                messages.append(ChatMessage(role='user',content=prompt))
                
            messages.append(ChatMessage(role='assistant',content=response_text))
            messages = filter_old_messages(messages, self.tokenizer)
        
        if not messages:
            prompt = INSTRUCTION_TEMPLATE + self._get_templated_query(question)
            messages.append(ChatMessage(role='user',content=prompt))
        else:
            prompt = self._get_templated_query(question)
            messages.append(ChatMessage(role='user',content=prompt))

        return messages
    
    
    def _log_prompt_data(self, question, response, messages, chat_history, time_taken):
        
        logger.info(f"Logging the prompt data onto comet-ml")
        
        prompt_tokenids = self.tokenizer.apply_chat_template(messages)
        
        num_prompt_tokens = len(prompt_tokenids)
        num_response_tokens = len(self.tokenizer(response))
        total_tokens = num_prompt_tokens + num_response_tokens
        
        
        log_dict = {"project":self.comet_project_name,
                    "model": self.model_name,
                    "prompt":self.tokenizer.decode(prompt_tokenids),
                    "prompt_template_variables":{'question':question, 'chat_history':chat_history},
                    "prompt_tokens": num_prompt_tokens,
                    "total_tokens": total_tokens,
                    "actual_new_tokens": num_response_tokens,
                    "duration": time_taken,
                    "output": response}
        
        log_prompt(log_dict)
    

    def stream_answer(self, question, chat_history):
        
        messages = self._get_inference_prompt(question, chat_history)
        messages = [ChatMessage(role='user',content="Say 'Hi'.")]
        response = ''
        
        start = time.time()
        
        logger.info(f"Calling the API Client for response generation")
        stream_response = self.client.chat_stream(model=self.model_name, messages=messages, temperature=self.temperature)
        
        for chunk in stream_response:
            text = chunk.choices[0].delta.content
            response += text
            
            yield post_process_output(response)
            
        response_time = time.time() - start
        
        logger.info(f"Response generation Completed..")
        
        logging_thread = threading.Thread(target=self._log_prompt_data, args=(question, response, messages, chat_history, response_time))
        logging_thread.start() 