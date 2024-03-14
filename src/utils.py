import yaml
import logging
from pathlib import Path
from typing import List, Tuple, Union, Dict

import comet_llm
from transformers import AutoTokenizer
from langchain.schema.messages import HumanMessage, AIMessage


def create_logger(log_file_path:str)->logging.Logger:
    """
    Creates and configures a logger with both filehandler and consolde handler at INFO level.

    Args:
        log_file_path (str): The path to the log file.

    Returns:
        logging.Logger: A configured logger instance.
    """
    # Create a logger
    logger = logging.getLogger(__name__)
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Create a file handler for the log file
        file_handler = logging.FileHandler(log_file_path)
        
        # Create a stream handler for console output
        console_handler = logging.StreamHandler()

        # Create a formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add both handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


def parse_chat_history_as_tuples(
    message_list: List[Union[HumanMessage, AIMessage]]
    )->List[Tuple[str,str]]:
    
    """Helper function to parse the chat history's content 

    Args:
        message_list (List): Previous chat messages text stored in Langchain's memory

    Returns:
        List[Tuple[str,str]]: Parsed chat Items as a List of tuples.[(human_query, ai_response)].
    """
    
    parsed_messages_list = []  
    
    try:
        assert len(message_list)%2 == 0 #Each query should have a response and so the number of chat items should be even.
        
        for index in range(0,len(message_list)-2,2): 
            #Item at index is query and item and index+1 will its coresponding answer. Fetch the content and append the pair as a tuple after striping.
            parsed_messages_list.append((message_list[index].content.strip(), message_list[index+1].content.strip()))
            
    except:
        human_messages, ai_messages = [], []
        
        #Incase if the count doesn't match up. Iterate through each message
        #If the message is Human generated add to human_messages list else add to ai_messages list
        for message in message_list:
            if type(message)==HumanMessage:
                human_messages.append(message.content.strip())
            else:
                ai_messages.append(ai_messages.content.strip())
        
        #Zip and pack the messages as tuples.
        for human,ai in zip(human_messages, ai_messages):
            parsed_messages_list.append((human,ai))      
    
    return parsed_messages_list


def filter_old_messages(messages: List, tokenizer: AutoTokenizer)->List:
    """Function to the filter out the old messages on history before preparing the prompt. 
       Filters the past (query, response) pair from history when the total tokens count exceeds MAX_ACCEPTED_TOKENS.
       Mistral Instruct model has a context length of 4096.

    Args:
        messageList (List): A list containing the chat prompts. 
        tokenizer (AutoTokenizer): Tokenizer object to count the number of token in the processed prompt.

    Returns:
        List: A list containing the filtered messages
    """
    
    MAX_ACCEPTED_TOKENS = 3500
    
    #Convert the messages into prompt tokens.
    tokens = tokenizer.apply_chat_template(messages)
    
    #Remove the question and answer pair from the messages untill the total token count in the prompt exceeds the limit.
    #Always Leave the first 2 conversations to preserve the initial instructions. 
    #Remove the prompts from the middle for getting better response. 
    while len(tokens)>MAX_ACCEPTED_TOKENS:
        
        if len(messages)>4:
            messages.pop(4)
            messages.pop(4)
        elif len(messages)>2:
            messages.pop(2)
            messages.pop(2)
        else:
            break
        
        #Recompute the new prompt tokens based on filtered messages.
        tokens = tokenizer.apply_chat_template(messages)
        
    return messages


def log_prompt(log_dict: Dict[str,Union[str,int]]):
    """Log the query, response, prompt and other metadata onto the comet-ml dashboard for tracking.

    Args:
        log_dict (Dict[str,Union[str,int]]): A dictionary containing all the necassary data and metadata for logging the prompt.
    """
    
    comet_llm.log_prompt(
        project=log_dict['project'],
        prompt=log_dict['prompt'],
        output=log_dict['output'],
        prompt_template_variables=log_dict['prompt_template_variables'],
        metadata={
            "usage.prompt_tokens": log_dict['prompt_tokens'],
            "usage.total_tokens": log_dict['total_tokens'],
            "usage.actual_new_tokens": log_dict['actual_new_tokens'],
            "model": log_dict['model'],
        },
        duration=log_dict['duration']
        )
   
 
def post_process_output(text: str)-> str:
    """Post-process the commonly occuring errors/patterns from the generated response.

    Args:
        text (str): Response string generated by the AI model.

    Returns:
        str: Response string after applying post-processing.
    """
    
    #Remove the trailing spaces.
    text = text.lstrip()
    
    #Remove the 'Diya:' token from the beginning of the response if it exists. 
    if 'Diya:' == text[:5]:
        return text[5:]
    
    return text


def convert_chat_history_as_string(messages: List[Union[HumanMessage, AIMessage]])-> str:
    """Convert the chat_history list with conversations into a single string.
       This is necassary while logging the prompt details.

    Args:
        messages (List[Union[HumanMessage, AIMessage]]): A list containing chat history.

    Returns:
        str: A concatenated string containing all the all the past queries and responses.
    """
    
    result_string = ''
    
    #Iterate through each message and figure out whether the message is human provided or LLM generated.
    #If human generated append to the result with 'Human:' prefix else append to the result with 'AI:' prefix followed by a new line.
    for message in messages:
        if type(message)==HumanMessage:
            result_string += f'Human: {message.content.strip()}\n'
        else:
            result_string += f'AI: {message.content.strip()}\n'
            
    return result_string