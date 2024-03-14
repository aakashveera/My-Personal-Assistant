import yaml
import logging
from pathlib import Path
from typing import List, Tuple, Union, Dict

import comet_llm
from transformers import AutoTokenizer
from langchain.schema.messages import HumanMessage, AIMessage


def create_logger(log_file_path:str)->logging.Logger:
    """
    Create and configure a logger at INFO level with a log file.

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


def load_yaml(path: Path) -> dict:
    """
    Load a YAML file from the given path and return its contents as a dictionary.

    Args:
        path (Path): The path to the YAML file.

    Returns:
        dict: The contents of the YAML file as a dictionary.
    """

    with path.open("r") as f:
        config = yaml.safe_load(f)

    return config


def parse_chat_history_as_tuples(message_list:List[Tuple[HumanMessage, AIMessage]])->List[Tuple[str,str]]:
    """Helper function to parse the chat history stored in memory chain

    Args:
        message_list (List): Previous chat messages text stored in Langchain's memory

    Returns:
        List[Tuple[str,str]]: Parsed chat Items.
    """
    
    parsed_messages_list = []  
    
    try:
        assert len(message_list)%2 == 0 
        
        for index in range(0,len(message_list)-2,2):
            parsed_messages_list.append((message_list[index].content.strip(), message_list[index+1].content.strip()))
            
    except:
        human_messages, ai_messages = [], []
        
        for message in message_list:
            if type(message)==HumanMessage:
                human_messages.append(message.content.strip())
            else:
                ai_messages.append(ai_messages.content.strip())
        
        for human,ai in zip(human_messages, ai_messages):
            parsed_messages_list.append((human,ai))      
    
    return parsed_messages_list


def filter_old_messages(messages: List, tokenizer: AutoTokenizer)->List:
    """_summary_

    Args:
        messageList (List): _description_
        tokenizer (AutoTokenizer): _description_

    Returns:
        List: _description_
    """
    
    tokens = tokenizer.apply_chat_template(messages, return_tensors="pt")[0]
            
    while len(tokens)>3500:
        if len(messages)>4:
            messages.pop(4)
            messages.pop(4)
        elif len(messages)>2:
            messages.pop(2)
            messages.pop(2)
        else:
            break
            
        tokens = tokenizer.apply_chat_template(messages, return_tensors="pt")[0]
        
    return messages


def log_prompt(log_dict: Dict[str,Union[str,int]]):
    """_summary_

    Args:
        log_dict (Dict[str,Union[str,int]]): _description_
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
    """_summary_

    Args:
        text (str): _description_

    Returns:
        str: _description_
    """
    
    text = text.lstrip()
    
    if 'Diya:' == text[:5]:
        return text[5:]
    return text


def convert_chat_history_as_string(messages: List[Union[HumanMessage, AIMessage]])-> str:
    """_summary_

    Args:
        messages (List[Union[HumanMessage, AIMessage]]): _description_

    Returns:
        str: _description_
    """
    
    result_string = ''
    
    
    for message in messages:
        if type(message)==HumanMessage:
            result_string += f'Human: {message.content.strip()}\n'
        else:
            result_string += f'AI: {message.content.strip()}\n'
            
    return result_string