import re
import yaml
import logging
import comet_llm
from pathlib import Path
from typing import List, Tuple, Union, Dict

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


def parse_chat_history(chat_text:str)->List[Tuple[str,str]]:
    """Helper function to parse the chat history stored in memory chain

    Args:
        chat_text (str): Chat text stored in Langchain's memory

    Returns:
        List[Tuple[str,str]]: Parsed chat Items.
    """
    
    # Define regex patterns for Human and AI parts
    human_pattern = re.compile(r'Human: (.*?)(?:AI:|$)', re.DOTALL)
    ai_pattern = re.compile(r'AI: (.*?)(?:Human:|$)', re.DOTALL)

    # Find all matches for Human and AI parts
    human_matches = human_pattern.findall(chat_text)
    ai_matches = ai_pattern.findall(chat_text)

    return [(human.strip(),ai.strip()) for human,ai in zip(human_matches,ai_matches)]


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