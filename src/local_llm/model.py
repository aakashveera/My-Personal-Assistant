from typing import Tuple, Optional, List

import torch
from langchain.llms import HuggingFacePipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
    pipeline
)

from src.constants import *

class StopOnTokens(StoppingCriteria):
    """
    A stopping criteria that stops generation when a specific token is generated.

    Args:
        stop_ids (List[int]): A list of token ids that will trigger the stopping criteria.
    """

    def __init__(self, stop_ids: List[int]):
        super().__init__()

        self._stop_ids = stop_ids

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        """
        Check if the last generated token is in the stop_ids list.

        Args:
            input_ids (torch.LongTensor): The input token ids.
            scores (torch.FloatTensor): The scores of the generated tokens.

        Returns:
            bool: True if the last generated token is in the stop_ids list, False otherwise.
        """

        for stop_id in self._stop_ids:
            if input_ids[0][-1] == stop_id:
                return True

        return False

    
def get_model(
    model_name:str = 'mistralai/Mistral-7B-Instruct-v0.2',
    device: str = 'cuda:0',
    gradient_checkpointing: bool = True,
) -> AutoModelForCausalLM:
    """
    Function that builds a 4-bit quantized LLM model based on the given HuggingFace's model name:

    Args:
        model_name (str,optional): A pretrained huggingface model name. Defaults to mistralai/Mistral-7B-Instruct-v0.2
        device (str, optional): Device to use while loading the model.
        gradient_checkpointing (bool): Whether or not to enable gradient checkpoint while training.

    Returns:
        AutoModelForCausalLM: A built huggigface model.
    """
    
    bnb_config =  BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device
        )
    
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = (
            False  # Gradient checkpointing is not compatible with caching.
        )
    else:
        model.gradient_checkpointing_disable()
        model.config.use_cache = True
        
    return model


def get_tokenizer(
    tokenzier_name: str = 'mistralai/Mistral-7B-Instruct-v0.2'
)->AutoTokenizer:
    """Function to instantiate a pre-trained huggingface tokenizer.

    Args:
        tokenzier_name (str): Huggingface tokenizer model name

    Returns:
        AutoTokenizer: A pretrained tokenizer object
    """
    
    tokenizer = AutoTokenizer.from_pretrained(tokenzier_name)
    
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer
    
    
def build_pipeline(
    model_name:str = 'mistralai/Mistral-7B-Instruct-v0.2',
    device: str = 'cuda:0',
    gradient_checkpointing: bool = False,
    use_streamer: bool = False
) -> Tuple[HuggingFacePipeline, Optional[TextIteratorStreamer]]:
    """
    Builds a HuggingFace pipeline for text generation using a pretrained LLM.

    Args:
        model_name (str,optional): A pretrained huggingface model name. Defaults to mistralai/Mistral-7B-Instruct-v0.2
        device (str, optional): Device to use while loading the model.
        gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
        use_streamer (bool, optional): Whether to use a text iterator streamer. Defaults to False.

    Returns:
        Tuple[HuggingFacePipeline, Optional[TextIteratorStreamer]]: A tuple containing the HuggingFace pipeline
            and the text iterator streamer (if used).
    """

    model = get_model(
        model_name=model_name,
        device=device,
        gradient_checkpointing=gradient_checkpointing,
    )
    model.eval()
    
    tokenizer = get_tokenizer(model_name)

    if use_streamer:
        streamer = TextIteratorStreamer(
            tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
        )
        stop_on_tokens = StopOnTokens(stop_ids=[tokenizer.eos_token_id])
        stopping_criteria = StoppingCriteriaList([stop_on_tokens])
    else:
        streamer = None
        stopping_criteria = StoppingCriteriaList([])

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        streamer=streamer,
        stopping_criteria=stopping_criteria,
    )
    hf = HuggingFacePipeline(pipeline=pipe)

    return hf, streamer, tokenizer.eos_token