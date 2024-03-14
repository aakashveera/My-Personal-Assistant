from langchain.callbacks.base import BaseCallbackHandler
from typing import Dict, Any

from src.constants import *
from src.utils import log_prompt

class CometLLMMonitoringHandler(BaseCallbackHandler):
    """
    A callback handler for monitoring LLM models using Comet.ml.

    Args:
        project_name (str): The name of the Comet.ml project to log to.
        llm_model_id (str): The ID of the LLM model to use for inference.
    """
    
    def __init__(
        self,
        project_name: str = None,
        llm_model_id: str = MODEL_NAME
    ):
        self._project_name = project_name
        self._llm_model_id = llm_model_id

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """
        A callback function that logs the prompt and output to Comet.ml.

        Args:
            outputs (Dict[str, Any]): The output of the LLM model.
            **kwargs (Any): Additional arguments passed to the function.
        """
        
        #Add project name, answer and model to the data dict before passing it to prompt logger        
        log_data_dict = kwargs["metadata"]
        log_data_dict['project'] = self._project_name
        log_data_dict['output'] = outputs["answer"]
        log_data_dict['model'] = self._llm_model_id
        
        log_prompt(log_data_dict)