import os
from typing import List, Tuple, Iterable

from langchain import chains
from langchain.callbacks import FileCallbackHandler
from langchain.memory import ConversationBufferMemory

from src.constants import *
from src.utils import create_logger, post_process_output
from .model import build_pipeline
from .chains import LLMChain, StatelessMemorySequentialChain
from .handlers import CometLLMMonitoringHandler

logger = create_logger(LOGFILE_PATH)

class LangChainChatBot:
    """
    A language chain bot that uses a LLM model to generate responses to user inputs.

    Args:
        llm_model_id (str, optional): The Hugging Face model ID to use to instantiate a LLM. Defaults to MODEL_NAME.
        device (str, optional): Device to use for inferencing. Defaults to DEVICE.
        llm_inference_max_new_tokens (int, optional): The maximum number of new tokens to generate during inference. Defaults to MAX_NEW_TOKENS.
        llm_inference_temperature (float, optional): The temperature to use during inference. Defaults to TEMPERATURE.
        streaming (bool, optional): Whether to use the Hugging Face streaming API for inference. Defaults to streaming.
        
    Attributes:
        bot_chain (Chain): The language chain that generates responses to user inputs.
    """
    
    def __init__(
        self,
        llm_model_id: str = MODEL_NAME,
        device = DEVICE,
        llm_inference_max_new_tokens: int = MAX_NEW_TOKENS,
        llm_inference_temperature: float = TEMPERATURE,
        streaming: bool = True,
    ):
        self._llm_model_id = llm_model_id
        self._llm_inference_max_new_tokens = llm_inference_max_new_tokens
        self._llm_inference_temperature = llm_inference_temperature
        self._device = device

        logger.info("Building a huggingface inference pipeline")
        self._llm_agent, self._streamer, self._eos_token = build_pipeline(
            model_name=self._llm_model_id,
            device=self._device,
            gradient_checkpointing=False,
            use_streamer=streaming
        )
        self.bot_chain = self.build_chain() #Build the chabot chain
        
        
    @property
    def is_streaming(self) -> bool:
        return self._streamer is not None
    
    
    def _get_comet_project_name(self) -> str:
        """Get comet project name for logging the prompts via environment variable.

        Returns:
            str: Comet-ml project name as a string. 
        """
        
        try:
            comet_project_name = os.environ["COMET_PROJECT_NAME"]
        except KeyError:
            raise RuntimeError(
                "Please set the COMET_PROJECT_NAME environment variable."
            )
            
        return comet_project_name
    
    
    def build_chain(self)-> chains.SequentialChain:
        """
        Builds and returns a chatbot chain. This chain is designed to take query as input, 
        Turn the query onto a prompt as per LLM's requirement and
        then feed it to the model to get a response that is relevant to the user's question.

        Returns: [chains.SequentialChain]: The constructed chatbot chain.
        """
        
        logger.info("Building 1/2 - FinancialBotQAChain. Initalizaing a prompt logger and LLM agent")
        
        comet_project_name = self._get_comet_project_name()
        
        #Instantiate a callback handler to log prompts after response generation.
        callbacks = [
            CometLLMMonitoringHandler(
                project_name=f"{comet_project_name}-monitor-prompts",
                llm_model_id=self._llm_model_id
            )
        ]
        
        #Instantiate a LLM chain for generating response.
        llm_generator_chain = LLMChain(
            hf_pipeline=self._llm_agent,
            callbacks=callbacks,
        )
        
        logger.info("Building 2/2 - Connecting chains into SequentialChain")
        log_handler = FileCallbackHandler(LOGFILE_PATH)
        
        seq_chain = StatelessMemorySequentialChain(
            memory=ConversationBufferMemory(
                memory_key="chat_history",
                input_key="question",
                output_key="answer",
                return_messages=True
            ),
            chains=[llm_generator_chain],
            input_variables=["question", "to_load_history"],
            output_variables=["answer"],
            verbose=True,
            callbacks=[log_handler]
        )

        logger.info("Done building SequentialChain.")
        
        return seq_chain
    
    
    def answer(
        self,
        question: str,
        chat_history: List[Tuple[str, str]] = None,
    ) -> str:
        """Given a question and past chat messages, generates a response
           to the current query using the initialized LLM Chain.

        Args:
            question (str): A question provided by the user.
            chat_history (List[Tuple[str, str]], optional): List containing the past conversation's
                         query & responses as a list of tuples. Defaults to None.

        Returns:
            str: A reponse generated for the query.
        """

        inputs = {
            "question": question,
            "to_load_history": chat_history if chat_history else [],
        }
        response = self.bot_chain.run(inputs) #Generate response using the llm chain.

        return response
    

    def stream_answer(self) -> Iterable[str]:
        """Stream the answer from the LLM after each token is generated"""

        partial_answer = ""
        
        #Iterate through each streamed token,
        #Return the token after post-processing untill eos-token is generated.
        for new_token in self._streamer:
            if new_token != self._eos_token:
                partial_answer += new_token

                yield post_process_output(partial_answer)