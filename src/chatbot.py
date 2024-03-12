import os
from typing import List, Tuple, Iterable

from langchain import chains
from langchain.memory import ConversationBufferWindowMemory

from .constants import *
from .utils import create_logger
from .model import build_pipeline
from .chains import LLMChain, StatelessMemorySequentialChain
from .handlers import CometLLMMonitoringHandler

logger = create_logger(LOGFILE_PATH)

class PersonalAssistant:
    """
    A language chain bot that uses a language model to generate responses to user inputs.

    Args:
        llm_model_id (str): The ID of the Hugging Face language model to use.
        device (str): Device to use for inferencing.
        llm_inference_max_new_tokens (int): The maximum number of new tokens to generate during inference.
        llm_inference_temperature (float): The temperature to use during inference.
        streaming (bool): Whether to use the Hugging Face streaming API for inference.
        
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

        self._llm_agent, self._streamer, self._eos_token = build_pipeline(
            model_name=self._llm_model_id,
            device=self._device,
            gradient_checkpointing=False,
            use_streamer=streaming
        )
        self.bot_chain = self.build_chain()
        
        
    @property
    def is_streaming(self) -> bool:
        return self._streamer is not None
    
    
    def build_chain(self)-> chains.SequentialChain:
        """
        Constructs and returns a financial bot chain.
        This chain is designed to take as input the user description, `about_me` and a `question` and it will
        connect to the VectorDB, searches the financial news that rely on the user's question and injects them into the
        payload that is further passed as a prompt to a financial fine-tuned LLM that will provide answers.


        2. LLM Generator: Once the context is extracted,
        this stage uses it to format a full prompt for the LLM and
        then feed it to the model to get a response that is relevant to the user's question.

        Returns
        -------
        chains.SequentialChain
            The constructed financial bot chain.

        Notes
        -----
        The actual processing flow within the chain can be visualized as:
        [about: str][question: str] > ContextChain >
        [about: str][question:str] + [context: str] > FinancialChain >
        [answer: str]
        """
        
        
        logger.info("Building 1/2 - FinancialBotQAChain")

        try:
            comet_project_name = os.environ["COMET_PROJECT_NAME"]
        except KeyError:
            raise RuntimeError(
                "Please set the COMET_PROJECT_NAME environment variable."
            )
            
        callbacks = [
            CometLLMMonitoringHandler(
                project_name=f"{comet_project_name}-monitor-prompts",
                llm_model_id=self._llm_model_id
            )
        ]
        
        llm_generator_chain = LLMChain(
            hf_pipeline=self._llm_agent,
            callbacks=callbacks,
        )
        
        
        logger.info("Building 2/2 - Connecting chains into SequentialChain")
        seq_chain = StatelessMemorySequentialChain(
            memory=ConversationBufferWindowMemory(
                memory_key="chat_history",
                input_key="question",
                output_key="answer",
                k=10,
            ),
            chains=[llm_generator_chain],
            input_variables=["question", "to_load_history"],
            output_variables=["answer"],
            verbose=True,
        )

        logger.info("Done building SequentialChain.")
        
        return seq_chain
    
    
    def answer(
        self,
        question: str,
        to_load_history: List[Tuple[str, str]] = None,
    ) -> str:
        """
        Given a short description about the user and a question make the LLM
        generate a response.

        Parameters
        ----------
        question : str
            User question.

        Returns
        -------
        str
            LLM generated response.
        """

        inputs = {
            "question": question,
            "to_load_history": to_load_history if to_load_history else [],
        }
        response = self.bot_chain.run(inputs)

        return response
    

    def stream_answer(self) -> Iterable[str]:
        """Stream the answer from the LLM after each token is generated after calling `answer()`."""

        assert (
            self.is_streaming
        ), "Stream answer not available. Build the bot with `use_streamer=True`."

        partial_answer = ""
        for new_token in self._streamer:
            if new_token != self._eos_token:
                partial_answer += new_token

                yield partial_answer