import time
from typing import Any, Dict, List, Optional, Union

from langchain import chains
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.llms import HuggingFacePipeline

from src.constants import INSTRUCTION_TEMPLATE, MODEL_NAME, LOGFILE_PATH
from src.utils import parse_chat_history_as_tuples, filter_old_messages, convert_chat_history_as_string, create_logger
from .model import get_tokenizer

logger = create_logger(LOGFILE_PATH)

class StatelessMemorySequentialChain(chains.SequentialChain):
    """
    A sequential chain that uses a stateless memory to store context between calls.

    This chain overrides the _call and prep_outputs methods to load and clear the memory
    before and after each call, respectively.
    """

    history_input_key: str = "to_load_history"

    def _call(self, inputs: Dict[str, str], **kwargs) -> Dict[str, str]:
        """
        Override _call to load history before calling the chain.

        This method loads the history from the input dictionary and saves it to the
        stateless memory. It then updates the inputs dictionary with the memory values
        and removes the history input key. Finally, it calls the parent _call method
        with the updated inputs and returns the results.
        """

        to_load_history = inputs[self.history_input_key]
        for human,ai in to_load_history:
            self.memory.save_context(
                inputs={self.memory.input_key: human},
                outputs={self.memory.output_key: ai},
            )
        memory_values = self.memory.load_memory_variables({})
        inputs.update(memory_values)

        del inputs[self.history_input_key]

        return super()._call(inputs, **kwargs)

    def prep_outputs(
        self,
        inputs: Dict[str, str],
        outputs: Dict[str, str],
        return_only_outputs: bool = False,
    ) -> Dict[str, str]:
        """
        Override prep_outputs to clear the internal memory after each call.

        This method calls the parent prep_outputs method to get the results, then
        clears the stateless memory and removes the memory key from the results
        dictionary. It then returns the updated results.
        """

        results = super().prep_outputs(inputs, outputs, return_only_outputs)

        # Clear the internal memory.
        self.memory.clear()
        if self.memory.memory_key in results:
            results[self.memory.memory_key] = ""

        return results


class LLMChain(Chain):
    """This custom chain handles LLM generation upon given prompt"""

    hf_pipeline: HuggingFacePipeline
    tokenizer = get_tokenizer(MODEL_NAME)   
    
    @property
    def input_keys(self) -> List[str]:
        """Returns a list of input keys for the chain"""

        return ["question"]

    @property
    def output_keys(self) -> List[str]:
        """Returns a list of output keys for the chain"""

        return ["answer"] 

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Calls the chain with the given inputs and returns the output"""

        #Convert the query and past chats into a prompt text as per LLM's template
        prompt = self._get_inference_prompt(
            {
                "question": inputs["question"],
                "chat_history": inputs["chat_history"],
            }
        )
        
        logger.info(f"Preparing response for the prompt")

        start_time = time.time()
        response = self.hf_pipeline(prompt["prompt"]) #Generate response to the prompt
        end_time = time.time()
        
        #Prepare metadata for logging the prompt
        duration_milliseconds = (end_time - start_time) * 1000
        num_prompt_tokens = len(self.tokenizer(prompt["prompt"]))
        num_response_tokens = len(self.tokenizer(response))
        total_tokens = num_prompt_tokens + num_response_tokens
        prompt['payload']['chat_history'] = convert_chat_history_as_string(prompt['payload']['chat_history'])

        #Log the prompt, response and prepared metadata
        #onto comet-ml dashboard using initialized logging callback handler. 
        if run_manager:
            logger.info(f"Logging the prompt data onto comet-ml")
            
            run_manager.on_chain_end(
                outputs={
                    "answer": response,
                },
                
                metadata={
                    "prompt": prompt["prompt"],
                    "prompt_template_variables": prompt["payload"],
                    "prompt_tokens": num_prompt_tokens,
                    "total_tokens": total_tokens,
                    "actual_new_tokens": num_response_tokens,
                    "duration": duration_milliseconds,
                },
            )

        return {"answer": response}
    
    
    def _get_templated_query(
        self,
        question:str
        )->str:
        
        """Converts the query provdied by the user as per the format neccasary by the LLM."""
        
        return f"""<<<\nQUESTION: {question} >>>.\n\n\n\n\n"""
    
    
    def _get_inference_prompt(
        self,
        sample: Dict[str, str]
        ) -> Dict[str, Union[str, Dict]]:
        """Convert the given query and past chat history into a prompt as per mistral's prompt template.

        Args:
            sample (Dict[str, str]): A Dict containing query and chat_history as key-value pairs.

        Returns:
            Dict[str, Union[str, Dict]]: A Dictionary containing the prepared prompt text and the 
                                         input sample used to create the prompt.
        """
        
        logger.info(f"Preparing prompt for response generation")
        
        current_query = sample['question']
        
        #Convert a chat history list into tuples of (query, response) pair
        chat_history = parse_chat_history_as_tuples(sample['chat_history'])
        
        messages = []
        
        #Iterate through the question & answer pair from the past chat history
        for index,(past_question, response_text) in enumerate(chat_history):
            
            #Attach the instruction_template as a prefix to the query template for first question alone to add general instructions.
            #Else add only the question in the template format.
            if not index:
                prompt = INSTRUCTION_TEMPLATE + self._get_templated_query(past_question)
                messages.append({"role": "user", "content": prompt})
            else:
                prompt = self._get_templated_query(past_question)
                messages.append({"role": "user", "content": prompt})

            #Add the answer generated by LLM for the query to the message list.
            messages.append({"role": "assistant", "content": response_text})
            messages = filter_old_messages(messages, self.tokenizer)
        
        #Add the current query to the messages list in the prompt format.
        #Attach instruction template as prefix to the query if current query is the first query.
        if not messages:
            prompt = INSTRUCTION_TEMPLATE + self._get_templated_query(current_query)
            messages.append({"role": "user", "content": prompt})
        else:
            prompt = self._get_templated_query(current_query)
            messages.append({"role": "user", "content": prompt})

        #Conver the message template into prompt text.
        model_inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt")[0]
        prompt = self.tokenizer.decode(model_inputs)

        return {"prompt": prompt, "payload": sample}