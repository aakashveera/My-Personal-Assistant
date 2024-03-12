import time
from typing import Any, Dict, List, Optional, Union

from langchain import chains
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.llms import HuggingFacePipeline

from .constants import INSTRUCTION_TEMPLATE, MODEL_NAME
from .utils import parse_chat_history
from .model import get_tokenizer

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

        prompt = self._get_inference_prompt(
            {
                "question": inputs["question"],
                "chat_history": inputs["chat_history"],
            }
        )

        start_time = time.time()
        response = self.hf_pipeline(prompt["prompt"])
        end_time = time.time()
        
        duration_milliseconds = (end_time - start_time) * 1000
        num_prompt_tokens = len(self.tokenizer(prompt["prompt"]))
        num_response_tokens = len(self.tokenizer(response))
        total_tokens = num_prompt_tokens + num_response_tokens

        if run_manager:
            run_manager.on_chain_end(
                outputs={
                    "answer": response,
                },
                
                metadata={
                    "prompt": prompt["prompt"],
                    "prompt_template_variables": prompt["payload"],
                    "usage.prompt_tokens": num_prompt_tokens,
                    "usage.total_tokens": total_tokens,
                    "usage.actual_new_tokens": num_response_tokens,
                    "duration_milliseconds": duration_milliseconds,
                },
            )

        return {"answer": response}
    
    
    def _get_templated_query(self, question):
        return f"""<<<\nQUESTION: {question} >>>.\n\n\n\n\n"""
    
    
    def _get_inference_prompt(self, sample: Dict[str, str]) -> Dict[str, Union[str, Dict]]:
        
        current_query = sample['question']
        chat_history = parse_chat_history(sample['chat_history'])       
                
        messages = []
        
        for index,(past_question, response_text) in enumerate(chat_history):
            
            if not index:
                prompt = INSTRUCTION_TEMPLATE + self._get_templated_query(past_question)
                messages.append({"role": "user", "content": prompt})
            else:
                prompt = self._get_templated_query(past_question)
                messages.append({"role": "user", "content": prompt})
                
            messages.append({"role": "assistant", "content": response_text})
        
        
        if not messages:
            prompt = INSTRUCTION_TEMPLATE + self._get_templated_query(current_query)
            messages.append({"role": "user", "content": prompt})
        else:
            prompt = self._get_templated_query(current_query)
            messages.append({"role": "user", "content": prompt})

                
        model_inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt")[0]
        prompt = self.tokenizer.decode(model_inputs)

        return {"prompt": prompt, "payload": sample}