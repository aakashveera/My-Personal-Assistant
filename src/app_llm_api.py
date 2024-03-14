import warnings
warnings.filterwarnings("ignore")

import gradio as gr
from typing import List

from .utils import create_logger
from .mistral_client import MistralAPIClient

logger = create_logger("logs/outputs.log")

client = MistralAPIClient()

# === Gradio Interface ===


def predict(message: str, history: List[List[str]], about_me: str):
    """
    Predicts a response to a given message using the financial_bot Gradio UI.

    Args:
        message (str): The message to generate a response for.
        history (List[List[str]]): A list of previous conversations.
        
    Returns:
        str: The generated response.
    """
        
    for text in client.stream_answer(message, history):
        yield text


demo = gr.ChatInterface(
    predict,
    textbox=gr.Textbox(
        placeholder="Ask me a any question",
        label="question",
        container=False,
        scale=5,
    ),
    title="Diya - The Sassy AI Assistant",
    description="Ask me any question about myself of about my boss Aakash, and I will do my best to answer them.",
    theme="soft",
    examples=[
        [
            "Hi There! What is your name?"
        ],
        [
            "Where does Aakash Currently Works?"
        ],
        [
            "What is the favourite food of Aakash?"
        ],
        [
            "What does Aakash loves to do in his free time?"
        ]
    ],
    cache_examples=False,
    retry_btn=None,
    undo_btn=None,
    clear_btn="Clear",
)


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=True)