"""
Simple Gradio UI for AtlasCare
"""

import os
import requests
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

PORT = os.getenv("PORT", "8000")
API_HOST = "127.0.0.1"

API_URL = f"http://{API_HOST}:{PORT}/query"

GRADIO_HOST = os.getenv("GRADIO_HOST", "127.0.0.1")
GRADIO_PORT = int(os.getenv("GRADIO_PORT", "7860"))

DEFAULT_SESSION_ID = os.getenv(
    "DEFAULT_SESSION_ID",
    "sess_001"
)

API_URL = f"http://{API_HOST}:{PORT}/query"


def chat(message, history, session_id):
    if not message.strip():
        return history, ""

    payload = {
        "message": message,
        "session_id": session_id,
    }

    try:
        response = requests.post(
            API_URL,
            json=payload,
            timeout=30
        )
        response.raise_for_status()

        data = response.json()
        bot_reply = data["response"]

    except Exception as exc:
        bot_reply = f"Error: {str(exc)}"

    history.append((message, bot_reply))
    return history, ""


with gr.Blocks(title="AtlasCare") as demo:
    gr.Markdown("# AtlasCare Customer Support")

    session_id = gr.Textbox(
        label="Session ID",
        value=DEFAULT_SESSION_ID
    )

    chatbot = gr.Chatbot(height=450)

    message = gr.Textbox(
        placeholder="Type your query here..."
    )

    send = gr.Button("Send")

    send.click(
        fn=chat,
        inputs=[message, chatbot, session_id],
        outputs=[chatbot, message]
    )

    message.submit(
        fn=chat,
        inputs=[message, chatbot, session_id],
        outputs=[chatbot, message]
    )


if __name__ == "__main__":
    demo.launch(
        server_name=GRADIO_HOST,
        server_port=GRADIO_PORT
    )