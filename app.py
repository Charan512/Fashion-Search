"""
Hugging Face Spaces Entry Point.

Hugging Face Spaces looks for an `app.py` at the root of the repository.
This file simply imports the Gradio app we built in `demo/gradio_app.py`
and launches it.
"""
from demo.gradio_app import build_app
import os

# We don't strictly need dotenv here since HF Spaces provides secrets via environment variables,
# but it's safe to try loading for local testing if someone runs `python app.py`.
try:
    import dotenv
    dotenv.load_dotenv()
except ImportError:
    pass

app = build_app()

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
