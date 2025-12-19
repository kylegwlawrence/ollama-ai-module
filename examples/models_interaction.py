import sys
from pathlib import Path

# Add parent directory to path so we can import src module
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import OllamaModel
from src.server import OllamaServer

if __name__ == "__main__":
    model_name = "smollm2:135m"
    prompt = "What is 4 divided by 2?"

    server = OllamaServer()
    model = OllamaModel(model_name, server)
    model.run_prompt(prompt)