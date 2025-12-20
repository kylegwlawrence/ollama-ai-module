import sys
from pathlib import Path

# Add parent directory to path so we can import src module
sys.path.insert(0, str(Path(__file__).parent.parent))

from ollama_chat import start_chat_session

if __name__ == "__main__":
    prompt = "Why is the sky not green?"
    model_name = "smollm2:135m"
    system_prompt = "You are a scientist that studies the atmosphere"
    
    resp = start_chat_session(prompt, model_name, system_prompt)
    
    print(resp)