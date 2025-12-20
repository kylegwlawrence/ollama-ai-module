import sys
from pathlib import Path

# Add parent directory to path so we can import src module
sys.path.insert(0, str(Path(__file__).parent.parent))

from ollama_chat import continue_chat_session

if __name__ == "__main__":
    prompt = "Then why isn't it purple?"
    session_name = "grow-huge-state"
    
    resp = continue_chat_session(prompt, session_name)
    
    print(resp)