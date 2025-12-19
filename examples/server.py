import sys
from pathlib import Path

# Add parent directory to path so we can import src module
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.server import OllamaServer

if __name__ == "__main__":
    server = OllamaServer() # return None
    print(server.get_process()) # return psutil.Process object