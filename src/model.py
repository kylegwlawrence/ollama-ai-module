import subprocess
import requests
from typing import Optional

from src.server import OllamaServer

class OllamaModel:
    """Manages interactions with the AI model."""

    def __init__(self, model_name: str, OllamaServer: OllamaServer) -> None:
        """Initialize OllamaModel with the model name and an OllamaServer object

        Args:
        host: Ollama server host (default: localhost)
        port: Ollama server port (default: 11434)
        """
        self.model_name = model_name
        self.server = OllamaServer

    def is_model_running(self) -> bool:
        """Check if a specific model is currently running.

        Args:
            model_name: Name of the model to check
            host: Ollama server host (default: localhost)
            port: Ollama server port (default: 11434)

        Returns:
            True if the model is running, False otherwise
        """
        try:
            url = f'http://{self.server.host}:{self.server.port}/api/tags'
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                data = response.json()
                models = data.get('models', [])
                for model in models:
                    if model.get('name') == self.model_name or self.model_name in model.get('name', ''):
                        print(f"...{self.model_name} is already running...")
                        return True
            return False
        except Exception:
            return False
        
        
    def stop_model(self) -> None:
        """Stop a running Ollama model.

        Args:
            model_name: Name of the model to stop
        """
        try:
            subprocess.run(['ollama', 'stop', self.model_name], check=False, capture_output=True)
            print(f"Model '{self.model_name}' stopped.")
        except Exception as e:
            print(f"Error stopping model '{self.model_name}': {e}")
         
            
    def prompt_cli(self, prompt: str, return_output: bool = False) -> Optional[str]:
        """Runs an Ollama model via CLI and returns the output.

        Args:
            model_name: Name of the model to run
            prompt: The prompt to send to the model
            return_output: If True, return the output. If False, print it.

        Returns:
            The model's output if return_output=True, None otherwise
        """
        try:
            print("Running model via CLI...")
            result = subprocess.run(['ollama', 'run', self.model_name],
            input=prompt, capture_output=True, text=True, check=True)
            if return_output:
                return result.stdout
            else:
                print(result.stdout)
        except subprocess.CalledProcessError as e:
            if return_output:
                raise
            else:
                print(f"Error running Ollama: {e}")
                
    def prompt_http(self, prompt: str) -> str:
        """Send a prompt to an already running Ollama model via HTTP API.

        Args:
            model_name: Name of the model
            prompt: The prompt to send
            host: Ollama server host (default: localhost)
            port: Ollama server port (default: 11434)

        Returns:
            The model's response as a string
        """
        try:
            print("Running model via API...")
            url = f'http://{self.server.host}:{self.server.port}/api/generate'
            payload = {
            'model': self.model_name,
            'prompt': prompt,
            'stream': False
            }
            response = requests.post(url, json=payload, timeout=300)
            if response.status_code == 200:
                data = response.json()
                return data.get('response', '')
            else:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
        except Exception as e:
            raise Exception(f"Error sending prompt to model: {e}")
        
    def send_prompt(self, prompt: str, return_output: bool = False) -> Optional[str]:
        """Run a prompt on a model, using the API if model is running, otherwise use CLI.

        Args:
            model_name: Name of the model
            prompt: The prompt to send
            return_output: If True, return the response. If False, print it.

        Returns:
            The model's response if return_output=True, None otherwise

        Raises:
            RuntimeError: If Ollama server is not running
        """
        # Check if model is already running
        if self.is_model_running():
            response = self.prompt_http(prompt)
        else:
            response = self.prompt_cli(prompt, return_output=True)

        if return_output:
            return response
        else:
            print(response)

