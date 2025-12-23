import subprocess
import threading
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple, Union, List

from src.server import OllamaServer
from src.api_client import OllamaConnectionError, OllamaTimeoutError, OllamaAPIException
from src.resource_monitor import ResourceMonitor

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
            data = self.server.api_client.get_tags()
            models = data.get('models', [])
            for model in models:
                if model.get('name') == self.model_name or self.model_name in model.get('name', ''):
                    return True
            return False
        except (OllamaAPIException, OllamaConnectionError, OllamaTimeoutError):
            return False
        
        
    def stop_model(self) -> None:
        """Stop a running Ollama model via terminal command.

        Args:
            model_name: Name of the model to stop
        """
        try:
            subprocess.run(['ollama', 'stop', self.model_name], check=False, capture_output=True)
            print(f"Model '{self.model_name}' stopped.")
        except Exception as e:
            print(f"Error stopping model '{self.model_name}': {e}")
         
            
    def prompt_cli(self, prompt: str, return_output: bool = False) -> Optional[str]:
        """Runs an Ollama model via terminal and returns the output.

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
                
    def prompt_generate_api(self, prompt: str, num_ctx: Optional[int] = None, timeout: Optional[float] = None) -> str:
        """Send one chat message to Ollama using the /api/generate endpoint.

        Args:
            prompt: The prompt to send
            num_ctx: Context window size for the model (optional)
            timeout: Request timeout in seconds (optional)

        Returns:
            The model's response as a string
        """
        try:
            kwargs = {'model': self.model_name, 'prompt': prompt}
            if num_ctx is not None:
                kwargs['num_ctx'] = num_ctx
            if timeout is not None:
                kwargs['timeout'] = timeout
            data = self.server.api_client.generate(**kwargs)
            return data.get('response', '')
        except OllamaAPIException as e:
            raise Exception(f"Error sending prompt to model: {e}")
    
    def prompt_chat_api(self, messages: List[Dict[str, str]], num_ctx: Optional[int] = None, timeout: Optional[float] = None) -> str:
        """Send chat messages to Ollama using /api/chat endpoint.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            num_ctx: Context window size for the model (optional)
            timeout: Request timeout in seconds (optional)

        Returns:
            The assistant's response as a string
        """
        try:
            kwargs = {'model': self.model_name, 'messages': messages}
            if num_ctx is not None:
                kwargs['num_ctx'] = num_ctx
            if timeout is not None:
                kwargs['timeout'] = timeout
            data = self.server.api_client.chat(**kwargs)
            return data.get('message', {}).get('content', '')
        except OllamaAPIException as e:
            raise Exception(f"Error sending chat message to model: {e}")
        
    def send_prompt(self, prompt: Union[str, List[Dict[str, str]]], return_output: bool = False, num_ctx: Optional[int] = None, timeout: Optional[float] = None) -> Optional[str]:
        """Run a prompt on a model, using the API if model is running, otherwise use CLI.

        Args:
            prompt: Either a string prompt or a list of message dicts with 'role' and 'content' keys
            return_output: If True, return the response. If False, print it.
            num_ctx: Context window size for the model (optional)
            timeout: Request timeout in seconds (optional)

        Returns:
            The model's response if return_output=True, None otherwise

        Raises:
            RuntimeError: If Ollama server is not running
        """
        # Handle chat message history (list of dicts)
        if isinstance(prompt, list):
            # Chat API only works with running model
            if self.is_model_running():
                response = self.prompt_chat_api(prompt, num_ctx=num_ctx, timeout=timeout)
            else:
                raise RuntimeError("Chat API requires the model to be running. Please start the model first.")
        else:
            # Handle string prompt
            if self.is_model_running():
                response = self.prompt_generate_api(prompt, num_ctx=num_ctx, timeout=timeout)
            else:
                response = self.prompt_cli(prompt, return_output=True)

        if return_output:
            return response
        else:
            print(response)


class ModelInactivityMonitor:
    """Monitors inactivity and stops a model after n minutes of no interaction."""

    def __init__(self, model: OllamaModel, inactivity_minutes: float) -> None:
        self.model = model
        self.inactivity_minutes = inactivity_minutes
        self.last_interaction = datetime.now()
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None

    def record_interaction(self) -> None:
        """Update the last interaction time."""
        self.last_interaction = datetime.now()

    def start_monitoring(self) -> None:
        """Start monitoring for inactivity in a background thread."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_inactivity, daemon=True)
        self.monitor_thread.start()
        print(f"Inactivity monitor started for model '{self.model.model_name}' ({self.inactivity_minutes} minute(s)).")

    def stop_monitoring(self) -> None:
        """Stop monitoring for inactivity."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        print(f"Inactivity monitor stopped for model '{self.model.model_name}'.")

    def _monitor_inactivity(self) -> None:
        """Run in background thread to check for inactivity."""
        while self.monitoring:
            elapsed = datetime.now() - self.last_interaction
            inactivity_threshold = timedelta(minutes=self.inactivity_minutes)

            if elapsed >= inactivity_threshold:
                print(f"Model '{self.model.model_name}' inactive for {self.inactivity_minutes} minute(s). Stopping...")
                self.model.stop_model()
                self.monitoring = False
                break

            time.sleep(10)  # Check every 10 seconds


class ModelResourceMonitor:
    """Runs an OllamaModel with resource monitoring capabilities."""

    def __init__(self, model: OllamaModel) -> None:
        """Initialize with an OllamaModel instance.

        Args:
            model: An OllamaModel instance to run with monitoring
        """
        self.model = model
        self.resource_monitor = None
        self.monitor_thread = None

    def send_prompt_with_monitoring(self, prompt: str) -> Tuple[str, Dict[str, Optional[float]]]:
        """Runs a model with resource monitoring using OllamaModel.send_prompt().

        Args:
            prompt: The prompt to send to the model

        Returns:
            Tuple of (output string, resource statistics dictionary)
        """
        # Get the model's process to monitor
        process = self.model.server.get_process_id()

        if process is None:
            raise RuntimeError("Ollama server process not available")

        # Start monitoring thread
        self.resource_monitor = ResourceMonitor(process.pid, interval=0.5)
        self.monitor_thread = threading.Thread(target=self.resource_monitor.monitor)
        self.monitor_thread.start()

        try:
            # Run the model using OllamaModel method
            output = self.model.send_prompt(prompt, return_output=True)

            # Stop monitoring and collect stats
            self.resource_monitor.stop()
            self.monitor_thread.join()

            return output, self.resource_monitor.get_statistics()

        except Exception:
            self.resource_monitor.stop()
            self.monitor_thread.join()
            raise