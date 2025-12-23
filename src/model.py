import subprocess
import threading
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple, List

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
         
            
    def prompt_generate_api(self, prompt: str, num_ctx: Optional[int] = None, num_predict: Optional[int] = None, suffix: Optional[str] = None, timeout: Optional[float] = None) -> str:
        """Send one chat message to Ollama using the /api/generate endpoint.

        Args:
            prompt: The prompt to send
            num_ctx: Context window size for the model (optional)
            num_predict: Maximum tokens allowed in the response (optional)
            suffix: Text to append after the generated response (optional)
            timeout: Request timeout in seconds (optional)

        Returns:
            The model's response as a string
        """
        try:
            data = self.server.api_client.generate(
                model=self.model_name,
                prompt=prompt,
                num_ctx=num_ctx,
                num_predict=num_predict,
                suffix=suffix,
                timeout=timeout
            )
            return data.get('response', '')
        except OllamaAPIException as e:
            raise Exception(f"Error sending prompt to model: {e}")
    
    def prompt_chat_api(self, messages: List[Dict[str, str]], num_ctx: Optional[int] = None, num_predict: Optional[int] = None, suffix: Optional[str] = None, timeout: Optional[float] = None) -> str:
        """Send chat messages to Ollama using /api/chat endpoint.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            num_ctx: Context window size for the model (optional)
            num_predict: Maximum tokens allowed in the response (optional)
            suffix: Text to append after the generated response (optional)
            timeout: Request timeout in seconds (optional)

        Returns:
            The assistant's response as a string
        """
        try:
            data = self.server.api_client.chat(
                model=self.model_name,
                messages=messages,
                num_ctx=num_ctx,
                num_predict=num_predict,
                suffix=suffix,
                timeout=timeout
            )
            return data.get('message', {}).get('content', '')
        except OllamaAPIException as e:
            raise Exception(f"Error sending chat message to model: {e}")


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