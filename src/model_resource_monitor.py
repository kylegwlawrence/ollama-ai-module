import threading
from typing import Dict, Optional, Tuple

from src.resource_monitor import ResourceMonitor
from src.model import OllamaModel


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
        process = self.model.server.get_process()

        if process is None:
            raise RuntimeError("Ollama server process not available")

        # Start monitoring thread
        self.resource_monitor = ResourceMonitor(process.pid, interval=0.5)
        self.monitor_thread = threading.Thread(target=self.resource_monitor.monitor)
        self.monitor_thread.start()

        try:
            # Run the model using OllamaModel's smart routing
            output = self.model.send_prompt(prompt, return_output=True)

            # Stop monitoring and collect stats
            self.resource_monitor.stop()
            self.monitor_thread.join()

            return output, self.resource_monitor.get_statistics()

        except Exception:
            self.resource_monitor.stop()
            self.monitor_thread.join()
            raise
