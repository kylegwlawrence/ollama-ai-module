import threading
import time
from datetime import datetime, timedelta
from typing import Optional

from models import stop_model


class ModelInactivityMonitor:
  """Monitors inactivity and stops a model after n minutes of no interaction."""

  def __init__(self, model_name: str, inactivity_minutes: float) -> None:
    self.model_name = model_name
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
    print(f"Inactivity monitor started for model '{self.model_name}' ({self.inactivity_minutes} minute(s)).")

  def stop_monitoring(self) -> None:
    """Stop monitoring for inactivity."""
    self.monitoring = False
    if self.monitor_thread:
      self.monitor_thread.join(timeout=1)
    print(f"Inactivity monitor stopped for model '{self.model_name}'.")

  def _monitor_inactivity(self) -> None:
    """Run in background thread to check for inactivity."""
    while self.monitoring:
      elapsed = datetime.now() - self.last_interaction
      inactivity_threshold = timedelta(minutes=self.inactivity_minutes)

      if elapsed >= inactivity_threshold:
        print(f"Model '{self.model_name}' inactive for {self.inactivity_minutes} minute(s). Stopping...")
        stop_model(self.model_name)
        self.monitoring = False
        break

      time.sleep(10)  # Check every 10 seconds
