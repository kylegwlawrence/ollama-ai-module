import human_id
import time
import psutil
from typing import List, Dict, Optional


class ResourceMonitor:
  """Monitors CPU and memory usage of a process in a background thread."""

  def __init__(self, pid: int, interval: float = 0.5) -> None:
    self.pid = pid
    self.interval = interval
    self.running = False
    self.cpu_samples: List[float] = []
    self.memory_samples: List[float] = []

  def monitor(self) -> None:
    """Run in background thread to collect samples."""
    try:
      process = psutil.Process(self.pid)
      self.running = True

      while self.running:
        cpu_percent = process.cpu_percent(interval=None)
        memory_mb = process.memory_info().rss / (1024 * 1024)

        self.cpu_samples.append(cpu_percent)
        self.memory_samples.append(memory_mb)

        time.sleep(self.interval)

    except psutil.NoSuchProcess:
      pass  # Process ended normally
    except Exception:
      pass  # Log but don't crash

  def stop(self) -> None:
    """Stop monitoring."""
    self.running = False

  def get_statistics(self) -> Dict[str, Optional[float]]:
    """Calculate average and peak values."""
    if not self.cpu_samples:
      return {
        'cpu_avg_percent': None,
        'cpu_peak_percent': None,
        'memory_avg_mb': None,
        'memory_peak_mb': None
      }

    return {
      'cpu_avg_percent': round(sum(self.cpu_samples) / len(self.cpu_samples), 2),
      'cpu_peak_percent': round(max(self.cpu_samples), 2),
      'memory_avg_mb': round(sum(self.memory_samples) / len(self.memory_samples), 2),
      'memory_peak_mb': round(max(self.memory_samples), 2)
    }


def generate_human_id(num_words: int = 3) -> str:
    """Generate a human-readable ID using random words.

    Args:
        num_words: Number of words to use in the ID (default: 3).

    Returns:
        A string containing a human-readable ID (e.g., "happy-blue-penguin").

    Raises:
        ValueError: If num_words is less than 3.
    """

    if num_words < 3:
        raise ValueError(f"num_words must be at least 3, got {num_words}")

    return human_id.generate_id(word_count=num_words)