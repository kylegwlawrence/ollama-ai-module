import subprocess
import threading
import time
import psutil
from typing import Tuple, Dict


class ResourceMonitor:
  """Monitors CPU and memory usage of a process in a background thread."""

  def __init__(self, pid, interval=0.5):
    self.pid = pid
    self.interval = interval
    self.running = False
    self.cpu_samples = []
    self.memory_samples = []

  def monitor(self):
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

  def stop(self):
    """Stop monitoring."""
    self.running = False

  def get_statistics(self):
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


def run_ollama(model_name, prompt, return_output=False):
  """Runs an Ollama model and returns the output."""
  try:
    result = subprocess.run(['ollama', 'run', model_name],
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


def run_ollama_with_monitoring(model_name, prompt) -> Tuple[str, Dict]:
  """Runs an Ollama model with resource monitoring.

  Args:
    model_name: Name of the Ollama model to run
    prompt: The prompt to send to the model

  Returns:
    Tuple of (output string, resource statistics dictionary)
  """
  try:
    # Launch process
    process = subprocess.Popen(
      ['ollama', 'run', model_name],
      stdin=subprocess.PIPE,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      text=True
    )

    # Start monitoring thread
    resource_monitor = ResourceMonitor(process.pid, interval=0.5)
    monitor_thread = threading.Thread(target=resource_monitor.monitor)
    monitor_thread.start()

    # Send input and wait for completion
    stdout, stderr = process.communicate(input=prompt)

    # Stop monitoring and collect stats
    resource_monitor.stop()
    monitor_thread.join()

    if process.returncode != 0:
      raise subprocess.CalledProcessError(process.returncode, process.args, stdout, stderr)

    return stdout, resource_monitor.get_statistics()

  except subprocess.CalledProcessError as e:
    raise
  except Exception as e:
    raise