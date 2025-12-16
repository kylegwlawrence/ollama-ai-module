import subprocess
import threading
import time
import psutil
import requests
from typing import Tuple, Dict
from datetime import datetime, timedelta


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


def get_ollama_process():
  """Get the Ollama server process object.

  Returns:
    psutil.Process object for the Ollama server, or None if not found
  """
  try:
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
      try:
        if 'ollama' in proc.name().lower() or (proc.cmdline() and 'ollama' in ' '.join(proc.cmdline()).lower()):
          # Filter for the main serve process, not child processes
          if 'serve' in ' '.join(proc.cmdline()).lower():
            return proc
      except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        pass
    return None
  except Exception:
    return None


def get_ollama_server_metrics():
  """Get real-time metrics for the running Ollama server.

  Returns:
    Dictionary with CPU usage, memory usage, and timestamp
  """
  try:
    proc = get_ollama_process()
    if not proc:
      return None

    cpu_percent = proc.cpu_percent(interval=None)
    memory_info = proc.memory_info()
    memory_mb = memory_info.rss / (1024 * 1024)

    return {
      'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
      'cpu_percent': round(cpu_percent, 2),
      'memory_mb': round(memory_mb, 2),
      'memory_percent': round(proc.memory_percent(), 2)
    }
  except Exception:
    return None


def is_ollama_server_running(host='127.0.0.1', port=11434, timeout=2):
  """Check if Ollama server is running by attempting to connect.

  Args:
    host: Ollama server host (default: localhost)
    port: Ollama server port (default: 11434)
    timeout: Connection timeout in seconds

  Returns:
    True if Ollama is running, False otherwise
  """
  try:
    result = subprocess.run(
      ['curl', '-s', '-o', '/dev/null', '-w', '%{http_code}', f'http://{host}:{port}/api/tags'],
      capture_output=True,
      text=True,
      timeout=timeout
    )
    return result.stdout.strip() == '200'
  except Exception:
    return False


def start_ollama_server(detach=True):
  """Start Ollama server in a background process.

  Args:
    detach: If True, start in background. If False, return the process object.

  Returns:
    subprocess.Popen object if detach=False, None otherwise
  """
  try:
    if detach:
      subprocess.Popen(
        ['ollama', 'serve'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True
      )
      time.sleep(2)  # Give server time to start
    else:
      return subprocess.Popen(
        ['ollama', 'serve'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
      )
  except Exception as e:
    print(f"Error starting Ollama server: {e}")
    return None


def ensure_ollama_server_running(host='127.0.0.1', port=11434):
  """Check if Ollama is running, and start it if not.

  Args:
    host: Ollama server host
    port: Ollama server port
  """
  if not is_ollama_server_running(host, port):
    print("Ollama server not running. Starting it...")
    start_ollama_server(detach=True)
    print("Ollama server started.")
  else:
    print("Ollama server is already running.")


def stop_ollama_server():
  """Stop the Ollama server gracefully."""
  try:
    subprocess.run(['ollama', 'kill'], check=False, capture_output=True)
    print("Ollama server stopped.")
  except Exception as e:
    print(f"Error stopping Ollama server: {e}")


def stop_ollama_server_after(minutes):
  """Stop the Ollama server after a specified number of minutes.

  Runs in a background thread, so it doesn't block your code.

  Args:
    minutes: Number of minutes to wait before stopping the server
  """
  def delayed_stop():
    time.sleep(minutes * 60)
    stop_ollama_server()

  stop_thread = threading.Thread(target=delayed_stop, daemon=True)
  stop_thread.start()
  print(f"Ollama server will stop in {minutes} minute(s).")


def stop_ollama_model(model_name):
  """Stop a running Ollama model.

  Args:
    model_name: Name of the model to stop
  """
  try:
    subprocess.run(['ollama', 'stop', model_name], check=False, capture_output=True)
    print(f"Model '{model_name}' stopped.")
  except Exception as e:
    print(f"Error stopping model '{model_name}': {e}")


def stop_ollama_model_after(model_name, minutes):
  """Stop an Ollama model after a specified number of minutes.

  Runs in a background thread, so it doesn't block your code.

  Args:
    model_name: Name of the model to stop
    minutes: Number of minutes to wait before stopping the model
  """
  def delayed_stop():
    time.sleep(minutes * 60)
    stop_ollama_model(model_name)

  stop_thread = threading.Thread(target=delayed_stop, daemon=True)
  stop_thread.start()
  print(f"Model '{model_name}' will stop in {minutes} minute(s).")


def is_model_running(model_name, host='127.0.0.1', port=11434):
  """Check if a specific model is currently running.

  Args:
    model_name: Name of the model to check
    host: Ollama server host (default: localhost)
    port: Ollama server port (default: 11434)

  Returns:
    True if the model is running, False otherwise
  """
  try:
    url = f'http://{host}:{port}/api/tags'
    response = requests.get(url, timeout=2)
    if response.status_code == 200:
      data = response.json()
      models = data.get('models', [])
      for model in models:
        if model.get('name') == model_name or model_name in model.get('name', ''):
          return True
    return False
  except Exception:
    return False


def send_prompt_to_running_model(model_name, prompt, host='127.0.0.1', port=11434):
  """Send a prompt to a running Ollama model via HTTP API.

  Args:
    model_name: Name of the model
    prompt: The prompt to send
    host: Ollama server host (default: localhost)
    port: Ollama server port (default: 11434)

  Returns:
    The model's response as a string
  """
  try:
    url = f'http://{host}:{port}/api/generate'
    payload = {
      'model': model_name,
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


def run_ollama_smart(model_name, prompt, return_output=False):
  """Run a prompt on a model, using the API if model is running, otherwise use CLI.

  Args:
    model_name: Name of the model
    prompt: The prompt to send
    return_output: If True, return the response. If False, print it.

  Returns:
    The model's response if return_output=True, None otherwise
  """
  try:
    # First ensure the server is running
    ensure_ollama_server_running()

    # Check if model is already running
    if is_model_running(model_name):
      print(f"Model '{model_name}' is already running. Sending prompt via API...")
      response = send_prompt_to_running_model(model_name, prompt)
    else:
      print(f"Model '{model_name}' not running. Starting via CLI...")
      response = run_ollama(model_name, prompt, return_output=True)

    if return_output:
      return response
    else:
      print(response)
  except Exception as e:
    if return_output:
      raise
    else:
      print(f"Error: {e}")


class InactivityMonitor:
  """Monitors inactivity and stops a model after n minutes of no interaction."""

  def __init__(self, model_name, inactivity_minutes):
    self.model_name = model_name
    self.inactivity_minutes = inactivity_minutes
    self.last_interaction = datetime.now()
    self.monitoring = False
    self.monitor_thread = None

  def record_interaction(self):
    """Update the last interaction time."""
    self.last_interaction = datetime.now()

  def start_monitoring(self):
    """Start monitoring for inactivity in a background thread."""
    self.monitoring = True
    self.monitor_thread = threading.Thread(target=self._monitor_inactivity, daemon=True)
    self.monitor_thread.start()
    print(f"Inactivity monitor started for model '{self.model_name}' ({self.inactivity_minutes} minute(s)).")

  def stop_monitoring(self):
    """Stop monitoring for inactivity."""
    self.monitoring = False
    if self.monitor_thread:
      self.monitor_thread.join(timeout=1)
    print(f"Inactivity monitor stopped for model '{self.model_name}'.")

  def _monitor_inactivity(self):
    """Run in background thread to check for inactivity."""
    while self.monitoring:
      elapsed = datetime.now() - self.last_interaction
      inactivity_threshold = timedelta(minutes=self.inactivity_minutes)

      if elapsed >= inactivity_threshold:
        print(f"Model '{self.model_name}' inactive for {self.inactivity_minutes} minute(s). Stopping...")
        stop_ollama_model(self.model_name)
        self.monitoring = False
        break

      time.sleep(10)  # Check every 10 seconds


class ChatSession:
  """Manages a conversation session with an Ollama model."""

  def __init__(self, model_name, host='127.0.0.1', port=11434):
    """Initialize chat session.

    Args:
      model_name: Name of the model to chat with
      host: Ollama server host (default: localhost)
      port: Ollama server port (default: 11434)
    """
    self.model_name = model_name
    self.host = host
    self.port = port
    self.messages = []

  def send_message(self, user_message):
    """Send a message and get response.

    Args:
      user_message: User's message text

    Returns:
      Assistant's response text
    """
    # Add user message to history
    self.messages.append({'role': 'user', 'content': user_message})

    # Call send_chat_message() with full history
    response = send_chat_message(self.model_name, self.messages, self.host, self.port)

    # Add assistant response to history
    self.messages.append({'role': 'assistant', 'content': response})

    return response

  def reset(self):
    """Clear conversation history."""
    self.messages = []

  def get_history(self):
    """Get conversation history.

    Returns:
      List of message dictionaries
    """
    return self.messages.copy()

  def set_system_prompt(self, system_prompt):
    """Set or update system prompt.

    Args:
      system_prompt: System instruction text
    """
    # Remove existing system message if present
    self.messages = [m for m in self.messages if m.get('role') != 'system']

    # Add new system message at the beginning
    self.messages.insert(0, {'role': 'system', 'content': system_prompt})


def send_chat_message(model_name, messages, host='127.0.0.1', port=11434, stream=False):
  """Send a chat message to Ollama using /api/chat endpoint.

  Args:
    model_name: Name of the model
    messages: List of message dicts with 'role' and 'content' keys
              Example: [{'role': 'user', 'content': 'Hello'}]
    host: Ollama server host (default: localhost)
    port: Ollama server port (default: 11434)
    stream: Enable streaming responses (default: False)

  Returns:
    The assistant's response as a string
  """
  try:
    url = f'http://{host}:{port}/api/chat'
    payload = {
      'model': model_name,
      'messages': messages,
      'stream': stream
    }
    response = requests.post(url, json=payload, timeout=300)
    if response.status_code == 200:
      data = response.json()
      return data.get('message', {}).get('content', '')
    else:
      raise Exception(f"HTTP {response.status_code}: {response.text}")
  except Exception as e:
    raise Exception(f"Error sending chat message to model: {e}")


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


def run_ollama_chat_smart(chat_session, user_message):
  """Send chat message using API if server is running.

  Args:
    chat_session: ChatSession instance
    user_message: User's message text

  Returns:
    Assistant's response text
  """
  try:
    # Ensure the server is running
    ensure_ollama_server_running()

    # Send message via chat session (which maintains context)
    response = chat_session.send_message(user_message)

    return response
  except Exception as e:
    print(f"Error in chat: {e}")
    raise


def run_chat_with_monitoring(chat_session, user_message) -> Tuple[str, Dict]:
  """Send chat message with resource monitoring.

  Args:
    chat_session: ChatSession instance
    user_message: User's message text

  Returns:
    Tuple of (response text, resource statistics dict)
  """
  try:
    # Ensure the server is running
    ensure_ollama_server_running()

    # Get Ollama server process for monitoring
    proc = get_ollama_process()
    if not proc:
      # If can't find process, just send message without monitoring
      response = chat_session.send_message(user_message)
      return response, None

    # Start monitoring thread
    resource_monitor = ResourceMonitor(proc.pid, interval=0.5)
    monitor_thread = threading.Thread(target=resource_monitor.monitor)
    monitor_thread.start()

    # Send message and get response
    response = chat_session.send_message(user_message)

    # Stop monitoring and collect stats
    resource_monitor.stop()
    monitor_thread.join()

    stats = resource_monitor.get_statistics()
    return response, stats

  except Exception as e:
    print(f"Error in monitored chat: {e}")
    raise