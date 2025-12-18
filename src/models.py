import subprocess
import threading
import requests
from typing import List, Dict, Optional, Tuple, Any

from resource_monitor import ResourceMonitor
from server import test_ollama_server_running, get_ollama_process


# Model state management functions

def is_model_running(model_name: str, host: str = '127.0.0.1', port: int = 11434) -> bool:
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


def stop_model(model_name: str) -> None:
  """Stop a running Ollama model.

  Args:
    model_name: Name of the model to stop
  """
  try:
    subprocess.run(['ollama', 'stop', model_name], check=False, capture_output=True)
    print(f"Model '{model_name}' stopped.")
  except Exception as e:
    print(f"Error stopping model '{model_name}': {e}")


def run_ollama_cli(model_name: str, prompt: str, return_output: bool = False) -> Optional[str]:
  """Runs an Ollama model via CLI and returns the output.

  Args:
    model_name: Name of the model to run
    prompt: The prompt to send to the model
    return_output: If True, return the output. If False, print it.

  Returns:
    The model's output if return_output=True, None otherwise
  """
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


def send_prompt_to_running_model(model_name: str, prompt: str, host: str = '127.0.0.1', port: int = 11434) -> str:
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


def run_ollama_smart(model_name: str, prompt: str, return_output: bool = False) -> Optional[str]:
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
  # Check that the server is running
  test_ollama_server_running()

  # Check if model is already running
  if is_model_running(model_name):
    print(f"Model '{model_name}' is already running. Sending prompt via API...")
    response = send_prompt_to_running_model(model_name, prompt)
  else:
    print(f"Model '{model_name}' not running. Starting via CLI...")
    response = run_ollama_cli(model_name, prompt, return_output=True)

  if return_output:
    return response
  else:
    print(response)


def send_chat_message(model_name: str, messages: List[Dict[str, str]], host: str = '127.0.0.1', port: int = 11434, stream: bool = False) -> str:
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


def run_ollama_with_monitoring(model_name: str, prompt: str) -> Tuple[str, Dict[str, Optional[float]]]:
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


# Model installation functions
def get_installed_models() -> List[str]:
    """Get the list of locally installed Ollama models.

    Returns:
        A list of model names installed locally.

    Raises:
        subprocess.CalledProcessError: If the ollama list command fails.
        FileNotFoundError: If ollama is not installed or not in PATH.
    """
    try:
        result = subprocess.run(['ollama', 'list'],
                              capture_output=True, text=True, check=True)

        lines = result.stdout.strip().split('\n')
        models = []

        # Skip the header line and parse model names
        for line in lines[1:]:
            if line.strip():
                # The first column is the model name
                model_name = line.split()[0]
                models.append(model_name)

        return models

    except FileNotFoundError:
        raise FileNotFoundError("Ollama is not installed or not in PATH")
    except subprocess.CalledProcessError as e:
        raise subprocess.CalledProcessError(
            e.returncode,
            e.cmd,
            stderr=f"Error running 'ollama list': {e.stderr}"
        )


def is_model_installed(model_name: str) -> bool:
  """
  Checks if the Ollama model is installed locally.
  """
  try:
    print(f"Checking if model '{model_name}' is available...")
    installed_models = get_installed_models()

    if model_name in installed_models:
      print(f"Model '{model_name}' is already installed.")
      return True
    else:
      print(f"Model '{model_name}' not found.")
      return False

  except FileNotFoundError:
    print("Ollama not found.  Please ensure Ollama is installed and in your PATH.")
    return False
  except subprocess.CalledProcessError:
    print("Failed to get list of installed models.")
    return False


def install_model(model_name: str) -> bool:
  """
  Installs the Ollama model by pulling it.
  """
  try:
    print(f"Pulling model '{model_name}'...")
    pull_command = ["ollama", "pull", model_name]
    subprocess.run(pull_command, capture_output=True, text=True, check=True)
    print(f"Model '{model_name}' successfully pulled.")
    return True

  except subprocess.CalledProcessError:
    print(f"Failed to pull model '{model_name}'.")
    return False

def check_and_install_model(model_name: str) -> bool:
  """
  Checks if the Ollama model is installed locally and pulls it if not.
  """
  # Ensure the Ollama server is running before attempting to pull models
  test_ollama_server_running()

  if is_model_installed(model_name):
    return True
  else:
    return install_model(model_name)


# Model grouping and filtering functions

def get_models_grouped_by_base_name() -> Dict[str, List[str]]:
    """Get locally installed Ollama models grouped by their base name.

    Models are grouped by removing the tag (the part after the colon).
    For example:
        - llama3.2:1b and llama3.2:7b are grouped as llama3.2
        - qwen2.5:3b is grouped as qwen2.5

    Returns:
        A dictionary where keys are base model names and values are lists
        of full model names (with tags) for that base model.

    Raises:
        subprocess.CalledProcessError: If the ollama list command fails.
        FileNotFoundError: If ollama is not installed or not in PATH.
    """
    models = get_installed_models()
    grouped = {}

    for model in models:
        # Extract base name (part before the colon)
        base_name = model.split(':')[0]

        if base_name not in grouped:
            grouped[base_name] = []

        grouped[base_name].append(model)

    return grouped


def get_models_grouped_by_size() -> Dict[float, List[str]]:
    """Get locally installed Ollama models grouped by their parameter size.

    Models are grouped by extracting the size tag (the part after the colon)
    and converting it to billions. For example:
        - llama3.2:7b and qwen2.5:7b are grouped as 7.0 (7 billion)
        - smollm:135m is grouped as 0.135 (135 million)
        - llama3.2:1b is grouped as 1.0 (1 billion)

    Returns:
        A dictionary where keys are numeric sizes in billions and values are lists
        of full model names (with tags) for that size.

    Raises:
        subprocess.CalledProcessError: If the ollama list command fails.
        FileNotFoundError: If ollama is not installed or not in PATH.
    """
    models = get_installed_models()
    grouped = {}

    for model in models:
        # Extract size tag (part after the colon)
        parts = model.split(':')
        size_str = parts[1] if len(parts) > 1 else 'unknown'

        # Convert size to billions
        if size_str != 'unknown':
            size_value = float(size_str[:-1])  # Remove the letter and convert to float
            unit = size_str[-1].lower()  # Get the last character (b or m)

            if unit == 'b':
                # Already in billions
                size_numeric = size_value
            elif unit == 'm':
                # Convert millions to billions
                size_numeric = size_value / 1000
            else:
                # Unknown unit, skip
                continue
        else:
            size_numeric = 0.0

        if size_numeric not in grouped:
            grouped[size_numeric] = []

        grouped[size_numeric].append(model)

    return grouped


def get_models_in_size_range(min_size: float, max_size: float) -> List[str]:
    """Get models within a specific parameter size range (in billions).

    Args:
        min_size: Minimum parameter size in billions (e.g., 0.1 for 100M)
        max_size: Maximum parameter size in billions (e.g., 7.0 for 7B)

    Returns:
        A list of model names that fall within the specified size range.

    Raises:
        subprocess.CalledProcessError: If the ollama list command fails.
        FileNotFoundError: If ollama is not installed or not in PATH.
        ValueError: If min_size is greater than max_size.
    """
    if min_size > max_size:
        raise ValueError(f"min_size ({min_size}) cannot be greater than max_size ({max_size})")

    models_by_size = get_models_grouped_by_size()
    filtered_models = []

    for size, models in models_by_size.items():
        if min_size <= size <= max_size:
            filtered_models.extend(models)

    return filtered_models


def get_coding_models() -> List[str]:
    """Get models designed for coding tasks.

    Coding models are identified by containing the words "code", "coding", or "coder"
    in their model name (case-insensitive).

    Returns:
        A list of model names that are designed for coding tasks.

    Raises:
        subprocess.CalledProcessError: If the ollama list command fails.
        FileNotFoundError: If ollama is not installed or not in PATH.
    """
    models = get_installed_models()
    coding_keywords = {'code', 'coding', 'coder'}
    coding_models = []

    for model in models:
        model_lower = model.lower()
        if any(keyword in model_lower for keyword in coding_keywords):
            coding_models.append(model)

    return coding_models


def get_non_coding_models() -> List[str]:
    """Get models that are not designed for coding tasks.

    Returns:
        A list of model names that are not designed for coding tasks.

    Raises:
        subprocess.CalledProcessError: If the ollama list command fails.
        FileNotFoundError: If ollama is not installed or not in PATH.
    """
    all_models = get_installed_models()
    coding_models = get_coding_models()
    coding_models_set = set(coding_models)
    non_coding_modules = [model for model in all_models if model not in coding_models_set]

    return non_coding_modules
