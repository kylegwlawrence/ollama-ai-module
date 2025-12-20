import subprocess
import psutil
from typing import Optional, List, Dict


class OllamaServer:
  """Manages interactions with Ollama server."""

  def __init__(self, host: str = '127.0.0.1', port: int = 11434):
    """Initialize OllamaServer with connection parameters.

    Args:
      host: Ollama server host (default: localhost)
      port: Ollama server port (default: 11434)
    """
    self.host = host
    self.port = port
    self.test_running()

  def test_running(self, timeout: int = 2) -> None:
    """Check if Ollama server is running, raise error if not.

    Args:
      timeout: Connection timeout in seconds

    Raises:
      RuntimeError: If Ollama server is not running
    """
    try:
      result = subprocess.run(
        ['curl', '-s', '-o', '/dev/null', '-w', '%{http_code}', f'http://{self.host}:{self.port}/api/tags'],
        capture_output=True,
        text=True,
        timeout=timeout
      )
      if result.stdout.strip() != '200':
        raise RuntimeError(
          f"Non-200 code. Ollama server is not running on {self.host}:{self.port}. "
          "Please start the Ollama server manually before running this function."
        )
      else:
          print("...Ollama server is running...")
    except subprocess.SubprocessError:
      raise RuntimeError(
        f"Ollama server is not running on {self.host}:{self.port}. "
        "Please start the Ollama server manually before running this function."
      )

  @staticmethod
  def get_process() -> Optional[psutil.Process]:
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

  def get_installed_models(self) -> List[str]:
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

  def get_models_grouped_by_base_name(self) -> Dict[str, List[str]]:
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
    models = self.get_installed_models()
    grouped = {}

    for model in models:
        # Extract base name (part before the colon)
        base_name = model.split(':')[0]

        if base_name not in grouped:
            grouped[base_name] = []

        grouped[base_name].append(model)

    return grouped


def get_models_grouped_by_size(self) -> Dict[float, List[str]]:
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
    models = self.get_installed_models()
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


def get_models_in_size_range(self, min_size: float, max_size: float) -> List[str]:
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

    models_by_size = self.get_models_grouped_by_size()
    filtered_models = []

    for size, models in models_by_size.items():
        if min_size <= size <= max_size:
            filtered_models.extend(models)

    return filtered_models


def get_coding_models(self) -> List[str]:
    """Get models designed for coding tasks.

    Coding models are identified by containing the words "code", "coding", or "coder"
    in their model name (case-insensitive).

    Returns:
        A list of model names that are designed for coding tasks.

    Raises:
        subprocess.CalledProcessError: If the ollama list command fails.
        FileNotFoundError: If ollama is not installed or not in PATH.
    """
    models = self.get_installed_models()
    coding_keywords = {'code', 'coding', 'coder'}
    coding_models = []

    for model in models:
        model_lower = model.lower()
        if any(keyword in model_lower for keyword in coding_keywords):
            coding_models.append(model)

    return coding_models


def get_non_coding_models(self) -> List[str]:
    """Get models that are not designed for coding tasks.

    Returns:
        A list of model names that are not designed for coding tasks.

    Raises:
        subprocess.CalledProcessError: If the ollama list command fails.
        FileNotFoundError: If ollama is not installed or not in PATH.
    """
    all_models = self.get_installed_models()
    coding_models = get_coding_models()
    coding_models_set = set(coding_models)
    non_coding_modules = [model for model in all_models if model not in coding_models_set]

    return non_coding_modules