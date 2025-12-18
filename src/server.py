import subprocess
import psutil
from typing import Optional


def test_ollama_server_running(host: str = '127.0.0.1', port: int = 11434, timeout: int = 2) -> None:
  """Check if Ollama server is running, raise error if not.

  Args:
    host: Ollama server host (default: localhost)
    port: Ollama server port (default: 11434)
    timeout: Connection timeout in seconds

  Raises:
    RuntimeError: If Ollama server is not running
  """
  try:
    result = subprocess.run(
      ['curl', '-s', '-o', '/dev/null', '-w', '%{http_code}', f'http://{host}:{port}/api/tags'],
      capture_output=True,
      text=True,
      timeout=timeout
    )
    if result.stdout.strip() != '200':
      raise RuntimeError(
        f"Non-200 code. Ollama server is not running on {host}:{port}. "
        "Please start the Ollama server manually before running this function."
      )
  except subprocess.SubprocessError:
    raise RuntimeError(
      f"Ollama server is not running on {host}:{port}. "
      "Please start the Ollama server manually before running this function."
    )


def get_ollama_process() -> Optional[psutil.Process]:
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
