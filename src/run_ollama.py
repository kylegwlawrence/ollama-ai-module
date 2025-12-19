import threading
from typing import Tuple, Dict, Optional, Any

from resource_monitor import ResourceMonitor
from sessions import ChatSession
from server import OllamaServer

def run_ollama_chat_smart(chat_session: ChatSession, user_message: str) -> str:
  """Send chat message using API if server is running.

  Args:
    chat_session: ChatSession instance
    user_message: User's message text

  Returns:
    Assistant's response text

  Raises:
    RuntimeError: If Ollama server is not running
  """
  server = OllamaServer()

  # Send message via chat session (which maintains context)
  response = chat_session.send_message(user_message)

  return response


def run_chat_with_monitoring(chat_session: ChatSession, user_message: str) -> Tuple[str, Optional[Dict[str, Optional[float]]]]:
  """Send chat message with resource monitoring.

  Args:
    chat_session: ChatSession instance
    user_message: User's message text

  Returns:
    Tuple of (response text, resource statistics dict)

  Raises:
    RuntimeError: If Ollama server is not running
  """
  # Check that the server is running
  server = OllamaServer()
  server.test_running()

  # Get Ollama server process for monitoring
  proc = OllamaServer.get_process()
  if not proc:
    # If can't find process, just send message without monitoring
    print("Cannot find ollama PID. Sending chat without monitoring")
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