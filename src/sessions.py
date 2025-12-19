import json
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
#### USE OllamaServer class and OllamaModel - just use OllamaModel because this contains OllamaServer


class ChatSession:
  """Manages a conversation session with persistence."""

  def __init__(self, model_name: str, session_name: Optional[str] = None,
               host: str = '127.0.0.1', port: int = 11434,
               sessions_dir: str = ".conversations") -> None:
    """Initialize chat session.

    Args:
      model_name: Name of the model to chat with
      session_name: Optional session name for persistence
      host: Ollama server host (default: localhost)
      port: Ollama server port (default: 11434)
      sessions_dir: Directory to store session files (default: .conversations)
    """
    self.model_name = model_name
    self.host = host
    self.port = port
    self.session_name = session_name
    self.sessions_dir = Path(sessions_dir)
    self.sessions_dir.mkdir(exist_ok=True)

    self.messages: List[Dict[str, str]] = []
    self.created_at: Optional[str] = None

    # Load existing session if session_name provided
    if self.session_name:
      self._load_session()

  def _get_session_path(self) -> Path:
    """Get the file path for this session."""
    if not self.session_name:
      return None
    # Sanitize session name to avoid path traversal
    safe_name = "".join(c for c in self.session_name if c.isalnum() or c in ('-', '_'))
    return self.sessions_dir / f"{safe_name}.json"

  def _send_chat_message(self, messages: List[Dict[str, str]], stream: bool = False) -> str:
    """Send chat messages to Ollama using /api/chat endpoint.

    Args:
      messages: List of message dicts with 'role' and 'content' keys
      stream: Enable streaming responses (default: False)

    Returns:
      The assistant's response as a string
    """
    try:
      url = f'http://{self.host}:{self.port}/api/chat'
      payload = {
        'model': self.model_name,
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

  def _load_session(self) -> None:
    """Load session from storage if it exists."""
    if not self.session_name:
      return

    session_path = self._get_session_path()
    if not session_path.exists():
      return

    try:
      with open(session_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        self.messages = data.get('messages', [])
        self.created_at = data.get('created_at')
    except (json.JSONDecodeError, IOError) as e:
      print(f"Error loading session '{self.session_name}': {e}")

  def _save_session(self) -> None:
    """Save session to storage."""
    if not self.session_name:
      return

    session_path = self._get_session_path()
    if self.created_at is None:
      self.created_at = datetime.now().isoformat()

    # Extract system prompt if present
    system_prompt = None
    system_messages = [m for m in self.messages if m.get('role') == 'system']
    if system_messages:
      system_prompt = system_messages[0].get('content')

    session_data = {
      'session_name': self.session_name,
      'model': self.model_name,
      'created_at': self.created_at,
      'last_updated': datetime.now().isoformat(),
      'system_prompt': system_prompt,
      'messages': self.messages
    }

    with open(session_path, 'w', encoding='utf-8') as f:
      json.dump(session_data, f, indent=2, ensure_ascii=False)

  def send_message(self, user_message: str) -> str:
    """Send a message and get response.

    Args:
      user_message: User's message text

    Returns:
      Assistant's response text
    """
    # Add user message to history
    self.messages.append({'role': 'user', 'content': user_message})

    # Send message with full history
    response = self._send_chat_message(self.messages)

    # Add assistant response to history
    self.messages.append({'role': 'assistant', 'content': response})

    # Save session after each message if persistence is enabled
    self._save_session()

    return response

  def reset(self) -> None:
    """Clear conversation history."""
    self.messages = []
    self.created_at = None
    self._save_session()

  def get_history(self) -> List[Dict[str, str]]:
    """Get conversation history.

    Returns:
      List of message dictionaries
    """
    return self.messages.copy()

  def set_system_prompt(self, system_prompt: str) -> None:
    """Set or update system prompt.

    Args:
      system_prompt: System instruction text
    """
    # Remove existing system message if present
    self.messages = [m for m in self.messages if m.get('role') != 'system']

    # Add new system message at the beginning
    self.messages.insert(0, {'role': 'system', 'content': system_prompt})

    # Save session after updating system prompt
    self._save_session()

  # Static methods for session management
  @staticmethod
  def list_sessions(sessions_dir: str = ".conversations") -> List[Dict[str, Any]]:
    """List all available sessions.

    Returns:
      List of session info dictionaries
    """
    sessions_path = Path(sessions_dir)
    if not sessions_path.exists():
      return []

    sessions = []
    for session_file in sessions_path.glob("*.json"):
      try:
        with open(session_file, 'r', encoding='utf-8') as f:
          data = json.load(f)
          sessions.append({
            'name': data.get('session_name', session_file.stem),
            'model': data.get('model', 'unknown'),
            'created_at': data.get('created_at', 'unknown'),
            'last_updated': data.get('last_updated', 'unknown'),
            'message_count': len(data.get('messages', []))
          })
      except (json.JSONDecodeError, IOError):
        continue

    return sorted(sessions, key=lambda x: x.get('last_updated', ''), reverse=True)

  @staticmethod
  def session_exists(session_name: str, sessions_dir: str = ".conversations") -> bool:
    """Check if a session exists."""
    if not session_name:
      return False
    safe_name = "".join(c for c in session_name if c.isalnum() or c in ('-', '_'))
    path = Path(sessions_dir) / f"{safe_name}.json"
    return path.exists()

  @staticmethod
  def delete_session(session_name: str, sessions_dir: str = ".conversations") -> bool:
    """Delete a conversation session.

    Returns:
      True if session was deleted, False if it didn't exist
    """
    if not session_name:
      return False
    safe_name = "".join(c for c in session_name if c.isalnum() or c in ('-', '_'))
    path = Path(sessions_dir) / f"{safe_name}.json"
    if path.exists():
      path.unlink()
      return True
    return False

  @staticmethod
  def get_session_info(session_name: str, sessions_dir: str = ".conversations") -> Optional[Dict[str, Any]]:
    """Get detailed information about a session.

    Returns:
      Session info dictionary or None if session doesn't exist
    """
    if not session_name:
      return None
    safe_name = "".join(c for c in session_name if c.isalnum() or c in ('-', '_'))
    path = Path(sessions_dir) / f"{safe_name}.json"

    if not path.exists():
      return None

    try:
      with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return {
          'name': data.get('session_name'),
          'model': data.get('model'),
          'created_at': data.get('created_at'),
          'last_updated': data.get('last_updated'),
          'system_prompt': data.get('system_prompt'),
          'message_count': len(data.get('messages', [])),
          'messages': data.get('messages', [])
        }
    except (json.JSONDecodeError, IOError):
      return None
