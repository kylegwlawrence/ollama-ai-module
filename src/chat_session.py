from typing import Dict, List, Optional, Any

from models import send_chat_message


class ChatSession:
  """Manages a conversation session with an Ollama model."""

  def __init__(self, model_name: str, host: str = '127.0.0.1', port: int = 11434,
               session_name: Optional[str] = None, session_manager: Optional[Any] = None) -> None:
    """Initialize chat session.

    Args:
      model_name: Name of the model to chat with
      host: Ollama server host (default: localhost)
      port: Ollama server port (default: 11434)
      session_name: Optional session name for persistence
      session_manager: Optional SessionManager instance for persistence
    """
    self.model_name = model_name
    self.host = host
    self.port = port
    self.session_name = session_name
    self.session_manager = session_manager
    self.messages: List[Dict[str, str]] = []
    self.created_at: Optional[str] = None

    # Load existing session if session_name and session_manager provided
    if self.session_name and self.session_manager:
      self._load_session()

  def _load_session(self) -> None:
    """Load session from storage if it exists."""
    if not self.session_manager or not self.session_name:
      return

    session_data = self.session_manager.load_session(self.session_name)
    if session_data:
      self.messages = session_data.get('messages', [])
      self.created_at = session_data.get('created_at')
      # Restore system prompt if it exists
      system_messages = [m for m in self.messages if m.get('role') == 'system']
      if not system_messages and session_data.get('system_prompt'):
        self.messages.insert(0, {'role': 'system', 'content': session_data['system_prompt']})

  def _save_session(self) -> None:
    """Save session to storage."""
    if not self.session_manager or not self.session_name:
      return

    # Extract system prompt if present
    system_prompt = None
    system_messages = [m for m in self.messages if m.get('role') == 'system']
    if system_messages:
      system_prompt = system_messages[0].get('content')

    self.session_manager.save_session(
      session_name=self.session_name,
      model=self.model_name,
      messages=self.messages,
      system_prompt=system_prompt,
      created_at=self.created_at
    )

  def send_message(self, user_message: str) -> str:
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
