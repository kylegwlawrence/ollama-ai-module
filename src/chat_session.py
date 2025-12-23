import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

from src.model import OllamaModel

CHAT_SUMMARY_MODEL = 'gemma3:1b-it-q4_K_M'

class ChatSession:
  """Manages a conversation session with persistence."""

  def __init__(self, model: OllamaModel, session_name: str,
               sessions_dir: str = ".conversations") -> None:
    """Initialize chat session.

    Args:
      model: OllamaModel instance to chat with
      session_name: Session name for persistence
      sessions_dir: Directory to store session files (default: .conversations)
    """
    self.model = model
    self.session_name = session_name
    self.sessions_dir = Path(sessions_dir)
    self.sessions_dir.mkdir(exist_ok=True)

    self.messages: List[Dict[str, str]] = []
    self.created_at: Optional[str] = None

    # Load existing session if it exists
    self._load_session()

  def _get_session_path(self) -> Path:
    """Get the file path for this session."""
    # Sanitize session name to avoid path traversal
    safe_name = "".join(c for c in self.session_name if c.isalnum() or c in ('-', '_'))
    return self.sessions_dir / f"{safe_name}.json"

  def _load_session(self) -> None:
    """Load session from storage if it exists."""
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
    session_path = self._get_session_path()
    if self.created_at is None:
      self.created_at = datetime.now().isoformat()

    # Extract system prompt if it exists
    system_prompt = None
    system_messages = [m for m in self.messages if m.get('role') == 'system']
    if system_messages:
      system_prompt = system_messages[0].get('content')

    session_data = {
      'session_name': self.session_name,
      'model': self.model.model_name,
      'created_at': self.created_at,
      'last_updated': datetime.now().isoformat(),
      'system_prompt': system_prompt,
      'messages': self.messages
    }

    with open(session_path, 'w', encoding='utf-8') as f:
      json.dump(session_data, f, indent=2, ensure_ascii=False)

  def _update_session(self, **fields) -> None:
    """Update specific fields in an existing session file without overwriting.

    Args:
      **fields: Key-value pairs of fields to update in the session file.

    Raises:
      ValueError: If session file doesn't exist.

    Example:
      >>> session._update_session(conversation_summary="New summary")
      >>> session._update_session(last_updated=datetime.now().isoformat(), messages=new_messages)
    """
    session_path = self._get_session_path()

    if not session_path.exists():
      raise ValueError(f"Cannot update session '{self.session_name}' - file does not exist")

    # Load existing session data
    try:
      with open(session_path, 'r', encoding='utf-8') as f:
        session_data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
      raise ValueError(f"Error loading session '{self.session_name}': {e}")

    # Update with new fields
    session_data.update(fields)

    # Always update the last_updated timestamp
    session_data['last_updated'] = datetime.now().isoformat()

    # Save back to file
    with open(session_path, 'w', encoding='utf-8') as f:
      json.dump(session_data, f, indent=2, ensure_ascii=False)


  def send_message(self, user_message: str) -> str:
    """Send a message with chat history, get a response, and update chat history

    Args:
      user_message: User's message text

    Returns:
      Assistant's response text
    """
    # Add user message to history
    self.messages.append({'role': 'user', 'content': user_message})

    # Send message with full history using OllamaModel.send_prompt
    response = self.model.send_prompt(self.messages, return_output=True)

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

  def print_conversation(self) -> None:
    """Print the conversation history in a nicely formatted way.

    Displays all messages with the same formatting as the live chat session,
    making it look like the conversation is continuing from where it left off.
    System messages are skipped in the display.
    """
    if not self.messages:
      print("No conversation history.")
      return

    print("\n" + "=" * 50)
    print("CONVERSATION HISTORY")
    print("=" * 50 + "\n")

    for message in self.messages:
      role = message.get('role', 'unknown')
      content = message.get('content', '')

      if role == 'system':
        # Skip system messages in the display
        continue
      elif role == 'user':
        print(f"👤 You: {content}\n")
      elif role == 'assistant':
        print(f"🤖 Assistant: {content}")
        print(f"\n{'─' * 40}")
        print(f"  model: {self.model.model_name}")
        print(f"{'─' * 40}\n")
      else:
        print(f"[{role}]: {content}\n")

    print("=" * 50)
    print("END OF HISTORY")
    print("=" * 50 + "\n")

  def set_system_prompt(self, system_prompt: str) -> None:
    """Set or update system prompt.

    Args:
      system_prompt: System instruction text

    Note:
      This method does not automatically save the session. The session will be
      saved when send_message() is called or when explicitly saved via other methods.
    """
    # Remove existing system message if present
    self.messages = [m for m in self.messages if m.get('role') != 'system']

    # Add new system message at the beginning
    self.messages.insert(0, {'role': 'system', 'content': system_prompt})

  def summarize_conversation(self, model_name=CHAT_SUMMARY_MODEL, max_word_count=8, num_ctx: Optional[int] = None, num_predict: Optional[int] = None, timeout: Optional[float] = None) -> None:
    """Generate a concise summary of the entire conversation and save it to the session file.

    Analyzes all user and assistant messages to create a summary that describes
    the overall conversation topic and content.

    Args:
      model_name: The name of the model to use for summarization (default: 'qwen3:0.6b').
      max_word_count: Maximum number of words in the summary (default: 8).

    Returns:
      None

    Raises:
      ValueError: If no user messages exist in the conversation.

    Example:
      >>> session.summarize_conversation()
      >>> session.summarize_conversation(model_name='smollm2:135m', max_word_count=5)
    """
    from src.server import OllamaServer
    from src.model import OllamaModel

    # Get all user and assistant messages from the messages key (exclude system messages)
    conversation_messages = [m for m in self.messages if m.get('role') in ['user', 'assistant']]

    if not conversation_messages:
      raise ValueError(f"No conversation messages found in session '{self.session_name}'")

    # Build a chronological text representation of the conversation using only user and assistant messages
    conversation_text = ""
    for msg in conversation_messages:
      role = msg.get('role', '').capitalize()
      content = msg.get('content', '')
      conversation_text += f"{role}: {content}\n"

    # Initialize server and model for summarization
    server = OllamaServer()
    model = OllamaModel(model_name, server)

    # Create a prompt that instructs the model to summarize the conversation with heavy focus on early messages
    prompt = f"""You are a conversation summarization assistant. Your task is to create a concise summary focused on the ORIGINAL topic from the start of the conversation.

Instructions:
- Create a summary of the conversation in {max_word_count} words or fewer
- **FOCUS PRIMARILY on the first 3-5 message exchanges** - these contain the main topic and original intent
- The beginning of the conversation is the most important part for the summary
- Later messages may elaborate, drift, or change topics - de-emphasize these
- Identify the main topic, question, or problem from the BEGINNING of the conversation
- If the conversation shifted topics, summarize the ORIGINAL topic, not where it ended
- Use clear, descriptive language that captures the primary initial topic
- Output ONLY the summary text, nothing else

CONVERSATION THREAD:
{conversation_text}

Summary ({max_word_count} words max, focusing on the ORIGINAL topic from the start):"""

    # Get the summary from the model
    summary = model.prompt_generate_api(prompt, num_ctx=num_ctx, num_predict=num_predict, timeout=timeout)

    # Clean up the response (strip whitespace and newlines)
    summary = summary.strip()

    # Update the session file with the summary
    self._update_session(conversation_summary=summary)

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
          'messages': data.get('messages', []),
          'conversation_summary': data.get('conversation_summary')
        }
    except (json.JSONDecodeError, IOError):
      return None
