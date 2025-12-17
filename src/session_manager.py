import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path


class SessionManager:
    """Manages persistent conversation sessions stored as JSON files."""

    def __init__(self, sessions_dir: str = ".conversations") -> None:
        """Initialize session manager.

        Args:
            sessions_dir: Directory to store session files (default: .conversations)
        """
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(exist_ok=True)

    def _get_session_path(self, session_name: str) -> Path:
        """Get the file path for a session.

        Args:
            session_name: Name of the session

        Returns:
            Path to the session file
        """
        # Sanitize session name to avoid path traversal
        safe_name = "".join(c for c in session_name if c.isalnum() or c in ('-', '_'))
        return self.sessions_dir / f"{safe_name}.json"

    def save_session(self, session_name: str, model: str, messages: List[Dict[str, str]],
                     system_prompt: Optional[str] = None, created_at: Optional[str] = None) -> None:
        """Save a conversation session to file.

        Args:
            session_name: Name of the session
            model: Model name used in the session
            messages: List of message dictionaries
            system_prompt: Optional system prompt
            created_at: Optional creation timestamp (defaults to now if not provided)
        """
        session_path = self._get_session_path(session_name)

        # Load existing session to preserve created_at if it exists
        existing_session = self.load_session(session_name)
        if existing_session and created_at is None:
            created_at = existing_session.get('created_at')

        if created_at is None:
            created_at = datetime.now().isoformat()

        session_data = {
            'session_name': session_name,
            'model': model,
            'created_at': created_at,
            'last_updated': datetime.now().isoformat(),
            'system_prompt': system_prompt,
            'messages': messages
        }

        with open(session_path, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)

    def load_session(self, session_name: str) -> Optional[Dict[str, Any]]:
        """Load a conversation session from file.

        Args:
            session_name: Name of the session

        Returns:
            Session data dictionary or None if session doesn't exist
        """
        session_path = self._get_session_path(session_name)

        if not session_path.exists():
            return None

        try:
            with open(session_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading session '{session_name}': {e}")
            return None

    def session_exists(self, session_name: str) -> bool:
        """Check if a session exists.

        Args:
            session_name: Name of the session

        Returns:
            True if session exists, False otherwise
        """
        return self._get_session_path(session_name).exists()

    def delete_session(self, session_name: str) -> bool:
        """Delete a conversation session.

        Args:
            session_name: Name of the session

        Returns:
            True if session was deleted, False if it didn't exist
        """
        session_path = self._get_session_path(session_name)

        if session_path.exists():
            session_path.unlink()
            return True
        return False

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all available sessions.

        Returns:
            List of session info dictionaries with name, model, created_at, last_updated
        """
        sessions = []

        for session_file in self.sessions_dir.glob("*.json"):
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

    def get_session_info(self, session_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a session.

        Args:
            session_name: Name of the session

        Returns:
            Session info dictionary or None if session doesn't exist
        """
        session_data = self.load_session(session_name)

        if session_data is None:
            return None

        return {
            'name': session_data.get('session_name'),
            'model': session_data.get('model'),
            'created_at': session_data.get('created_at'),
            'last_updated': session_data.get('last_updated'),
            'system_prompt': session_data.get('system_prompt'),
            'message_count': len(session_data.get('messages', [])),
            'messages': session_data.get('messages', [])
        }
