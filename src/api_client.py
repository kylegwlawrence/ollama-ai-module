import requests
from typing import Optional, Dict, List


# Custom Exception Hierarchy
class OllamaAPIException(Exception):
    """Base exception for Ollama API errors"""
    pass


class OllamaConnectionError(OllamaAPIException):
    """Raised when unable to connect to Ollama server"""
    pass


class OllamaTimeoutError(OllamaAPIException):
    """Raised when request times out"""
    pass


class OllamaHTTPError(OllamaAPIException):
    """Raised when API returns non-200 status code"""

    def __init__(self, status_code: int, response_text: str):
        self.status_code = status_code
        self.response_text = response_text
        super().__init__(f"HTTP {status_code}: {response_text}")


class OllamaAPIClient:
    """Centralized client for making HTTP requests to Ollama API"""

    def __init__(self, host: str = '127.0.0.1', port: int = 11434):
        """Initialize the API client with server connection details.

        Args:
            host: Ollama server host (default: localhost)
            port: Ollama server port (default: 11434)
        """
        self.host = host
        self.port = port
        self.base_url = f'http://{host}:{port}'

    def get_tags(self, timeout: Optional[float] = 2) -> Dict:
        """Fetch the list of models available on the server.

        Args:
            timeout: Request timeout in seconds (default: 2s for quick health check)

        Returns:
            Dictionary with 'models' key containing list of model info

        Raises:
            OllamaConnectionError: If unable to connect to server
            OllamaTimeoutError: If request times out
            OllamaHTTPError: If API returns non-200 status
            OllamaAPIException: If response cannot be parsed
        """
        return self._request('GET', '/api/tags', timeout=timeout or 2)

    def generate(self, model: str, prompt: str, stream: bool = False,
                 timeout: Optional[float] = 300) -> Dict:
        """Send a text prompt to generate a response.

        Args:
            model: Name of the model to use
            prompt: The prompt text to send
            stream: Whether to stream the response (not yet implemented)
            timeout: Request timeout in seconds (default: 300s for generation)

        Returns:
            Dictionary with 'response' key containing generated text

        Raises:
            OllamaConnectionError: If unable to connect to server
            OllamaTimeoutError: If request times out
            OllamaHTTPError: If API returns non-200 status
            OllamaAPIException: If response cannot be parsed
        """
        payload = {
            'model': model,
            'prompt': prompt,
            'stream': stream
        }
        return self._request('POST', '/api/generate', json_data=payload,
                           timeout=timeout or 300)

    def chat(self, model: str, messages: List[Dict[str, str]], stream: bool = False,
             timeout: Optional[float] = 300) -> Dict:
        """Send chat messages to get a conversational response.

        Args:
            model: Name of the model to use
            messages: List of message dicts with 'role' and 'content' keys
            stream: Whether to stream the response (not yet implemented)
            timeout: Request timeout in seconds (default: 300s for chat)

        Returns:
            Dictionary with 'message' key containing response

        Raises:
            OllamaConnectionError: If unable to connect to server
            OllamaTimeoutError: If request times out
            OllamaHTTPError: If API returns non-200 status
            OllamaAPIException: If response cannot be parsed
        """
        payload = {
            'model': model,
            'messages': messages,
            'stream': stream
        }
        return self._request('POST', '/api/chat', json_data=payload,
                           timeout=timeout or 300)

    def _request(self, method: str, endpoint: str, json_data: Optional[Dict] = None,
                 timeout: float = 2) -> Dict:
        """Make an HTTP request to the Ollama API.

        Args:
            method: HTTP method ('GET' or 'POST')
            endpoint: API endpoint (e.g., '/api/tags')
            json_data: JSON payload for POST requests
            timeout: Request timeout in seconds

        Returns:
            Parsed JSON response as dictionary

        Raises:
            OllamaConnectionError: If unable to connect to server
            OllamaTimeoutError: If request times out
            OllamaHTTPError: If API returns non-200 status
            OllamaAPIException: If response cannot be parsed
        """
        url = f'{self.base_url}{endpoint}'

        try:
            if method == 'GET':
                response = requests.get(url, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, json=json_data, timeout=timeout)
            else:
                raise OllamaAPIException(f"Unsupported HTTP method: {method}")

            if response.status_code != 200:
                raise OllamaHTTPError(response.status_code, response.text)

            return response.json()

        except requests.exceptions.ConnectionError as e:
            raise OllamaConnectionError(
                f"Failed to connect to Ollama server at {self.base_url}"
            ) from e
        except requests.exceptions.Timeout as e:
            raise OllamaTimeoutError(
                f"Request to {self.base_url}{endpoint} timed out after {timeout}s"
            ) from e
        except requests.exceptions.RequestException as e:
            raise OllamaAPIException(f"Request failed: {e}") from e
        except OllamaHTTPError:
            raise
        except ValueError as e:
            raise OllamaAPIException(f"Failed to parse JSON response: {e}") from e
        except OllamaAPIException:
            raise
