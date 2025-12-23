import requests
from typing import Optional, Dict, List

HOST = '127.0.0.1'
PORT = 11434
CONTEXT_WINDOW = 4096 # Includes system message, prompt and response. Default a high context window is better than truncating prompts. 
MAX_RESPONSE_TOKENS = 2048 # Response context size only. Will truncate response if it exceeds this limit. 
TIMEOUT = 3600 # 1 hour

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

    def __init__(self) -> None:
        """Initialize the API client with server connection details.

        Args:
            host: Ollama server host
            port: Ollama server port
        """
        
        self.host = HOST
        self.port = PORT
        self.base_url = f'http://{self.host}:{self.port}'

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
        return self._request('GET', '/api/tags', timeout=timeout)

    def generate(self, model: str, prompt: str, stream: bool = False,
                 num_ctx: Optional[int] = None, num_predict: Optional[int] = None, suffix: Optional[str] = None, timeout: Optional[float] = TIMEOUT) -> Dict:
        """Send a text prompt to generate a response.

        Args:
            model: Name of the model to use
            prompt: The prompt text to send
            stream: Whether to stream the response (not yet implemented)
            num_ctx: Context window size for the model
            num_predict: Maximum tokens allowed in the response (optional)
            suffix: Text to append after the generated response (optional)
            timeout: Request timeout in seconds (default: 600s for generation)

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
        self._add_num_ctx_option(payload, num_ctx)
        self._add_num_predict_option(payload, num_predict)
        self._add_suffix_option(payload, suffix)
        return self._request('POST', '/api/generate', json_data=payload,
                           timeout=timeout)

    def chat(self, model: str, messages: List[Dict[str, str]], stream: bool = False,
             num_ctx: Optional[int] = None, num_predict: Optional[int] = None, suffix: Optional[str] = None, timeout: Optional[float] = TIMEOUT) -> Dict:
        """Send chat messages to get a conversational response.

        Args:
            model: Name of the model to use
            messages: List of message dicts with 'role' and 'content' keys
            stream: Whether to stream the response (not yet implemented)
            num_ctx: Context window size for the model
            num_predict: Maximum tokens allowed in the response (optional)
            suffix: Text to append after the generated response (optional)
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
        self._add_num_ctx_option(payload, num_ctx)
        self._add_num_predict_option(payload, num_predict)
        self._add_suffix_option(payload, suffix)
        return self._request('POST', '/api/chat', json_data=payload,
                           timeout=timeout)

    def _add_num_ctx_option(self, payload: Dict, num_ctx: Optional[int]) -> None:
        """Add num_ctx to payload options.

        Args:
            payload: The payload dictionary to modify
            num_ctx: Context window size for the model (uses CONTEXT_WINDOW if None)
        """
        if num_ctx is None:
            num_ctx = CONTEXT_WINDOW
        payload['options'] = {'num_ctx': num_ctx}

    def _add_num_predict_option(self, payload: Dict, num_predict: Optional[int]) -> None:
        """Add num_predict to payload options.

        Args:
            payload: The payload dictionary to modify
            num_predict: Maximum tokens allowed in response (uses MAX_RESPONSE_TOKENS if None)
        """
        if num_predict is None:
            num_predict = MAX_RESPONSE_TOKENS
        if 'options' not in payload:
            payload['options'] = {}
        payload['options']['num_predict'] = num_predict

    def _add_suffix_option(self, payload: Dict, suffix: Optional[str]) -> None:
        """Add suffix to payload if provided.

        Args:
            payload: The payload dictionary to modify
            suffix: Text to append after the generated response (added if not None)
        """
        if suffix is not None:
            payload['suffix'] = suffix

    def show_model_info(self, model: str, timeout: Optional[float] = 2) -> Dict:
        """Fetch detailed information about a specific model.

        Args:
            model: Name of the model to get information about
            timeout: Request timeout in seconds (default: 2s)

        Returns:
            Dictionary containing detailed model information

        Raises:
            OllamaConnectionError: If unable to connect to server
            OllamaTimeoutError: If request times out
            OllamaHTTPError: If API returns non-200 status
            OllamaAPIException: If response cannot be parsed
        """
        payload = {'model': model}
        return self._request('POST', '/api/show', json_data=payload,
                           timeout=timeout or 2)

    def embeddings(self, model: str, prompt: str, num_ctx: int = CONTEXT_WINDOW, timeout: Optional[float] = TIMEOUT) -> Dict:
        """Generate embeddings for a given text using a specified model.

        Args:
            model: Name of the model to use for embedding
            prompt: Text to generate embeddings for
            num_ctx: Context window size for the model
            timeout: Request timeout in seconds (default: 30s)

        Returns:
            Dictionary containing the embedding vector and metadata

        Raises:
            OllamaConnectionError: If unable to connect to server
            OllamaTimeoutError: If request times out
            OllamaHTTPError: If API returns non-200 status
            OllamaAPIException: If response cannot be parsed
        """
        payload = {
            'model': model,
            'prompt': prompt
        }
        self._add_num_ctx_option(payload, num_ctx)
        return self._request('POST', '/api/embeddings', json_data=payload,
                           timeout=timeout or 30)

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
