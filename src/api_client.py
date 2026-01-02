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

    def generate(self, model: str, prompt: str, stream: bool = False, **kwargs) -> Dict:
        """Send a text prompt to generate a response.

        Args:
            model: Name of the model to use
            prompt: The prompt text to send
            stream: Whether to stream the response (not yet implemented)
            **kwargs: Optional parameters including:
                num_ctx: Context window size for the model
                num_predict: Maximum tokens allowed in the response
                suffix: Text to append after the generated response
                timeout: Request timeout in seconds (default: 3600s)

        Returns:
            Dictionary with 'response' key containing generated text

        Raises:
            OllamaConnectionError: If unable to connect to server
            OllamaTimeoutError: If request times out
            OllamaHTTPError: If API returns non-200 status
            OllamaAPIException: If response cannot be parsed
        """
        timeout = kwargs.pop('timeout', TIMEOUT)
        payload = {
            'model': model,
            'prompt': prompt,
            'stream': stream
        }
        self._build_options(payload, kwargs)
        return self._request('POST', '/api/generate', json_data=payload,
                           timeout=timeout)

    def chat(self, model: str, messages: List[Dict[str, str]], stream: bool = False, **kwargs) -> Dict:
        """Send chat messages to get a conversational response.

        Args:
            model: Name of the model to use
            messages: List of message dicts with 'role' and 'content' keys
            stream: Whether to stream the response (not yet implemented)
            **kwargs: Optional parameters including:
                num_ctx: Context window size for the model
                num_predict: Maximum tokens allowed in the response
                suffix: Text to append after the generated response
                timeout: Request timeout in seconds (default: 3600s)

        Returns:
            Dictionary with 'message' key containing response

        Raises:
            OllamaConnectionError: If unable to connect to server
            OllamaTimeoutError: If request times out
            OllamaHTTPError: If API returns non-200 status
            OllamaAPIException: If response cannot be parsed
        """
        timeout = kwargs.pop('timeout', TIMEOUT)
        payload = {
            'model': model,
            'messages': messages,
            'stream': stream
        }
        self._build_options(payload, kwargs)
        return self._request('POST', '/api/chat', json_data=payload,
                           timeout=timeout)

<<<<<<< HEAD
    def _build_options(self, payload: Dict, options: Dict) -> None:
        """Build options dictionary from kwargs and add to payload.

        Args:
            payload: The payload dictionary to modify
            options: Dictionary of optional parameters (num_ctx, num_predict, suffix)
        """
        payload_options = {}

        # Handle num_ctx with default
        if 'num_ctx' in options:
            payload_options['num_ctx'] = options.pop('num_ctx')
        else:
            payload_options['num_ctx'] = CONTEXT_WINDOW

        # Handle num_predict with default
        if 'num_predict' in options:
            payload_options['num_predict'] = options.pop('num_predict')
        else:
            payload_options['num_predict'] = MAX_RESPONSE_TOKENS

        payload['options'] = payload_options

        # Handle suffix (optional, no default)
        if 'suffix' in options:
            payload['suffix'] = options.pop('suffix')

    def show_model_info(self, model: str, timeout: Optional[float] = 2) -> Dict:
        """Fetch detailed information about a specific model.

        Args:
            model: Name of the model to get information about
            timeout: Request timeout in seconds (default: 2s)

        Returns:
            Dictionary containing detailed model information
=======
    def chat_with_tools(self, model: str, messages: List[Dict[str, str]],
                        tools: List[Dict], stream: bool = False,
                        timeout: Optional[float] = 600) -> Dict:
        """Send chat messages with tool definitions.

        Args:
            model: Name of the model to use (must support tools)
            messages: List of message dicts with 'role' and 'content' keys
            tools: List of tool definitions in OpenAI-compatible format
            stream: Whether to stream the response (not yet implemented)
            timeout: Request timeout in seconds

        Returns:
            Dictionary with 'message' key containing response and optional 'tool_calls'
>>>>>>> fa6db45 (Add ToolAgent with file read/write capabilities for Ollama models)

        Raises:
            OllamaConnectionError: If unable to connect to server
            OllamaTimeoutError: If request times out
            OllamaHTTPError: If API returns non-200 status
            OllamaAPIException: If response cannot be parsed
        """
<<<<<<< HEAD
        payload = {'model': model}
        return self._request('POST', '/api/show', json_data=payload,
                           timeout=timeout or 2)

    def embeddings(self, model: str, prompt: str, **kwargs) -> Dict:
        """Generate embeddings for a given text using a specified model.

        Args:
            model: Name of the model to use for embedding
            prompt: Text to generate embeddings for
            **kwargs: Optional parameters including:
                num_ctx: Context window size for the model (default: 4096)
                timeout: Request timeout in seconds (default: 3600s)

        Returns:
            Dictionary containing the embedding vector and metadata

        Raises:
            OllamaConnectionError: If unable to connect to server
            OllamaTimeoutError: If request times out
            OllamaHTTPError: If API returns non-200 status
            OllamaAPIException: If response cannot be parsed
        """
        timeout = kwargs.pop('timeout', TIMEOUT)
        payload = {
            'model': model,
            'prompt': prompt
        }
        self._build_options(payload, kwargs)
        return self._request('POST', '/api/embeddings', json_data=payload,
=======
        payload = {
            'model': model,
            'messages': messages,
            'tools': tools,
            'stream': stream
        }
        return self._request('POST', '/api/chat', json_data=payload,
>>>>>>> fa6db45 (Add ToolAgent with file read/write capabilities for Ollama models)
                           timeout=timeout)

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
