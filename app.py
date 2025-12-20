from src.server import OllamaServer
from src.model import OllamaModel
from src.chat_session import ChatSession
from src.utilities import generate_human_id

def start_conversation(prompt:str, model_name:str, system_prompt:str, **server_kwargs:dict) -> dict:
    """
    Start a new conversation with an Ollama model.

    Creates a new chat session with a generated session ID, sets the system prompt,
    and sends the initial prompt to the model.

    Args:
        prompt: The initial user prompt to send to the model.
        model_name: The name of the Ollama model to use (e.g., 'smollm2:360m', 'llama2').
        system_prompt: Instructions that guide the AI's behavior and response style.

    Returns:
        A dictionary containing:
            - 'response': The model's response text to the initial prompt
            - 'session_name': The unique identifier for this conversation session

    Example:
        >>> result = start_conversation("Hello", "smollm2:360m", "You are helpful")
        >>> print(result['response'])
        >>> print(result['session_name'])
    """

    # Instantiate OllamaServer and OllamaModel
    server = OllamaServer(**server_kwargs)
    model = OllamaModel(model_name, server)
    
    # Instantiate a new ChatSession
    session = ChatSession(model, generate_human_id())

    # Set up the system prompt (ie. instructions for the AI response behaviour)
    session.set_system_prompt(system_prompt)

    # Send a new message
    response = session.send_message(prompt)
    
    # Return the response and the session name
    return {"response": response, "session_name": session.session_name}
    
def continue_conversation(prompt:str, session_name:str, **server_kwargs) -> str:
    """
    Continue an existing conversation with an Ollama model.

    Retrieves a previous chat session by its name, restores the model context,
    and sends a new prompt while maintaining conversation history.

    Args:
        prompt: The user's message to send in the ongoing conversation.
        session_name: The unique identifier of the session to continue.

    Returns:
        The model's response text to the prompt.

    Raises:
        ValueError: If the session_name does not correspond to an existing session.

    Example:
        >>> response = continue_conversation("Tell me more", "session_abc123")
        >>> print(response)
    """

    # Retrieve session information
    session_information = ChatSession.get_session_info(session_name)
    session_model = session_information["model"]
    
    # Instantiate objects
    server = OllamaServer(**server_kwargs)
    model = OllamaModel(session_model, server)
    
    # Restore the session
    restored_session = ChatSession(model, session_name)
    
    # Send new message
    response = restored_session.send_message(prompt)
    
    return response