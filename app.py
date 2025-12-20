from src.server import OllamaServer
from src.model import OllamaModel
from src.sessions import ChatSession
from src.utilities import generate_human_id

def main(prompt, model_name, session_name=None, system_prompt=None):

    # instantiate an OllamaModel
    server = OllamaServer()
    model = OllamaModel(model_name, server)

    # Generate a new session name if one is not passed
    if session_name == None:
        session_name = generate_human_id()

    # Instantiate a ChatSession
    session = ChatSession(model, session_name)

    if system_prompt != None:
        # Set or update system prompt
        session.set_system_prompt(system_prompt)

    # send message with ChatSession.send_message()
    response = session.send_message(prompt)

    # print the response from the send_message() call
    print(response)
    
    # print session name
    print(session.session_name)
    
if __name__ == "__main__":
    main(
        prompt="do a better job summarizing the conversation",
        model_name="smollm2:360m",
        session_name="talk-ready-teacher",
        system_prompt="reply in less than 40 words"
    )