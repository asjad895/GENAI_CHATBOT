from typing import List, Dict

class ChatHistory:
    """
    A class to manage the history of chat messages, supporting adding, retrieving, and clearing messages.

    This class stores chat messages, where each message consists of a role (e.g., 'user' or 'assistant') 
    and its corresponding content. It provides methods to add messages to the history, retrieve the entire
    chat history, clear the history, and check the length of the chat history.

    Attributes:
        chat_history (List[Dict]): A list of dictionaries where each dictionary represents a message 
                                   with 'role' and 'content' keys.

    Methods:
        add_message(role: str, content: str) -> None:
            Adds a new message with the given role and content to the chat history.
        
        get_chat_history() -> List[Dict]:
            Returns the entire chat history as a list of dictionaries, each containing 'role' and 'content'.
        
        clear_chat_history() -> None:
            Clears the chat history, removing all stored messages.
        
        __len__() -> int:
            Returns the number of messages in the chat history.
    """
    
    def __init__(self) -> None:
        """
        Initializes an empty chat history.

        The chat history is represented as an empty list of dictionaries, where each dictionary contains
        a message with its 'role' and 'content'.
        """
        self.chat_history = []

    def add_message(self, role: str, content: str) -> None:
        """
        Adds a message to the chat history.

        Args:
            role (str): The role of the message sender, such as 'user' or 'assistant'.
            content (str): The content of the message to be added.
        
        Returns:
            None
        """
        self.chat_history.append({"role": role, "content": content})

    def get_chat_history(self) -> List[Dict]:
        """
        Retrieves the entire chat history.

        Returns:
            List[Dict]: A list of dictionaries where each dictionary represents a message, 
                        containing the 'role' and 'content' of the message.
        """
        return self.chat_history

    def clear_chat_history(self) -> None:
        """
        Clears the entire chat history.

        Removes all messages from the chat history.
        
        Returns:
            None
        """
        self.chat_history = []

    def __len__(self) -> int:
        """
        Returns the number of messages in the chat history.

        Returns:
            int: The number of messages stored in the chat history.
        """
        return len(self.chat_history)
