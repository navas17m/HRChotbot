"""
In-memory chat history manager.
Stores conversation history per session — no database required.
"""

from typing import List, Tuple
from langchain_core.messages import HumanMessage, AIMessage


class ChatHistoryManager:
    """
    Manages per-session conversation history entirely in memory.
    Each session is keyed by a session_id string.
    """

    def __init__(self):
        # { session_id: [(human_msg, ai_msg), ...] }
        self._store: dict[str, List[Tuple[str, str]]] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def get_history(self, session_id: str) -> List[Tuple[str, str]]:
        """Return the raw (human, ai) tuple list for a session."""
        return self._store.get(session_id, [])

    def get_langchain_history(self, session_id: str) -> List:
        """
        Return history as a list of LangChain message objects,
        ready to pass into a ConversationalRetrievalChain.
        """
        messages = []
        for human, ai in self._store.get(session_id, []):
            messages.append(HumanMessage(content=human))
            messages.append(AIMessage(content=ai))
        return messages

    def add_message(self, session_id: str, human: str, ai: str) -> None:
        """Append a new (human, ai) exchange to the session history."""
        if session_id not in self._store:
            self._store[session_id] = []
        self._store[session_id].append((human, ai))

    def clear_history(self, session_id: str) -> None:
        """Delete all history for a session."""
        self._store.pop(session_id, None)

    def list_sessions(self) -> List[str]:
        """Return all active session IDs."""
        return list(self._store.keys())

    def get_message_count(self, session_id: str) -> int:
        """Return the number of exchanges in a session."""
        return len(self._store.get(session_id, []))
