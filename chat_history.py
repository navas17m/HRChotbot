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
        self._store: dict[str, List[Tuple[str, str]]] = {}

    def get_history(self, session_id: str) -> List[Tuple[str, str]]:
        return self._store.get(session_id, [])

    def get_langchain_history(self, session_id: str) -> List:
        messages = []
        for human, ai in self._store.get(session_id, []):
            messages.append(HumanMessage(content=human))
            messages.append(AIMessage(content=ai))
        return messages

    def add_message(self, session_id: str, human: str, ai: str) -> None:
        if session_id not in self._store:
            self._store[session_id] = []
        self._store[session_id].append((human, ai))

    def clear_history(self, session_id: str) -> None:
        self._store.pop(session_id, None)

    def list_sessions(self) -> List[str]:
        return list(self._store.keys())

    def get_message_count(self, session_id: str) -> int:
        return len(self._store.get(session_id, []))
