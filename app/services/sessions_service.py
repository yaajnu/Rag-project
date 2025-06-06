from typing import List, Optional, Dict, Any
from datetime import datetime
import os
from models.schemas import *
import json
from services.constants import SESSIONS_DIR

# from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

os.makedirs(SESSIONS_DIR, exist_ok=True)


def get_session_path(session_id: str) -> str:
    return os.path.join(SESSIONS_DIR, f"{session_id}.json")


def save_session(session_id: str, messages: List[Message]):
    session_data = {
        "session_id": session_id,
        "created_at": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
        "messages": [msg.dict() for msg in messages],
    }

    with open(get_session_path(session_id), "w") as f:
        json.dump(session_data, f, indent=2)


def load_session(session_id: str) -> List[Message]:
    try:
        with open(get_session_path(session_id), "r") as f:
            data = json.load(f)
            return [Message(**msg) for msg in data["messages"]]
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def convert_to_langchain_messages(messages: List[Message]):
    result = []
    for msg in messages:
        if msg.role == "user":
            result.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            result.append(AIMessage(content=msg.content))
        elif msg.role == "system":
            result.append(SystemMessage(content=msg.content))
    return result
