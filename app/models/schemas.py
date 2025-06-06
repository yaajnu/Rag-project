# Pydantic models
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class Message(BaseModel):
    role: str
    content: str
    timestamp: Optional[str] = None


class ChatRequest(BaseModel):
    session_id: Optional[str] = Field(
        default=None, description="Session ID for continuing a conversation"
    )
    input: str = Field(..., description="The user's current query")
    # context: str = Field(default="", description="Additional context (optional)")


class ChatResponse(BaseModel):
    response: str
    session_id: str
    # messages: List[Message]


class SessionInfo(BaseModel):
    session_id: str
    created_at: str
    last_updated: str
    message_count: int
