"""
therapy_assistant_api.py - FastAPI backend with persistent chat history
"""

from fastapi import (
    FastAPI,
    HTTPException,
)
from typing import List, Optional, Dict, Any
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from models.schemas import Message, ChatRequest, ChatResponse, SessionInfo
from services.llm_service import RetrievalObject
import uvicorn
import os
import json
from datetime import datetime
import uuid
from services.sessions_service import (
    get_session_path,
    save_session,
    load_session,
    convert_to_langchain_messages,
)
from dotenv import load_dotenv
from services.constants import SESSIONS_DIR

# Load environment variables
load_dotenv()
from services.embeddings_service import initializeEmbeddings, initializeVectorStore

embedding_model = initializeEmbeddings("sentence-transformers/all-mpnet-base-v2")
vector_store = initializeVectorStore(embedding_model)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
# Initialize FastAPI app
app = FastAPI(
    title="Therapy Chat Assistant API with Persistent History",
    description="API for mental health chat with conversation persistence",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify in production to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize LLM
llm = RetrievalObject()


# API endpoints
@app.post("/chat", response_model=ChatResponse)
async def process_chat(request: ChatRequest):
    try:
        # Get or create session ID
        session_id = request.session_id or str(uuid.uuid4())

        # Load existing messages for this session
        messages = load_session(session_id)

        # Add the new user message
        user_message = Message(
            role="user", content=request.input, timestamp=datetime.now().isoformat()
        )
        messages.append(user_message)

        # Convert messages to LangChain format
        chat_history = convert_to_langchain_messages(messages)

        # Create history-aware retriever for this request
        retrieval_chain = llm.createChain()

        # Add context document if provided

        # Invoke the chain
        response = retrieval_chain.invoke(
            {"input": request.input, "chat_history": chat_history}
        )

        # Extract the answer
        assistant_response = response.get(
            "answer", "I'm not sure how to respond to that."
        )

        # Add assistant response to messages
        assistant_message = Message(
            role="assistant",
            content=assistant_response,
            timestamp=datetime.now().isoformat(),
        )
        messages.append(assistant_message)

        # Save updated session
        save_session(session_id, messages)

        # Return response with session info
        return ChatResponse(response=assistant_response, session_id=session_id)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")


@app.get("/sessions/{session_id}", response_model=List[Message])
async def get_session_messages(session_id: str):
    """Get all messages for a specific session"""
    messages = load_session(session_id)
    if not messages:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return messages


@app.get("/sessions", response_model=List[SessionInfo])
async def list_sessions():
    """List all available chat sessions"""
    sessions = []
    for filename in os.listdir(SESSIONS_DIR):
        if filename.endswith(".json"):
            try:
                with open(os.path.join(SESSIONS_DIR, filename), "r") as f:
                    data = json.load(f)
                    sessions.append(
                        SessionInfo(
                            session_id=data["session_id"],
                            created_at=data["created_at"],
                            last_updated=data["last_updated"],
                            message_count=len(data["messages"]),
                        )
                    )
            except (json.JSONDecodeError, KeyError):
                continue
    return sessions


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session"""
    session_path = get_session_path(session_id)
    if os.path.exists(session_path):
        os.remove(session_path)
        return {"status": "success", "message": f"Session {session_id} deleted"}
    raise HTTPException(status_code=404, detail=f"Session {session_id} not found")


# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# Serve static files for the web interface
app.mount("/", StaticFiles(directory="static", html=True), name="static")

# For local development
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
