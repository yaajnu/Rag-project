"""
therapy_assistant_api.py - FastAPI backend with persistent chat history
"""

from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    Body,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
import os
import json
from datetime import datetime
import uuid

from langchain.schema import Document, SystemMessage, HumanMessage, AIMessage
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings

# Initialize vector store and embedding model
embeddings = OllamaEmbeddings(model="llama3")
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = "mental-health-chatbot-rag-2"
index = pc.Index(index_name)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)
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
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", google_api_key=os.environ.get("googleAPIKey")
)


# Create the base document chain
def create_therapy_chain():
    # Define message templates
    system_template = """The context below consists of previous people speaking with their therapists about their issues, use these contexts 
as a pointer and then answer the query from the user based on the assistant's responses in the previous chats. Answer only questions relevant to mental health and don't answer if the question is irrelevant.
ABSOLUTELY AVOID ANSWERING IRRELEVANT QUESTIONS. 

    Contexts:
    {context}"""

    system_message_template = SystemMessagePromptTemplate.from_template(system_template)
    human_template = "{input}"
    human_message_template = HumanMessagePromptTemplate.from_template(human_template)

    # Create prompt template
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            system_message_template,
            MessagesPlaceholder(variable_name="chat_history"),
            human_message_template,
        ]
    )

    # Create and return document chain
    return create_stuff_documents_chain(llm=llm, prompt=chat_prompt)


# Create the base chain at startup
therapy_chain = create_therapy_chain()

# Session storage
# In production, consider using a database instead of in-memory storage
SESSIONS_DIR = "chat_sessions"
os.makedirs(SESSIONS_DIR, exist_ok=True)


# Pydantic models
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
    messages: List[Message]


class SessionInfo(BaseModel):
    session_id: str
    created_at: str
    last_updated: str
    message_count: int


# Helper functions
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
        retriever_prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
                (
                    "user",
                    "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation",
                ),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            llm=llm, retriever=retriever, prompt=retriever_prompt
        )

        # Create retrieval chain with the history-aware retriever
        retrieval_chain = create_retrieval_chain(
            history_aware_retriever,
            therapy_chain,
        )

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
        return ChatResponse(
            response=assistant_response, session_id=session_id, messages=messages
        )

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
# app.mount("/", StaticFiles(directory="static", html=True), name="static")

# For local development
if __name__ == "__main__":
    uvicorn.run("therapy_assistant_api:app", host="0.0.0.0", port=8000, reload=True)
