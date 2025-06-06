import os
from dotenv import load_dotenv

load_dotenv("./.env")

pinecone_key = os.environ.get("PINECONE_API_KEY")
google_api_key = os.environ.get("googleAPIKey")
SESSIONS_DIR = "chat_sessions"
