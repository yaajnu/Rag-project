from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
import os
from models.schemas import *


def initializeEmbeddings(model_name):
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    return embeddings


def initializeVectorStore(embedding_model):

    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index_name = "mental-health-chatbot-rag-with-hf"
    index = pc.Index(index_name)
    vector_store = PineconeVectorStore(index=index, embedding=embedding_model)
    return vector_store
