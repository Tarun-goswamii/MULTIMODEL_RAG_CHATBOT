"""
Multimodel RAG Chatbot

A Retrieval-Augmented Generation chatbot that supports multiple AI models
and can process various document types for enhanced responses.
"""

__version__ = "0.1.0"
__author__ = "Tarun Goswami"

from .core.chatbot import MultimodelRAGChatbot
from .core.document_processor import DocumentProcessor
from .core.vector_store import VectorStore

__all__ = ["MultimodelRAGChatbot", "DocumentProcessor", "VectorStore"]