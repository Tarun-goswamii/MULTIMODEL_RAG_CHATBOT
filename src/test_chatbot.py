#!/usr/bin/env python3
"""Test script for the multimodel RAG chatbot."""

import os
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from multimodel_rag_chatbot.core.chatbot import MultimodelRAGChatbot


def main():
    """Test the chatbot functionality."""
    print("🤖 Multimodel RAG Chatbot Test")
    print("=" * 40)
    
    # Initialize chatbot
    print("\n📥 Initializing chatbot...")
    chatbot = MultimodelRAGChatbot()
    
    # Load sample documents
    docs_path = Path(__file__).parent.parent / "data" / "documents"
    if docs_path.exists():
        print(f"\n📁 Loading documents from: {docs_path}")
        count = chatbot.load_documents(str(docs_path))
        print(f"✅ Loaded {count} document chunks")
    else:
        print("\n⚠️ No documents directory found")
    
    # Show system info
    print("\n📊 System Information:")
    vector_info = chatbot.get_vector_store_info()
    print(f"   • Vector Store: {vector_info['name']}")
    print(f"   • Documents: {vector_info['count']}")
    print(f"   • Type: {vector_info.get('metadata', {}).get('type', 'unknown')}")
    
    models = chatbot.get_available_models()
    print(f"\n🧠 Available Models: {len(models)}")
    for model_id, description in models.items():
        print(f"   • {model_id}: {description}")
    
    if not models:
        print("   ⚠️ No AI models available - configure API keys to enable")
    
    # Test chat functionality
    print("\n💬 Chat Test:")
    test_queries = [
        "What is RAG?",
        "How does the document processing work?",
        "What AI models are supported?"
    ]
    
    for query in test_queries:
        print(f"\n❓ Query: {query}")
        result = chatbot.chat(query, use_rag=True)
        print(f"🤖 Response: {result['response'][:100]}...")
        
        if result.get('sources'):
            print(f"📚 Sources: {len(result['sources'])} documents")
    
    print("\n✅ Test completed successfully!")


if __name__ == "__main__":
    main()