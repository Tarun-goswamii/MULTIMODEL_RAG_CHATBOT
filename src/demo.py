#!/usr/bin/env python3
"""Simple CLI demo for the multimodel RAG chatbot."""

import os
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from multimodel_rag_chatbot.core.chatbot import MultimodelRAGChatbot


def main():
    """Main demo function."""
    print("🤖 Multimodel RAG Chatbot Demo")
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
    
    # Interactive demo
    print("\n💬 Chat Demo:")
    print("Type 'quit' to exit")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            print("🤔 Processing...")
            result = chatbot.chat(user_input, use_rag=True)
            
            print(f"\n🤖 Assistant: {result['response']}")
            
            if result.get('sources'):
                print(f"\n📚 Sources ({len(result['sources'])}):")
                for i, source in enumerate(result['sources'][:3], 1):
                    source_file = Path(source['source']).name
                    print(f"   {i}. {source_file}")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    main()