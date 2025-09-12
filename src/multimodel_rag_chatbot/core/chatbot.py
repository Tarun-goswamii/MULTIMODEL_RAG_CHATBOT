"""Main chatbot implementation combining RAG and multiple models."""

from typing import List, Dict, Any, Optional, Tuple
import json

from ..core.document_processor import DocumentProcessor
from ..core.vector_store import VectorStore
from ..models.model_manager import ModelManager
from ..core.config import settings


class MultimodelRAGChatbot:
    """Main chatbot class that combines RAG with multiple AI models."""
    
    def __init__(self, collection_name: str = "multimodel_rag"):
        """Initialize the RAG chatbot.
        
        Args:
            collection_name: Name of the vector database collection
        """
        self.collection_name = collection_name
        
        # Initialize components
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore(collection_name=collection_name)
        self.model_manager = ModelManager()
        
        # Chat history
        self.chat_history = []
    
    def load_documents(self, directory_path: str) -> int:
        """Load documents from a directory into the vector store.
        
        Args:
            directory_path: Path to directory containing documents
            
        Returns:
            Number of document chunks loaded
        """
        print(f"Loading documents from: {directory_path}")
        
        # Process documents
        documents = self.document_processor.process_directory(directory_path)
        
        if documents:
            # Add to vector store
            self.vector_store.add_documents(documents)
            print(f"Successfully loaded {len(documents)} document chunks")
        else:
            print("No documents found or processed")
        
        return len(documents)
    
    def get_relevant_context(self, query: str, k: int = 4) -> Tuple[List[str], List[Dict]]:
        """Retrieve relevant context for a query.
        
        Args:
            query: User query
            k: Number of relevant documents to retrieve
            
        Returns:
            Tuple of (context_texts, source_metadata)
        """
        # Search for relevant documents
        results = self.vector_store.similarity_search_with_score(query, k=k)
        
        if not results:
            return [], []
        
        context_texts = []
        source_metadata = []
        
        for doc, score in results:
            context_texts.append(doc.page_content)
            
            # Extract source info
            metadata = doc.metadata.copy()
            metadata['similarity_score'] = float(score)
            source_metadata.append(metadata)
        
        return context_texts, source_metadata
    
    def create_rag_prompt(self, query: str, context_texts: List[str]) -> str:
        """Create a RAG prompt with context and query.
        
        Args:
            query: User query
            context_texts: List of relevant context texts
            
        Returns:
            Formatted prompt for the AI model
        """
        if not context_texts:
            return f"""
You are a helpful AI assistant. Please answer the following question:

Question: {query}

Note: No relevant documents were found in the knowledge base for this query.
Please provide a helpful response based on your general knowledge.
"""
        
        context_section = "\n\n".join([f"Document {i+1}:\n{text}" 
                                      for i, text in enumerate(context_texts)])
        
        prompt = f"""
You are a helpful AI assistant. Use the following documents to answer the question.
If the documents don't contain relevant information, say so and provide a general response.

Context Documents:
{context_section}

Question: {query}

Please provide a comprehensive answer based on the context documents above.
If you use information from the documents, please indicate which document(s) you referenced.
"""
        
        return prompt
    
    def chat(self, query: str, model_id: str = None, use_rag: bool = True, k: int = 4) -> Dict[str, Any]:
        """Process a chat query and return response with metadata.
        
        Args:
            query: User query
            model_id: ID of the model to use (optional)
            use_rag: Whether to use RAG for context retrieval
            k: Number of relevant documents to retrieve
            
        Returns:
            Dictionary containing response and metadata
        """
        # Get available models
        available_models = self.model_manager.get_available_models()
        
        if not available_models:
            return {
                "response": "No AI models are available. Please configure API keys for OpenAI or Anthropic.",
                "model_used": None,
                "context_used": [],
                "sources": [],
                "error": "No models available"
            }
        
        # Use RAG if enabled
        context_texts = []
        sources = []
        
        if use_rag:
            try:
                context_texts, sources = self.get_relevant_context(query, k=k)
                print(f"Retrieved {len(context_texts)} relevant documents")
            except Exception as e:
                print(f"Error retrieving context: {str(e)}")
        
        # Create prompt
        if use_rag and context_texts:
            prompt = self.create_rag_prompt(query, context_texts)
        else:
            prompt = query
        
        # Generate response
        try:
            response = self.model_manager.generate_response(prompt, model_id=model_id)
            
            # Determine which model was actually used
            actual_model = model_id or settings.default_model
            if not self.model_manager.is_model_available(actual_model):
                # Find the first available model
                for mid in available_models.keys():
                    if self.model_manager.is_model_available(mid):
                        actual_model = mid
                        break
            
            # Store in chat history
            chat_entry = {
                "query": query,
                "response": response,
                "model_used": actual_model,
                "context_used": len(context_texts),
                "use_rag": use_rag
            }
            self.chat_history.append(chat_entry)
            
            return {
                "response": response,
                "model_used": actual_model,
                "model_description": available_models.get(actual_model, "Unknown"),
                "context_used": context_texts,
                "sources": sources,
                "use_rag": use_rag,
                "available_models": available_models
            }
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(error_msg)
            
            return {
                "response": error_msg,
                "model_used": model_id,
                "context_used": [],
                "sources": [],
                "error": str(e)
            }
    
    def get_vector_store_info(self) -> Dict[str, Any]:
        """Get information about the vector store.
        
        Returns:
            Dictionary containing vector store information
        """
        return self.vector_store.get_collection_info()
    
    def clear_documents(self) -> None:
        """Clear all documents from the vector store."""
        self.vector_store.clear_collection()
        print("Cleared all documents from vector store")
    
    def get_chat_history(self) -> List[Dict[str, Any]]:
        """Get the chat history.
        
        Returns:
            List of chat interactions
        """
        return self.chat_history
    
    def clear_chat_history(self) -> None:
        """Clear the chat history."""
        self.chat_history = []
        print("Cleared chat history")
    
    def get_available_models(self) -> Dict[str, str]:
        """Get available AI models.
        
        Returns:
            Dictionary of model_id -> description
        """
        return self.model_manager.get_available_models()
    
    def export_chat_history(self, filename: str) -> None:
        """Export chat history to a JSON file.
        
        Args:
            filename: Name of the file to save the history to
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.chat_history, f, indent=2, ensure_ascii=False)
            print(f"Chat history exported to: {filename}")
        except Exception as e:
            print(f"Error exporting chat history: {str(e)}")
    
    def import_chat_history(self, filename: str) -> None:
        """Import chat history from a JSON file.
        
        Args:
            filename: Name of the file to load the history from
        """
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                self.chat_history = json.load(f)
            print(f"Chat history imported from: {filename}")
        except Exception as e:
            print(f"Error importing chat history: {str(e)}")