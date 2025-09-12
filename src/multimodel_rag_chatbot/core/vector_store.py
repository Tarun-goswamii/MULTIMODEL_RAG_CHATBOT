"""Vector store implementation using ChromaDB."""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.schema import Document
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False
    print("Warning: ChromaDB/LangChain not available. Using simple in-memory storage.")
    
    # Simple document class if not available
    class Document:
        def __init__(self, page_content, metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}
    
    # Simple fallback storage
    class SimpleVectorStore:
        def __init__(self):
            self.documents = []
            self.embeddings = []
        
        def add_documents(self, documents):
            self.documents.extend(documents)
        
        def similarity_search(self, query, k=4):
            # Simple text-based search fallback
            results = []
            query_lower = query.lower()
            for doc in self.documents:
                if query_lower in doc.page_content.lower():
                    results.append(doc)
                    if len(results) >= k:
                        break
            return results
        
        def similarity_search_with_score(self, query, k=4):
            docs = self.similarity_search(query, k)
            return [(doc, 0.5) for doc in docs]  # Dummy scores
        
        def persist(self):
            pass

from .config import settings


class VectorStore:
    """Manage document embeddings and similarity search using ChromaDB."""
    
    def __init__(self, collection_name: str = "multimodel_rag", persist_directory: str = None):
        """Initialize the vector store.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the vector database
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory or settings.vector_db_path
        
        # Create persist directory if it doesn't exist
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        if HAS_CHROMADB:
            self._init_chromadb()
        else:
            self._init_simple_store()
    
    def _init_chromadb(self):
        """Initialize ChromaDB vector store."""
        # Initialize embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.default_embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize vector store
        self.vector_store = None
        self._initialize_vector_store()
    
    def _init_simple_store(self):
        """Initialize simple fallback storage."""
        self.vector_store = SimpleVectorStore()
    
    def _initialize_vector_store(self):
        """Initialize the Chroma vector store."""
        if not HAS_CHROMADB:
            return
            
        try:
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
                client_settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
        except Exception as e:
            print(f"Error initializing vector store: {str(e)}")
            # Fallback to simple store
            self.vector_store = SimpleVectorStore()
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store.
        
        Args:
            documents: List of documents to add
        """
        if not documents:
            print("No documents to add")
            return
        
        try:
            # Add documents to vector store
            self.vector_store.add_documents(documents)
            print(f"Added {len(documents)} documents to vector store")
            
            # Persist the changes
            self.vector_store.persist()
            
        except Exception as e:
            print(f"Error adding documents to vector store: {str(e)}")
            raise
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search for similar documents.
        
        Args:
            query: Search query
            k: Number of documents to return
            
        Returns:
            List of similar documents
        """
        try:
            results = self.vector_store.similarity_search(query, k=k)
            return results
        except Exception as e:
            print(f"Error during similarity search: {str(e)}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        """Search for similar documents with similarity scores.
        
        Args:
            query: Search query
            k: Number of documents to return
            
        Returns:
            List of (document, score) tuples
        """
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            return results
        except Exception as e:
            print(f"Error during similarity search with score: {str(e)}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the vector store collection.
        
        Returns:
            Dictionary containing collection information
        """
        if HAS_CHROMADB:
            try:
                collection = self.chroma_client.get_collection(self.collection_name)
                return {
                    "name": collection.name,
                    "count": collection.count(),
                    "metadata": collection.metadata
                }
            except Exception as e:
                print(f"Error getting collection info: {str(e)}")
                return {"name": self.collection_name, "count": 0, "metadata": {}}
        else:
            # Simple store info
            return {
                "name": self.collection_name,
                "count": len(self.vector_store.documents),
                "metadata": {"type": "simple_store"}
            }
    
    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        try:
            # Get the collection and delete all documents
            collection = self.chroma_client.get_collection(self.collection_name)
            ids = collection.get()["ids"]
            if ids:
                collection.delete(ids=ids)
                print(f"Cleared {len(ids)} documents from collection")
            else:
                print("Collection is already empty")
                
        except Exception as e:
            print(f"Error clearing collection: {str(e)}")
    
    def delete_collection(self) -> None:
        """Delete the entire collection."""
        try:
            self.chroma_client.delete_collection(self.collection_name)
            print(f"Deleted collection: {self.collection_name}")
            # Reinitialize vector store
            self._initialize_vector_store()
        except Exception as e:
            print(f"Error deleting collection: {str(e)}")
    
    def list_collections(self) -> List[str]:
        """List all collections in the vector database.
        
        Returns:
            List of collection names
        """
        try:
            collections = self.chroma_client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            print(f"Error listing collections: {str(e)}")
            return []