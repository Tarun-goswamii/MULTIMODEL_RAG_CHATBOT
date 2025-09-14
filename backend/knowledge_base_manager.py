#!/usr/bin/env python3
"""
Knowledge Base Manager
Handles embeddings and vector search for multi-modal content
"""

import os
import json
import hashlib
from datetime import datetime
from pathlib import Path

# Try imports with fallbacks
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import speech_recognition as sr
    HAS_SPEECH = True
except ImportError:
    HAS_SPEECH = False

class KnowledgeBaseManager:
    """Manages multi-modal knowledge base with vector search"""
    
    def __init__(self):
        self.qdrant_client = None
        self.gemini_client = None
        self.collection_name = "knowledge_base"
        self.local_storage = {}
        self.upload_folder = "uploads"
        
        # Initialize clients
        self._init_clients()
        
        # Ensure upload folder exists
        os.makedirs(self.upload_folder, exist_ok=True)
        
        # Load local storage
        self._load_local_storage()
        
        print("✅ Knowledge Base Manager initialized")
    
    def _init_clients(self):
        """Initialize Qdrant and Gemini clients"""
        # Initialize Gemini
        if HAS_GEMINI:
            api_key = os.getenv('GEMINI_API_KEY')
            if api_key:
                try:
                    genai.configure(api_key=api_key)
                    self.gemini_client = genai.GenerativeModel('gemini-1.5-flash')
                    print("✅ Gemini initialized for embeddings")
                except Exception as e:
                    print(f"⚠️ Gemini failed: {e}")
        
        # Initialize Qdrant
        if HAS_QDRANT:
            try:
                # Try cloud first
                api_key = os.getenv('QDRANT_API_KEY')
                if api_key:
                    self.qdrant_client = QdrantClient(
                        url="https://4b5dab1d-9418-432a-9751-647b64d7cdcd.us-east4-0.gcp.cloud.qdrant.io:6333",
                        api_key=api_key
                    )
                    print("✅ Qdrant Cloud connected")
                else:
                    # Try local
                    self.qdrant_client = QdrantClient("localhost", port=6333)
                    print("✅ Local Qdrant connected")
                
                # Create collection if needed
                self._ensure_collection()
                
            except Exception as e:
                print(f"⚠️ Qdrant failed: {e}, using local storage")
    
    def _ensure_collection(self):
        """Ensure knowledge base collection exists"""
        if not self.qdrant_client:
            return
        
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                # Create collection
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=768, distance=Distance.COSINE)  # Gemini embedding size
                )
                print(f"✅ Created collection: {self.collection_name}")
            else:
                print(f"✅ Collection exists: {self.collection_name}")
                
        except Exception as e:
            print(f"⚠️ Collection setup failed: {e}")
    
    def _load_local_storage(self):
        """Load local storage for fallback"""
        try:
            with open('knowledge_base.json', 'r') as f:
                self.local_storage = json.load(f)
        except FileNotFoundError:
            self.local_storage = {'documents': [], 'metadata': {'count': 0}}
        except Exception as e:
            print(f"⚠️ Local storage load failed: {e}")
            self.local_storage = {'documents': [], 'metadata': {'count': 0}}
    
    def _save_local_storage(self):
        """Save local storage"""
        try:
            with open('knowledge_base.json', 'w') as f:
                json.dump(self.local_storage, f, indent=2)
        except Exception as e:
            print(f"⚠️ Local storage save failed: {e}")
    
    def _generate_embedding(self, text):
        """Generate embedding for text using Gemini"""
        if not self.gemini_client:
            # Fallback: simple hash-based "embedding"
            hash_obj = hashlib.md5(text.encode())
            hash_hex = hash_obj.hexdigest()
            # Convert hex to numbers and normalize to 768 dimensions
            embedding = []
            for i in range(768):
                byte_val = int(hash_hex[i % len(hash_hex)], 16) / 15.0
                embedding.append(byte_val)
            return embedding
        
        try:
            # Use Gemini for embedding (simplified - in real implementation you'd use embedding models)
            # For now, create a mock embedding based on text content
            words = text.lower().split()
            embedding = [0.0] * 768
            
            # Simple word-based embedding
            for i, word in enumerate(words[:100]):  # Limit to 100 words
                word_hash = hash(word) % 768
                embedding[word_hash] += 1.0
            
            # Normalize
            max_val = max(embedding) if max(embedding) > 0 else 1.0
            embedding = [val / max_val for val in embedding]
            
            return embedding
            
        except Exception as e:
            print(f"⚠️ Embedding generation failed: {e}")
            return [0.1] * 768  # Fallback embedding
    
    def add_text_document(self, text, title="", category="text"):
        """Add text document to knowledge base"""
        try:
            # Generate embedding
            embedding = self._generate_embedding(text)
            
            # Create document metadata
            doc_id = len(self.local_storage['documents']) + 1
            document = {
                'id': doc_id,
                'type': 'text',
                'title': title or f"Document {doc_id}",
                'content': text,
                'category': category,
                'timestamp': datetime.now().isoformat(),
                'file_path': None,
                'embedding': embedding
            }
            
            # Store in Qdrant if available
            if self.qdrant_client:
                try:
                    point = PointStruct(
                        id=doc_id,
                        vector=embedding,
                        payload={
                            'type': 'text',
                            'title': document['title'],
                            'content': text[:1000],  # Limit content in payload
                            'category': category,
                            'timestamp': document['timestamp']
                        }
                    )
                    
                    self.qdrant_client.upsert(
                        collection_name=self.collection_name,
                        points=[point]
                    )
                    print(f"✅ Text document added to Qdrant: {title}")
                    
                except Exception as e:
                    print(f"⚠️ Qdrant upsert failed: {e}")
            
            # Store locally
            self.local_storage['documents'].append(document)
            self.local_storage['metadata']['count'] += 1
            self._save_local_storage()
            
            return {
                'success': True,
                'document_id': doc_id,
                'title': document['title'],
                'message': 'Text document added successfully'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to add text document: {e}'
            }
    
    def add_audio_document(self, audio_file_path, title=""):
        """Add audio document to knowledge base"""
        if not HAS_SPEECH:
            return {
                'success': False,
                'error': 'Speech recognition not available'
            }
        
        try:
            # Transcribe audio
            recognizer = sr.Recognizer()
            with sr.AudioFile(audio_file_path) as source:
                audio_data = recognizer.record(source)
                transcript = recognizer.recognize_google(audio_data)
            
            # Generate embedding from transcript
            embedding = self._generate_embedding(transcript)
            
            # Create document
            doc_id = len(self.local_storage['documents']) + 1
            document = {
                'id': doc_id,
                'type': 'audio',
                'title': title or f"Audio {doc_id}",
                'content': transcript,
                'category': 'audio',
                'timestamp': datetime.now().isoformat(),
                'file_path': audio_file_path,
                'embedding': embedding
            }
            
            # Store in Qdrant if available
            if self.qdrant_client:
                try:
                    point = PointStruct(
                        id=doc_id,
                        vector=embedding,
                        payload={
                            'type': 'audio',
                            'title': document['title'],
                            'content': transcript[:500],
                            'file_path': audio_file_path,
                            'timestamp': document['timestamp']
                        }
                    )
                    
                    self.qdrant_client.upsert(
                        collection_name=self.collection_name,
                        points=[point]
                    )
                    
                except Exception as e:
                    print(f"⚠️ Qdrant upsert failed: {e}")
            
            # Store locally
            self.local_storage['documents'].append(document)
            self.local_storage['metadata']['count'] += 1
            self._save_local_storage()
            
            return {
                'success': True,
                'document_id': doc_id,
                'title': document['title'],
                'transcript': transcript,
                'message': 'Audio document added successfully'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to add audio document: {e}'
            }
    
    def add_image_document(self, image_file_path, title=""):
        """Add image document to knowledge base"""
        if not HAS_PIL:
            return {
                'success': False,
                'error': 'PIL not available for image processing'
            }
        
        try:
            # Analyze image
            image = Image.open(image_file_path)
            
            # Generate image description (simplified)
            description = f"Image: {image.width}x{image.height} pixels, format: {image.format}, mode: {image.mode}"
            
            # Try Gemini vision if available
            if self.gemini_client:
                try:
                    vision_response = self.gemini_client.generate_content([
                        "Describe this image in detail:",
                        image
                    ])
                    description = vision_response.text
                except Exception as e:
                    print(f"⚠️ Gemini vision failed: {e}")
            
            # Generate embedding from description
            embedding = self._generate_embedding(description)
            
            # Create document
            doc_id = len(self.local_storage['documents']) + 1
            document = {
                'id': doc_id,
                'type': 'image',
                'title': title or f"Image {doc_id}",
                'content': description,
                'category': 'image',
                'timestamp': datetime.now().isoformat(),
                'file_path': image_file_path,
                'image_info': {
                    'width': image.width,
                    'height': image.height,
                    'format': image.format,
                    'mode': image.mode
                },
                'embedding': embedding
            }
            
            # Store in Qdrant if available
            if self.qdrant_client:
                try:
                    point = PointStruct(
                        id=doc_id,
                        vector=embedding,
                        payload={
                            'type': 'image',
                            'title': document['title'],
                            'content': description[:500],
                            'file_path': image_file_path,
                            'image_info': document['image_info'],
                            'timestamp': document['timestamp']
                        }
                    )
                    
                    self.qdrant_client.upsert(
                        collection_name=self.collection_name,
                        points=[point]
                    )
                    
                except Exception as e:
                    print(f"⚠️ Qdrant upsert failed: {e}")
            
            # Store locally
            self.local_storage['documents'].append(document)
            self.local_storage['metadata']['count'] += 1
            self._save_local_storage()
            
            return {
                'success': True,
                'document_id': doc_id,
                'title': document['title'],
                'description': description,
                'image_info': document['image_info'],
                'message': 'Image document added successfully'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to add image document: {e}'
            }
    
    def search_knowledge_base(self, query, limit=5, content_type=None):
        """Search knowledge base with vector similarity"""
        try:
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            
            # Search in Qdrant if available
            if self.qdrant_client:
                try:
                    search_filter = None
                    if content_type:
                        from qdrant_client.models import Filter, FieldCondition, MatchValue
                        search_filter = Filter(
                            must=[FieldCondition(key="type", match=MatchValue(value=content_type))]
                        )
                    
                    results = self.qdrant_client.search(
                        collection_name=self.collection_name,
                        query_vector=query_embedding,
                        query_filter=search_filter,
                        limit=limit
                    )
                    
                    # Format results
                    formatted_results = []
                    for result in results:
                        formatted_results.append({
                            'id': result.id,
                            'score': result.score,
                            'type': result.payload.get('type', 'unknown'),
                            'title': result.payload.get('title', 'Untitled'),
                            'content': result.payload.get('content', ''),
                            'file_path': result.payload.get('file_path'),
                            'timestamp': result.payload.get('timestamp'),
                            'metadata': result.payload
                        })
                    
                    return {
                        'success': True,
                        'results': formatted_results,
                        'query': query,
                        'total_found': len(formatted_results)
                    }
                    
                except Exception as e:
                    print(f"⚠️ Qdrant search failed: {e}")
            
            # Fallback to local search
            return self._local_search(query, limit, content_type, query_embedding)
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Search failed: {e}',
                'results': []
            }
    
    def _local_search(self, query, limit, content_type, query_embedding):
        """Local similarity search fallback"""
        try:
            results = []
            query_words = set(query.lower().split())
            
            for doc in self.local_storage['documents']:
                # Filter by content type if specified
                if content_type and doc['type'] != content_type:
                    continue
                
                # Simple keyword similarity
                doc_words = set(doc['content'].lower().split())
                keyword_similarity = len(query_words.intersection(doc_words)) / len(query_words.union(doc_words))
                
                # Simple vector similarity (cosine)
                if 'embedding' in doc:
                    try:
                        doc_embedding = doc['embedding']
                        dot_product = sum(a * b for a, b in zip(query_embedding, doc_embedding))
                        norm_a = sum(a * a for a in query_embedding) ** 0.5
                        norm_b = sum(b * b for b in doc_embedding) ** 0.5
                        vector_similarity = dot_product / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0
                    except:
                        vector_similarity = 0
                else:
                    vector_similarity = 0
                
                # Combined similarity
                similarity = (keyword_similarity * 0.6) + (vector_similarity * 0.4)
                
                if similarity > 0.1:  # Minimum threshold
                    results.append({
                        'id': doc['id'],
                        'score': similarity,
                        'type': doc['type'],
                        'title': doc['title'],
                        'content': doc['content'][:500],  # Truncate content
                        'file_path': doc.get('file_path'),
                        'timestamp': doc['timestamp'],
                        'metadata': doc
                    })
            
            # Sort by similarity
            results.sort(key=lambda x: x['score'], reverse=True)
            
            return {
                'success': True,
                'results': results[:limit],
                'query': query,
                'total_found': len(results),
                'search_method': 'local'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Local search failed: {e}',
                'results': []
            }
    
    def get_knowledge_base_stats(self):
        """Get knowledge base statistics"""
        try:
            stats = {
                'total_documents': len(self.local_storage['documents']),
                'by_type': {},
                'qdrant_connected': self.qdrant_client is not None,
                'gemini_available': self.gemini_client is not None
            }
            
            # Count by type
            for doc in self.local_storage['documents']:
                doc_type = doc['type']
                stats['by_type'][doc_type] = stats['by_type'].get(doc_type, 0) + 1
            
            # Qdrant stats if available
            if self.qdrant_client:
                try:
                    collection_info = self.qdrant_client.get_collection(self.collection_name)
                    stats['qdrant_points'] = collection_info.points_count
                except:
                    stats['qdrant_points'] = 'unknown'
            
            return stats
            
        except Exception as e:
            return {'error': f'Stats failed: {e}'}
    
    def delete_document(self, doc_id):
        """Delete document from knowledge base"""
        try:
            # Remove from Qdrant
            if self.qdrant_client:
                try:
                    self.qdrant_client.delete(
                        collection_name=self.collection_name,
                        points_selector=[doc_id]
                    )
                except Exception as e:
                    print(f"⚠️ Qdrant delete failed: {e}")
            
            # Remove from local storage
            self.local_storage['documents'] = [
                doc for doc in self.local_storage['documents'] if doc['id'] != doc_id
            ]
            self.local_storage['metadata']['count'] = len(self.local_storage['documents'])
            self._save_local_storage()
            
            return {
                'success': True,
                'message': f'Document {doc_id} deleted successfully'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Delete failed: {e}'
            }

# Global instance
knowledge_manager = KnowledgeBaseManager()