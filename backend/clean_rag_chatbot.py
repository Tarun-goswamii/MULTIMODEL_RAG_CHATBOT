#!/usr/bin/env python3
"""
Clean RAG Chatbot - No Problematic Dependencies
Simplified version without sentence_transformers and scikit-learn
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path

# Core imports only
try:
    import google.generativeai as genai
    HAS_GEMINI = True
    print("✅ Google Gemini available")
except ImportError:
    HAS_GEMINI = False
    print("⚠️ Google Gemini not available")

# Simple Qdrant import
try:
    from qdrant_client import QdrantClient
    HAS_QDRANT = True
    print("✅ Qdrant client available")
except ImportError:
    HAS_QDRANT = False
    print("⚠️ Qdrant client not available")

class MultiModalRAGChatbot:
    """Clean Multi-Modal RAG Chatbot without problematic dependencies"""
    
    def __init__(self, qdrant_client=None):
        self.qdrant_client = qdrant_client
        self.gemini_client = None
        self.conversation_history = []
        self.knowledge_base = []
        
        # Initialize Gemini
        self._init_gemini()
        
        # Load basic knowledge
        self._load_basic_knowledge()
        
        print("✅ Clean RAG Chatbot initialized")
    
    def _init_gemini(self):
        """Initialize Google Gemini"""
        if not HAS_GEMINI:
            return
        
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key:
            try:
                genai.configure(api_key=api_key)
                self.gemini_client = genai.GenerativeModel('gemini-1.5-flash')
                print("✅ Gemini initialized for RAG")
            except Exception as e:
                print(f"⚠️ Gemini initialization failed: {e}")
    
    def _load_basic_knowledge(self):
        """Load basic knowledge base"""
        self.knowledge_base = [
            {
                "content": "This is a multi-modal RAG interview system that combines AI chatbots with real-time monitoring for comprehensive candidate evaluation.",
                "topic": "system_info",
                "keywords": ["system", "interview", "AI", "monitoring", "evaluation"]
            },
            {
                "content": "For coding interviews, practice data structures, algorithms, system design, and behavioral questions. Focus on problem-solving approach and communication.",
                "topic": "interview_prep",
                "keywords": ["coding", "interview", "algorithms", "data structures", "system design"]
            },
            {
                "content": "System design involves scalability, reliability, consistency, availability, and performance considerations. Start with requirements and scale gradually.",
                "topic": "system_design",
                "keywords": ["system design", "scalability", "reliability", "performance", "architecture"]
            },
            {
                "content": "RAG (Retrieval Augmented Generation) combines information retrieval with text generation to provide contextually relevant responses using external knowledge.",
                "topic": "rag",
                "keywords": ["RAG", "retrieval", "generation", "context", "knowledge"]
            },
            {
                "content": "Vector databases store and query high-dimensional vectors efficiently using similarity search algorithms like cosine similarity.",
                "topic": "vector_db",
                "keywords": ["vector", "database", "similarity", "search", "embeddings"]
            }
        ]
        print(f"📚 Loaded {len(self.knowledge_base)} knowledge entries")
    
    def search_similar_content(self, query, limit=3):
        """Search for similar content using simple keyword matching"""
        try:
            query_words = set(query.lower().split())
            results = []
            
            for item in self.knowledge_base:
                # Simple keyword similarity
                item_words = set(item['content'].lower().split())
                item_keywords = set([kw.lower() for kw in item['keywords']])
                
                # Calculate similarity scores
                content_similarity = len(query_words.intersection(item_words)) / len(query_words.union(item_words))
                keyword_similarity = len(query_words.intersection(item_keywords)) / len(query_words.union(item_keywords))
                
                # Combined similarity
                similarity = (content_similarity * 0.7) + (keyword_similarity * 0.3)
                
                if similarity > 0.1:  # Minimum threshold
                    results.append({
                        'content': item['content'],
                        'topic': item['topic'],
                        'similarity': similarity,
                        'keywords': item['keywords']
                    })
            
            # Sort by similarity
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:limit]
            
        except Exception as e:
            print(f"⚠️ Search failed: {e}")
            return []
    
    def generate_contextual_response(self, query, context_results):
        """Generate response using context and AI"""
        try:
            # Build context from search results
            context = ""
            if context_results:
                context = "\n".join([f"- {result['content']}" for result in context_results])
            
            if self.gemini_client and context:
                try:
                    prompt = f"""Based on the following context, answer the user's question:

Context:
{context}

Question: {query}

Please provide a helpful, accurate response based on the context provided."""

                    response = self.gemini_client.generate_content(prompt)
                    return response.text
                    
                except Exception as e:
                    print(f"⚠️ Gemini generation failed: {e}")
            
            # Fallback response
            if context_results:
                return f"Based on the available information: {context_results[0]['content']}\n\nThis relates to your question about: {query}"
            else:
                return f"I understand you're asking about: {query}. Let me help you with that based on my knowledge."
            
        except Exception as e:
            print(f"⚠️ Response generation failed: {e}")
            return "I'm having trouble generating a response right now. Could you please rephrase your question?"
    
    def generate_response(self, query):
        """Main response generation method"""
        try:
            # Search for relevant context
            context_results = self.search_similar_content(query)
            
            # Generate response with context
            response = self.generate_contextual_response(query, context_results)
            
            # Store in conversation history
            self.conversation_history.append({
                'user_query': query,
                'ai_response': response,
                'context_used': len(context_results),
                'timestamp': datetime.now().isoformat()
            })
            
            return {
                'response': response,
                'context_results': context_results,
                'success': True
            }
            
        except Exception as e:
            print(f"⚠️ Response generation failed: {e}")
            return {
                'response': "I apologize, but I'm having technical difficulties. Please try again.",
                'context_results': [],
                'success': False,
                'error': str(e)
            }
    
    def chat(self, message):
        """Simple chat interface"""
        print(f"\n❓ User: {message}")
        result = self.generate_response(message)
        print(f"🤖 Assistant: {result['response']}")
        
        if result['context_results']:
            print(f"📚 Used {len(result['context_results'])} context items")
        
        return result
    
    def get_interview_summary(self):
        """Get interview summary"""
        try:
            if not self.conversation_history:
                return {
                    'total_exchanges': 0,
                    'summary': 'No conversation history available.',
                    'topics_covered': [],
                    'timestamp': datetime.now().isoformat()
                }
            
            # Simple summary
            topics = set()
            for exchange in self.conversation_history:
                # Extract topics from context used
                if 'context_results' in exchange:
                    for context in exchange.get('context_results', []):
                        topics.add(context.get('topic', 'general'))
            
            return {
                'total_exchanges': len(self.conversation_history),
                'summary': f'Interview covered {len(topics)} main topics with {len(self.conversation_history)} exchanges.',
                'topics_covered': list(topics),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'total_exchanges': 0,
                'summary': f'Summary generation failed: {e}',
                'topics_covered': [],
                'timestamp': datetime.now().isoformat()
            }
    
    def add_knowledge(self, content, topic="general", keywords=None):
        """Add new knowledge to the base"""
        try:
            if keywords is None:
                keywords = content.lower().split()[:5]  # First 5 words as keywords
            
            self.knowledge_base.append({
                'content': content,
                'topic': topic,
                'keywords': keywords,
                'added': datetime.now().isoformat()
            })
            
            return {
                'success': True,
                'message': f'Added knowledge item: {topic}'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_stats(self):
        """Get chatbot statistics"""
        return {
            'conversation_exchanges': len(self.conversation_history),
            'knowledge_base_size': len(self.knowledge_base),
            'gemini_available': self.gemini_client is not None,
            'qdrant_available': self.qdrant_client is not None
        }

def test_clean_rag():
    """Test the clean RAG chatbot"""
    print("🧪 Testing Clean RAG Chatbot")
    print("=" * 40)
    
    chatbot = MultiModalRAGChatbot()
    
    test_queries = [
        "What is this system about?",
        "How to prepare for coding interviews?",
        "Tell me about system design",
        "What is RAG?",
        "How do vector databases work?"
    ]
    
    for query in test_queries:
        result = chatbot.chat(query)
        print("-" * 40)
    
    # Show stats
    stats = chatbot.get_stats()
    print(f"\n📊 Stats: {stats}")
    
    return True

if __name__ == "__main__":
    test_clean_rag()