import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import re

# Add current directory to Python path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# Load environment variables
def load_environment():
    """Load environment variables with multiple fallback paths"""
    try:
        from dotenv import load_dotenv
        
        env_paths = [
            Path(".env"),
            Path("backend/.env"),
            Path("../.env"),
            current_dir / ".env"
        ]
        
        for env_path in env_paths:
            if env_path.exists():
                load_dotenv(env_path)
                print(f"✅ Environment loaded from: {env_path}")
                return True
        
        print("⚠️ No .env file found")
        return False
    except ImportError:
        print("⚠️ python-dotenv not installed")
        return False

# Load environment on import
load_environment()

# Import dependencies with error handling
try:
    import google.generativeai as genai
    HAS_GEMINI = True
    print("✅ Google Gemini imported")
except ImportError:
    print("⚠️ Google Gemini not installed - chatbot will use fallback responses")
    print("💡 Install with: pip install google-generativeai")
    HAS_GEMINI = False

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    HAS_QDRANT = True
    print("✅ Qdrant imported")
except ImportError:
    print("⚠️ Qdrant not installed - using mock database")
    HAS_QDRANT = False

# Import mock Qdrant for fallback
try:
    from mock_qdrant import get_qdrant_client, get_qdrant_models
    HAS_MOCK_QDRANT = True
except ImportError:
    HAS_MOCK_QDRANT = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_EMBEDDINGS = True
    print("✅ Sentence transformers imported")
except ImportError:
    print("⚠️ Sentence transformers not installed - using simple text matching")
    HAS_EMBEDDINGS = False

try:
    from flask import Flask, request, jsonify, render_template_string
    from flask_cors import CORS
    HAS_FLASK = True
    print("✅ Flask imported")
except ImportError:
    print("❌ Flask not installed - web interface disabled")
    HAS_FLASK = False

class RAGChatbot:
    """RAG Chatbot with Google Gemini integration"""
    
    def __init__(self):
        """Initialize RAG Chatbot"""
        print("🤖 Initializing RAG Chatbot with Gemini...")
        
        self.gemini_client = None
        self.qdrant_client = None
        self.embedding_model = None
        self.knowledge_base = []
        
        # Initialize components
        self._init_gemini()
        self._init_qdrant()
        self._init_embeddings()
        self._load_knowledge_base()
        
        print("✅ RAG Chatbot ready!")
    
    def _init_gemini(self):
        """Initialize Google Gemini client"""
        if not HAS_GEMINI:
            print("⚠️ Google Gemini library not installed")
            return
        
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            print("⚠️ Gemini API key not configured")
            return
        
        print(f"🔄 Initializing Gemini with key: {api_key[:15]}...{api_key[-4:]}")
        
        try:
            # Configure Gemini
            genai.configure(api_key=api_key)
            print("✅ Gemini configured successfully")
            
            # Try different model names in order of preference
            model_names = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']
            
            for model_name in model_names:
                try:
                    self.gemini_client = genai.GenerativeModel(model_name)
                    print(f"✅ Gemini model '{model_name}' initialized")
                    
                    # Test the connection
                    print("🔄 Testing Gemini API connection...")
                    test_response = self.gemini_client.generate_content("Hello")
                    print("✅ Gemini initialized and tested successfully")
                    print(f"📝 Test response: {test_response.text[:50]}...")
                    return
                    
                except Exception as model_error:
                    print(f"⚠️ Model '{model_name}' failed: {model_error}")
                    continue
            
            # If all models failed
            print("❌ All Gemini models failed to initialize")
            self.gemini_client = None
            
        except Exception as e:
            print(f"❌ Gemini initialization failed: {e}")
            print("💡 Possible solutions:")
            print("   1. Check your internet connection")
            print("   2. Verify your Gemini API key is valid")
            print("   3. Try: pip install google-generativeai --upgrade")
            self.gemini_client = None
    
    def _init_qdrant(self):
        """Initialize Qdrant client with cloud and local support"""
        # First try cloud Qdrant if API key is provided
        qdrant_api_key = os.getenv('QDRANT_API_KEY')
        if qdrant_api_key and HAS_QDRANT:
            try:
                # Try Qdrant Cloud with your specific cluster URL
                self.qdrant_client = QdrantClient(
                    url="https://35f8ce34-d42f-4d11-a6f8-376f05d9b152.us-east-1-1.aws.cloud.qdrant.io:6333",
                    api_key=qdrant_api_key,
                )
                # Test connection
                collections = self.qdrant_client.get_collections()
                self.is_mock_qdrant = False
                print("✅ Qdrant Cloud connected successfully")
                print(f"📊 Found {len(collections.collections)} existing collections")
                return
            except Exception as e:
                print(f"⚠️ Qdrant Cloud failed: {e}")
        
        # Try local Qdrant
        if HAS_QDRANT:
            try:
                host = os.getenv('QDRANT_HOST', 'localhost')
                port = int(os.getenv('QDRANT_PORT', 6333))
                
                self.qdrant_client = QdrantClient(host=host, port=port)
                self.qdrant_client.get_collections()
                self.is_mock_qdrant = False
                print("✅ Local Qdrant connected")
                return
            except Exception as e:
                print(f"⚠️ Local Qdrant failed: {e}")
        
        # Try mock Qdrant
        if HAS_MOCK_QDRANT:
            try:
                self.qdrant_client, self.is_mock_qdrant = get_qdrant_client()
                print("✅ Mock Qdrant initialized")
                return
            except Exception as e:
                print(f"⚠️ Mock Qdrant failed: {e}")
        
        # No Qdrant available
        self.qdrant_client = None
        self.is_mock_qdrant = False
        print("❌ No Qdrant client available")
    
    def _init_embeddings(self):
        """Initialize embedding model"""
        if not HAS_EMBEDDINGS:
            return
        
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Embeddings loaded")
        except Exception as e:
            print(f"⚠️ Embeddings failed: {e}")
    
    def _load_knowledge_base(self):
        """Load comprehensive knowledge base"""
        self.knowledge_base = [
            {
                "text": "This is a RAG (Retrieval Augmented Generation) chatbot designed for hackathon interviews and technical assistance. It combines information retrieval with AI generation to provide contextual responses.",
                "category": "system",
                "keywords": ["rag", "chatbot", "interview", "hackathon", "retrieval", "generation"]
            },
            {
                "text": "I can help with coding questions, technical interviews, algorithm explanations, system design discussions, and hackathon preparation strategies.",
                "category": "capabilities",
                "keywords": ["coding", "technical", "interviews", "system design", "algorithms", "hackathon"]
            },
            {
                "text": "Common technical interview topics include data structures (arrays, linked lists, trees, graphs), algorithms (sorting, searching, dynamic programming), and system design principles.",
                "category": "interview_prep",
                "keywords": ["data structures", "algorithms", "system design", "arrays", "trees", "graphs", "sorting"]
            },
            {
                "text": "For coding interviews, focus on problem-solving approach: understand the problem, think of edge cases, start with a brute force solution, then optimize. Always explain your thought process.",
                "category": "interview_tips",
                "keywords": ["coding", "technical", "interviews", "problem solving", "algorithms", "optimization"]
            },
            {
                "text": "System design interviews evaluate your ability to design large-scale distributed systems. Key concepts include scalability, load balancing, database design, caching, and microservices.",
                "category": "system_design",
                "keywords": ["system design", "scalability", "load balancing", "database", "caching", "microservices"]
            },
            {
                "text": "Common behavioral interview questions include: Tell me about a time you faced a challenge, describe a situation where you worked with a difficult team member, give an example of learning something new quickly.",
                "category": "behavioral_questions",
                "keywords": ["behavioral", "behavioral questions", "interview questions", "challenges", "teamwork", "learning"]
            },
            {
                "text": "STAR method for behavioral interviews: Situation (set context), Task (describe what needed to be done), Action (explain what you did), Result (share the outcome). This framework helps structure responses effectively.",
                "category": "interview_techniques",
                "keywords": ["star method", "behavioral", "situation", "task", "action", "result", "interview technique"]
            }
        ]
        print(f"📚 Loaded {len(self.knowledge_base)} knowledge entries")
        
        # Populate Qdrant with knowledge base if available
        self._populate_qdrant_knowledge()
    
    def _populate_qdrant_knowledge(self):
        """Populate Qdrant with knowledge base data"""
        if not self.qdrant_client or not self.embedding_model:
            return
        
        try:
            collection_name = "interview_knowledge"
            
            # Check if collection exists
            try:
                collections = self.qdrant_client.get_collections()
                collection_names = [c.name for c in collections.collections] if hasattr(collections, 'collections') else []
            except:
                collection_names = []
            
            # Create collection if it doesn't exist
            if collection_name not in collection_names:
                if HAS_MOCK_QDRANT and hasattr(self, 'is_mock_qdrant') and self.is_mock_qdrant:
                    # Mock Qdrant
                    from mock_qdrant import MockModels
                    models = MockModels()
                    vector_config = models.VectorParams(size=384, distance=models.Distance.COSINE)
                else:
                    # Real Qdrant
                    from qdrant_client.http import models
                    vector_config = models.VectorParams(size=384, distance=models.Distance.COSINE)
                
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=vector_config
                )
                print(f"✅ Created Qdrant collection: {collection_name}")
            
            # Add knowledge base entries
            points = []
            for i, item in enumerate(self.knowledge_base):
                try:
                    # Generate embedding
                    embedding = self.embedding_model.encode(item["text"]).tolist()
                    
                    if HAS_MOCK_QDRANT and hasattr(self, 'is_mock_qdrant') and self.is_mock_qdrant:
                        # Mock Qdrant format
                        from mock_qdrant import MockPointStruct
                        point = MockPointStruct(
                            id=i,
                            vector=embedding,
                            payload={
                                "text": item["text"],
                                "category": item.get("category", ""),
                                "keywords": item.get("keywords", [])
                            }
                        )
                    else:
                        # Real Qdrant format
                        from qdrant_client.http import models
                        point = models.PointStruct(
                            id=i,
                            vector=embedding,
                            payload={
                                "text": item["text"],
                                "category": item.get("category", ""),
                                "keywords": item.get("keywords", [])
                            }
                        )
                    points.append(point)
                except Exception as e:
                    print(f"⚠️ Failed to create embedding for item {i}: {e}")
            
            # Upsert points
            if points:
                self.qdrant_client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                print(f"✅ Added {len(points)} knowledge entries to Qdrant")
        
        except Exception as e:
            print(f"⚠️ Failed to populate Qdrant: {e}")
    
    def search_knowledge(self, query: str, limit: int = 3) -> List[Dict]:
        """Enhanced knowledge search with vector similarity"""
        if not query.strip():
            return []
        
        # Try vector search first if Qdrant and embeddings available
        if self.qdrant_client and self.embedding_model:
            try:
                # Generate query embedding
                query_embedding = self.embedding_model.encode(query).tolist()
                
                # Search in Qdrant
                results = self.qdrant_client.search(
                    collection_name="interview_knowledge",
                    query_vector=query_embedding,
                    limit=limit
                )
                
                # Format results
                vector_results = []
                for result in results:
                    vector_results.append({
                        "text": result.payload.get("text", ""),
                        "score": float(result.score * 10),  # Scale up for consistency
                        "category": result.payload.get("category", "unknown"),
                        "relevance": float(result.score)
                    })
                
                if vector_results:
                    print(f"🔍 Vector search found {len(vector_results)} results")
                    return vector_results
                    
            except Exception as e:
                print(f"⚠️ Vector search failed: {e}")
        
        # Fallback to keyword search
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Remove stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        query_words = query_words - stop_words
        
        scored_results = []
        
        for item in self.knowledge_base:
            score = 0
            text_lower = item["text"].lower()
            keywords = item.get("keywords", [])
            
            # Keyword matches
            for keyword in keywords:
                if keyword.lower() in query_lower:
                    score += 5
            
            # Word matches
            for word in query_words:
                if len(word) > 2 and word in text_lower:
                    score += 2
            
            if score > 0:
                scored_results.append({
                    "text": item["text"],
                    "score": score,
                    "category": item.get("category", "unknown"),
                    "relevance": min(1.0, score / 10)
                })
        
        # Sort by score
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        print(f"🔍 Keyword search found {len(scored_results)} results")
        return scored_results[:limit]
    
    def generate_response(self, question: str, context: List[Dict] = None) -> str:
        """Generate response using RAG approach"""
        if not question.strip():
            return "Please ask a question!"
        
        if context is None:
            context = self.search_knowledge(question)
        
        # Try Gemini first
        if self.gemini_client:
            try:
                response = self._generate_gemini_response(question, context)
                if not response.startswith("[Gemini Error:"):
                    if context:
                        response += f"\n\n[RAG: Used {len(context)} context sources]"
                    return response
                else:
                    print("⚠️ Gemini failed, using fallback")
            except Exception as e:
                print(f"⚠️ Gemini exception: {e}")
        
        # Fallback response
        response = self._generate_enhanced_fallback(question, context)
        
        if context:
            response += f"\n\n[Local RAG: Used {len(context)} context sources]"
        else:
            response += "\n\n[No relevant context found in knowledge base]"
        
        return response
    
    def _generate_gemini_response(self, question: str, context: List[Dict]) -> str:
        """Generate response using Google Gemini"""
        # Prepare context
        context_text = ""
        if context:
            context_text = "\n".join([f"- {item['text']}" for item in context])
        else:
            context_text = "No specific context found in knowledge base."
        
        # Build prompt
        prompt = f"""You are a helpful AI assistant for interview preparation and technical guidance.
Use the following context to answer the user's question accurately and helpfully.

CONTEXT FROM KNOWLEDGE BASE:
{context_text}

USER QUESTION: {question}

Instructions:
1. Use the context information to provide accurate answers
2. Be conversational and supportive
3. Focus on technical interviews, coding, and hackathon-related topics
4. Keep responses informative but concise
5. If the context doesn't fully answer the question, provide general helpful guidance

RESPONSE:"""

        try:
            response = self.gemini_client.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            print(f"❌ Gemini API call failed: {e}")
            return f"[Gemini Error: {str(e)}] " + self._generate_enhanced_fallback(question, context)
    
    def _generate_enhanced_fallback(self, question: str, context: List[Dict]) -> str:
        """Generate fallback response using retrieved context"""
        if context:
            best_match = context[0]
            response = f"{best_match['text']}"
            
            # Add additional context if relevant
            if len(context) > 1 and context[1]['score'] >= context[0]['score'] * 0.7:
                response += f" {context[1]['text']}"
            
            # Add category-specific advice
            category = best_match.get('category', '')
            if 'interview' in category:
                response += " Remember to practice with real examples and explain your reasoning clearly."
            elif 'system' in category:
                response += " Consider scalability, reliability, and cost-effectiveness in your design."
            
            return response
        
        return f"I don't have specific information about '{question}' in my knowledge base. Could you ask about technical interviews, coding practices, or system design?"
    
    def chat(self, message: str) -> Dict:
        """Main chat interface"""
        start_time = datetime.now()
        
        backend_status = []
        if self.gemini_client:
            backend_status.append("Gemini: available")
        else:
            backend_status.append("Gemini: NOT available")
        if self.qdrant_client:
            qdrant_type = "Mock Qdrant" if getattr(self, 'is_mock_qdrant', False) else "Real Qdrant"
            backend_status.append(f"Qdrant: available ({qdrant_type})")
        else:
            backend_status.append("Qdrant: NOT available")
        
        try:
            # Search for context
            context = self.search_knowledge(message)
            debug_info = {
                "query": message,
                "context_found": len(context),
                "context_items": [{"text": c["text"][:100] + "...", "score": c["score"]} for c in context[:3]],
                "gemini_available": self.gemini_client is not None,
                "qdrant_available": self.qdrant_client is not None,
                "backend_status": backend_status
            }
            
            response = self.generate_response(message, context)
            
            return {
                "question": message,
                "response": response,
                "context_used": len(context),
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "timestamp": datetime.now().isoformat(),
                "status": "success",
                "debug": debug_info
            }
        except Exception as e:
            return {
                "question": message,
                "response": f"Sorry, I encountered an error: {str(e)}",
                "status": "error",
                "error": str(e),
                "debug": {"error_details": str(e), "backend_status": backend_status}
            }

def create_web_interface():
    """Create Flask web interface"""
    if not HAS_FLASK:
        print("❌ Flask not available")
        return None
    
    app = Flask(__name__)
    CORS(app)
    
    # Initialize chatbot
    chatbot = RAGChatbot()
    
    @app.route('/')
    def home():
        return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>RAG Chatbot - Interview Assistant</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .chat-container { border: 1px solid #ddd; border-radius: 10px; padding: 20px; height: 400px; overflow-y: auto; }
        .message { margin: 10px 0; padding: 10px; border-radius: 5px; }
        .user-message { background: #007bff; color: white; text-align: right; }
        .bot-message { background: #f5f5f5; }
        .input-area { margin: 20px 0; display: flex; gap: 10px; }
        .input-area input { flex: 1; padding: 10px; }
        .input-area button { padding: 10px 20px; background: #007bff; color: white; border: none; cursor: pointer; }
    </style>
</head>
<body>
    <h1>🤖 RAG Chatbot with Gemini</h1>
    <div class="chat-container" id="chat-messages">
        <div class="message bot-message">Hello! I'm your RAG chatbot assistant powered by Google Gemini. Ask me anything about interviews, coding, or system design!</div>
    </div>
    <div class="input-area">
        <input type="text" id="user-input" placeholder="Ask me anything..." onkeypress="if(event.key==='Enter') sendMessage()">
        <button onclick="sendMessage()">Send</button>
    </div>
    
    <script>
        function sendMessage() {
            const input = document.getElementById('user-input');
            const message = input.value.trim();
            if (!message) return;
            
            addMessage(message, 'user-message');
            input.value = '';
            
            fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: message })
            })
            .then(response => response.json())
            .then(data => {
                addMessage(data.response, 'bot-message');
            })
            .catch(error => {
                addMessage('Sorry, there was an error.', 'bot-message');
            });
        }
        
        function addMessage(text, className) {
            const chatMessages = document.getElementById('chat-messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message ' + className;
            messageDiv.textContent = text;
            chatMessages.appendChild(messageDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    </script>
</body>
</html>
        ''')
    
    @app.route('/chat', methods=['POST'])
    def chat():
        try:
            data = request.json
            message = data.get('message', '')
            result = chatbot.chat(message)
            return jsonify({'response': result['response']})
        except Exception as e:
            return jsonify({'response': f'Error: {str(e)}'}), 500
    
    return app

# Main execution
if __name__ == "__main__":
    print("🚀 Starting RAG Chatbot System with Gemini...")
    
    # Test basic functionality
    print("\n🧪 Testing basic functionality...")
    try:
        chatbot = RAGChatbot()
        
        # Test Gemini
        print("\n🔍 Testing Gemini integration...")
        if chatbot.gemini_client:
            print("✅ Gemini client initialized")
            try:
                test_result = chatbot._generate_gemini_response("What is coding?", [])
                if not test_result.startswith("[Gemini Error:"):
                    print("✅ Gemini API call successful")
                else:
                    print("❌ Gemini API call failed")
            except Exception as e:
                print(f"❌ Gemini test failed: {e}")
        else:
            print("❌ Gemini not available")
        
        # Test knowledge search
        print("\n🔍 Testing knowledge search...")
        context = chatbot.search_knowledge("coding interview")
        print(f"✅ Found {len(context)} relevant contexts")
        
        # Test full chat
        print("\n🔍 Testing full chat...")
        test_result = chatbot.chat("What is this system about?")
        print(f"✅ Chat test successful")
        print(f"📝 Response preview: {test_result['response'][:150]}...")
        
        # Show system status
        print("\n📊 System Status:")
        print(f"   Gemini: {'✅ Available' if chatbot.gemini_client else '❌ Not available'}")
        if chatbot.qdrant_client:
            qdrant_type = "Mock Qdrant" if getattr(chatbot, 'is_mock_qdrant', False) else "Real Qdrant"
            print(f"   Qdrant: ✅ Available ({qdrant_type})")
        else:
            print("   Qdrant: ❌ Not available")
        print(f"   Knowledge Base: ✅ {len(chatbot.knowledge_base)} entries loaded")
        
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # Start web interface
    if HAS_FLASK:
        print("\n🌐 Starting web interface...")
        app = create_web_interface()
        if app:
            print("🎯 RAG Chatbot running at: http://localhost:5000")
            print("📋 Press Ctrl+C to stop")
            try:
                app.run(host='0.0.0.0', port=5000, debug=False)
            except KeyboardInterrupt:
                print("\n👋 Chatbot stopped")
        else:
            print("❌ Failed to create web interface")
    else:
        print("\n💬 Web interface not available. Running console mode...")
        
        # Console chat mode
        print("Type 'quit' to exit")
        print("Type 'debug' to toggle debug mode")
        debug_mode = False
        
        while True:
            try:
                user_input = input("\n🧑 You: ").strip()
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == 'debug':
                    debug_mode = not debug_mode
                    print(f"🔧 Debug mode: {'ON' if debug_mode else 'OFF'}")
                    continue
                
                if user_input:
                    result = chatbot.chat(user_input)
                    
                    if debug_mode and 'debug' in result:
                        debug = result['debug']
                        print(f"\n🔍 DEBUG INFO:")
                        print(f"   - Context found: {debug['context_found']} items")
                        print(f"   - Gemini available: {debug['gemini_available']}")
                        print(f"   - Context used:")
                        for i, item in enumerate(debug['context_items']):
                            print(f"     {i+1}. Score: {item['score']} - {item['text']}")
                    
                    print(f"\n🤖 Bot: {result['response']}")
                    
                    if debug_mode:
                        print(f"📊 Processing time: {result.get('processing_time', 0):.3f}s")
                        
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break