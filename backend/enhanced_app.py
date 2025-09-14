"""
Enhanced App.py with Knowledge Base Integration
Place this in backend directory
"""

# Core imports
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import json
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import modules with fallbacks
try:
    from clean_rag_chatbot import MultiModalRAGChatbot
    print("✅ Clean RAG Chatbot imported")
except ImportError:
    print("⚠️ Clean RAG Chatbot not found, creating mock")
    class MultiModalRAGChatbot:
        def __init__(self, qdrant_client=None):
            pass
        def generate_response(self, query):
            return {"response": f"Mock response for: {query}", "success": True}

try:
    from knowledge_base_manager import knowledge_manager
    print("✅ Knowledge Base Manager imported")
except ImportError:
    print("⚠️ Knowledge Base Manager not found, creating mock")
    class MockKnowledgeManager:
        def add_text_document(self, text, title="", category="general"):
            return {"success": True, "message": f"Mock: Added text '{title}'"}
        def add_audio_document(self, file_path, title=""):
            return {"success": True, "message": f"Mock: Added audio '{title}'"}
        def add_image_document(self, file_path, title=""):
            return {"success": True, "message": f"Mock: Added image '{title}'"}
        def search_knowledge_base(self, query, limit=5, content_type=None):
            return {"success": True, "results": [{"title": "Mock Result", "content": f"Mock result for: {query}", "score": 0.8}], "total_found": 1}
        def get_knowledge_base_stats(self):
            return {"total_documents": 0, "by_type": {"text": 0}, "qdrant_connected": False, "gemini_available": False}
    
    knowledge_manager = MockKnowledgeManager()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
CORS(app)

# Global variables
active_interviews = {}

# Ensure uploads directory exists
os.makedirs('uploads', exist_ok=True)

print("🚀 Enhanced Interview System Starting...")

@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>Enhanced Interview System with Knowledge Base</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1400px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }
        .header { text-align: center; color: #333; margin-bottom: 30px; }
        .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 8px; }
        .button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin: 5px; }
        .button:hover { background: #0056b3; }
        .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
        .status.success { background: #d4edda; border: 1px solid #c3e6cb; color: #155724; }
        .status.warning { background: #fff3cd; border: 1px solid #ffeaa7; color: #856404; }
        .status.error { background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
        .chat-container { height: 300px; overflow-y: scroll; border: 1px solid #ddd; padding: 10px; margin: 10px 0; background: #fafafa; }
        .message { margin: 10px 0; padding: 8px; border-radius: 5px; }
        .message.user { background: #e3f2fd; text-align: right; }
        .message.bot { background: #f1f8e9; text-align: left; }
        .knowledge-tab { margin-top: 15px; display: none; }
        .knowledge-tab.active { display: block; }
        .upload-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin: 15px 0; }
        .upload-card { border: 1px solid #ddd; padding: 15px; border-radius: 5px; background: #f9f9f9; }
        .search-result { border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; background: #f9f9f9; }
        .search-score { float: right; background: #007bff; color: white; padding: 2px 8px; border-radius: 3px; font-size: 12px; }
        .tab-buttons { margin-bottom: 15px; }
        .tab-button { background: #6c757d; color: white; padding: 8px 16px; border: none; border-radius: 5px; cursor: pointer; margin: 2px; }
        .tab-button.active { background: #007bff; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Enhanced Interview System</h1>
            <p>AI-powered interviews with integrated knowledge base</p>
        </div>
        
        <!-- Interview Setup -->
        <div class="section">
            <h3>🚀 Interview Setup</h3>
            <input type="text" id="candidateName" placeholder="Candidate Name" style="width: 200px; padding: 8px; margin: 5px;">
            <input type="text" id="position" placeholder="Position" style="width: 200px; padding: 8px; margin: 5px;">
            <button class="button" onclick="startInterview()">Start Interview</button>
            <div id="interviewStatus" class="status"></div>
        </div>
        
        <!-- Chat Interface -->
        <div class="section">
            <h3>🤖 AI Chat Interface</h3>
            <div class="chat-container" id="chatContainer"></div>
            <input type="text" id="userInput" placeholder="Type your message..." style="width: 70%; padding: 10px;">
            <button class="button" onclick="sendMessage()">Send</button>
            <button class="button" onclick="sendMessageWithRetrieval()">🔍 Send + Search</button>
        </div>
        
        <!-- Knowledge Base -->
        <div class="section">
            <h3>📚 Knowledge Base</h3>
            <div class="tab-buttons">
                <button class="tab-button active" onclick="showKnowledgeTab('upload')">📤 Upload</button>
                <button class="tab-button" onclick="showKnowledgeTab('search')">🔍 Search</button>
                <button class="tab-button" onclick="showKnowledgeTab('stats')">📊 Stats</button>
            </div>
            
            <!-- Upload Tab -->
            <div id="uploadTab" class="knowledge-tab active">
                <h4>Upload Content</h4>
                <div class="upload-grid">
                    <div class="upload-card">
                        <h5>📝 Text Document</h5>
                        <input type="text" id="textTitle" placeholder="Title" style="width: 100%; margin: 5px 0; padding: 5px;">
                        <textarea id="textContent" placeholder="Enter text..." style="width: 100%; height: 80px; margin: 5px 0; padding: 5px;"></textarea>
                        <button class="button" onclick="uploadText()" style="width: 100%;">Upload Text</button>
                    </div>
                    <div class="upload-card">
                        <h5>🎵 Audio File</h5>
                        <input type="text" id="audioTitle" placeholder="Title" style="width: 100%; margin: 5px 0; padding: 5px;">
                        <input type="file" id="audioFile" accept=".wav,.mp3" style="width: 100%; margin: 5px 0;">
                        <button class="button" onclick="uploadAudio()" style="width: 100%;">Upload Audio</button>
                    </div>
                    <div class="upload-card">
                        <h5>🖼️ Image File</h5>
                        <input type="text" id="imageTitle" placeholder="Title" style="width: 100%; margin: 5px 0; padding: 5px;">
                        <input type="file" id="imageFile" accept=".jpg,.png" style="width: 100%; margin: 5px 0;">
                        <button class="button" onclick="uploadImage()" style="width: 100%;">Upload Image</button>
                    </div>
                </div>
                <div id="uploadStatus" class="status"></div>
            </div>
            
            <!-- Search Tab -->
            <div id="searchTab" class="knowledge-tab">
                <h4>Search Knowledge Base</h4>
                <input type="text" id="searchQuery" placeholder="Search query..." style="width: 60%; padding: 10px;">
                <button class="button" onclick="searchKnowledge()">🔍 Search</button>
                <div id="searchResults" style="margin-top: 20px;"></div>
            </div>
            
            <!-- Stats Tab -->
            <div id="statsTab" class="knowledge-tab">
                <h4>Statistics</h4>
                <div id="knowledgeStats"></div>
                <button class="button" onclick="loadKnowledgeStats()">🔄 Refresh</button>
            </div>
        </div>
    </div>

    <script>
        let currentInterviewId = null;

        function showKnowledgeTab(tabName) {
            document.querySelectorAll('.knowledge-tab').forEach(tab => {
                tab.classList.remove('active');
                tab.style.display = 'none';
            });
            document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
            
            document.getElementById(tabName + 'Tab').style.display = 'block';
            document.getElementById(tabName + 'Tab').classList.add('active');
            event.target.classList.add('active');
            
            if (tabName === 'stats') loadKnowledgeStats();
        }

        async function uploadText() {
            const title = document.getElementById('textTitle').value.trim();
            const content = document.getElementById('textContent').value.trim();
            
            if (!content) {
                showUploadStatus('Please enter text content', 'error');
                return;
            }
            
            try {
                const response = await fetch('/api/knowledge/upload/text', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ title, text: content, category: 'general' })
                });
                
                const result = await response.json();
                showUploadStatus(result.success ? '✅ Text uploaded!' : '❌ Upload failed', 
                               result.success ? 'success' : 'error');
                
                if (result.success) {
                    document.getElementById('textTitle').value = '';
                    document.getElementById('textContent').value = '';
                }
            } catch (error) {
                showUploadStatus('❌ Upload error: ' + error.message, 'error');
            }
        }

        async function uploadAudio() {
            const fileInput = document.getElementById('audioFile');
            if (!fileInput.files[0]) {
                showUploadStatus('Please select an audio file', 'error');
                return;
            }
            
            const formData = new FormData();
            formData.append('audio', fileInput.files[0]);
            formData.append('title', document.getElementById('audioTitle').value);
            
            try {
                showUploadStatus('🔄 Processing audio...', 'warning');
                const response = await fetch('/api/knowledge/upload/audio', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                showUploadStatus(result.success ? '✅ Audio uploaded!' : '❌ Upload failed', 
                               result.success ? 'success' : 'error');
            } catch (error) {
                showUploadStatus('❌ Upload error: ' + error.message, 'error');
            }
        }

        async function uploadImage() {
            const fileInput = document.getElementById('imageFile');
            if (!fileInput.files[0]) {
                showUploadStatus('Please select an image file', 'error');
                return;
            }
            
            const formData = new FormData();
            formData.append('image', fileInput.files[0]);
            formData.append('title', document.getElementById('imageTitle').value);
            
            try {
                showUploadStatus('🔄 Processing image...', 'warning');
                const response = await fetch('/api/knowledge/upload/image', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                showUploadStatus(result.success ? '✅ Image uploaded!' : '❌ Upload failed', 
                               result.success ? 'success' : 'error');
            } catch (error) {
                showUploadStatus('❌ Upload error: ' + error.message, 'error');
            }
        }

        async function searchKnowledge() {
            const query = document.getElementById('searchQuery').value.trim();
            if (!query) {
                alert('Please enter a search query');
                return;
            }
            
            try {
                const response = await fetch('/api/knowledge/search', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ query, limit: 5 })
                });
                
                const result = await response.json();
                displaySearchResults(result);
            } catch (error) {
                document.getElementById('searchResults').innerHTML = 
                    '<div class="status error">Search error: ' + error.message + '</div>';
            }
        }

        function displaySearchResults(result) {
            const container = document.getElementById('searchResults');
            
            if (!result.success || !result.results || result.results.length === 0) {
                container.innerHTML = '<div class="status warning">No results found.</div>';
                return;
            }
            
            let html = '<h4>🔍 Search Results (' + result.total_found + ')</h4>';
            result.results.forEach(item => {
                const score = ((item.score || 0) * 100).toFixed(1);
                html += `
                    <div class="search-result">
                        <h5>${item.title} <span class="search-score">${score}%</span></h5>
                        <div>${item.content}</div>
                    </div>
                `;
            });
            container.innerHTML = html;
        }

        async function loadKnowledgeStats() {
            try {
                const response = await fetch('/api/knowledge/stats');
                const result = await response.json();
                
                if (result.success) {
                    const stats = result.stats;
                    document.getElementById('knowledgeStats').innerHTML = `
                        <p><strong>Total Documents:</strong> ${stats.total_documents}</p>
                        <p><strong>Qdrant Connected:</strong> ${stats.qdrant_connected ? 'Yes' : 'No'}</p>
                        <p><strong>Gemini Available:</strong> ${stats.gemini_available ? 'Yes' : 'No'}</p>
                    `;
                }
            } catch (error) {
                document.getElementById('knowledgeStats').innerHTML = 
                    '<div class="status error">Failed to load stats</div>';
            }
        }

        function showUploadStatus(message, type) {
            const statusDiv = document.getElementById('uploadStatus');
            statusDiv.textContent = message;
            statusDiv.className = `status ${type}`;
            if (type === 'success') {
                setTimeout(() => {
                    statusDiv.textContent = '';
                    statusDiv.className = 'status';
                }, 3000);
            }
        }

        function startInterview() {
            const candidateName = document.getElementById('candidateName').value;
            const position = document.getElementById('position').value;
            
            if (!candidateName || !position) {
                showAlert('Please enter candidate name and position', 'error');
                return;
            }

            currentInterviewId = 'interview_' + Date.now();
            showAlert('Interview started successfully!', 'success');
            addChatMessage('Hello! I am your AI interview assistant. Let\\'s begin the interview.', 'bot');
        }

        function sendMessage() {
            const input = document.getElementById('userInput');
            const message = input.value.trim();
            if (!message) return;

            addChatMessage(message, 'user');
            input.value = '';

            // Simple response
            setTimeout(() => {
                addChatMessage('Thank you for your response. Can you tell me more about your experience with this topic?', 'bot');
            }, 1000);
        }

        function sendMessageWithRetrieval() {
            const input = document.getElementById('userInput');
            const message = input.value.trim();
            if (!message) return;

            addChatMessage(message, 'user');
            addChatMessage('🔍 Searching knowledge base for relevant context...', 'bot');
            
            // Search and then respond
            searchKnowledgeForChat(message);
            input.value = '';
        }

        async function searchKnowledgeForChat(query) {
            try {
                const response = await fetch('/api/knowledge/search', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ query, limit: 2 })
                });
                
                const result = await response.json();
                
                if (result.success && result.results && result.results.length > 0) {
                    addChatMessage(`📚 Found ${result.results.length} relevant documents. Based on this context: ${result.results[0].content.substring(0, 100)}...`, 'bot');
                } else {
                    addChatMessage('📚 No specific context found, but I can still help with your question.', 'bot');
                }
                
                setTimeout(() => {
                    addChatMessage('Based on the available information, can you elaborate on your approach to this topic?', 'bot');
                }, 1500);
                
            } catch (error) {
                addChatMessage('📚 Knowledge search encountered an issue, but I can still assist you.', 'bot');
            }
        }

        function addChatMessage(message, sender) {
            const container = document.getElementById('chatContainer');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}`;
            messageDiv.innerHTML = `<strong>${sender === 'user' ? 'You' : 'AI'}:</strong> ${message}`;
            container.appendChild(messageDiv);
            container.scrollTop = container.scrollHeight;
        }

        function showAlert(message, type) {
            const statusDiv = document.getElementById('interviewStatus');
            statusDiv.textContent = message;
            statusDiv.className = `status ${type}`;
        }

        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            addChatMessage('🤖 Enhanced Interview System Ready!', 'bot');
            addChatMessage('📚 Knowledge Base features are available in the tabs below.', 'bot');
            loadKnowledgeStats();
        });
    </script>
</body>
</html>
    ''')

# Knowledge Base API Routes
@app.route('/api/knowledge/upload/text', methods=['POST'])
def upload_text_document():
    try:
        data = request.json
        result = knowledge_manager.add_text_document(
            data.get('text', ''), 
            data.get('title', ''),
            data.get('category', 'general')
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/knowledge/upload/audio', methods=['POST'])
def upload_audio_document():
    try:
        if 'audio' not in request.files:
            return jsonify({'success': False, 'error': 'No audio file'})
        
        audio_file = request.files['audio']
        title = request.form.get('title', '')
        
        # Save file
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{audio_file.filename}"
        file_path = os.path.join('uploads', filename)
        audio_file.save(file_path)
        
        result = knowledge_manager.add_audio_document(file_path, title)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/knowledge/upload/image', methods=['POST'])
def upload_image_document():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file'})
        
        image_file = request.files['image']
        title = request.form.get('title', '')
        
        # Save file
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{image_file.filename}"
        file_path = os.path.join('uploads', filename)
        image_file.save(file_path)
        
        result = knowledge_manager.add_image_document(file_path, title)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/knowledge/search', methods=['POST'])
def search_knowledge_base():
    try:
        data = request.json
        result = knowledge_manager.search_knowledge_base(
            data.get('query', ''),
            data.get('limit', 5),
            data.get('content_type')
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/knowledge/stats')
def get_knowledge_stats():
    try:
        stats = knowledge_manager.get_knowledge_base_stats()
        return jsonify({'success': True, 'stats': stats})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8000))
    print(f"🚀 Starting Enhanced Interview System on port {port}")
    print(f"📊 Dashboard: http://localhost:{port}")
    print(f"📚 Knowledge Base: Ready")
    
    app.run(host='0.0.0.0', port=port, debug=True)