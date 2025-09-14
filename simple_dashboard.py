#!/usr/bin/env python3
"""
Simple Working Multi-Modal Chatbot
Avoids problematic dependencies, focuses on what works
"""

import os
import base64
import json
import io
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from datetime import datetime
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment loaded")
except ImportError:
    print("⚠️ python-dotenv not installed")

# Try to import Google Gemini
try:
    import google.generativeai as genai
    HAS_GEMINI = True
    print("✅ Google Gemini available")
except ImportError:
    HAS_GEMINI = False
    print("⚠️ Google Gemini not available")

# Try to import PIL for basic image handling
try:
    from PIL import Image
    HAS_PIL = True
    print("✅ PIL available")
except ImportError:
    HAS_PIL = False
    print("⚠️ PIL not available")

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class SimpleMultiModalBot:
    """Simple multi-modal bot that works with minimal dependencies"""
    
    def __init__(self):
        self.gemini_client = None
        self.conversation_history = []
        self.stats = {
            "total_queries": 0,
            "multimodal_queries": 0
        }
        
        # Initialize Gemini if available
        if HAS_GEMINI:
            api_key = os.getenv('GEMINI_API_KEY')
            if api_key:
                try:
                    genai.configure(api_key=api_key)
                    self.gemini_client = genai.GenerativeModel('gemini-1.5-flash')
                    print("✅ Gemini initialized")
                except Exception as e:
                    print(f"❌ Gemini initialization failed: {e}")
        
        print("🤖 Simple Multi-Modal Bot ready!")
    
    def process_text(self, text):
        """Process text input"""
        if not text.strip():
            return "Please provide some text to process."
        
        if self.gemini_client:
            try:
                response = self.gemini_client.generate_content(text)
                return response.text
            except Exception as e:
                return f"AI processing error: {str(e)}"
        else:
            return f"Echo response: {text} (Gemini not available - this is a fallback response)"
    
    def process_image(self, image_data):
        """Process image data"""
        if not HAS_PIL:
            return "Image processing not available (PIL not installed)"
        
        try:
            # Convert to PIL Image for basic info
            image = Image.open(io.BytesIO(image_data))
            basic_info = f"Image detected: {image.width}x{image.height} pixels, format: {image.format}"
            
            # Try Gemini Vision if available
            if self.gemini_client:
                try:
                    response = self.gemini_client.generate_content([
                        "Describe this image in detail:",
                        image
                    ])
                    return f"{basic_info}\n\nAI Description: {response.text}"
                except Exception as e:
                    return f"{basic_info}\n\nAI analysis failed: {str(e)}"
            else:
                return basic_info
                
        except Exception as e:
            return f"Image processing error: {str(e)}"
    
    def process_document(self, doc_data, filename):
        """Process document data"""
        try:
            if filename.lower().endswith('.txt'):
                text_content = doc_data.decode('utf-8', errors='ignore')
                word_count = len(text_content.split())
                
                summary = f"Document processed: {word_count} words"
                
                if self.gemini_client and len(text_content) < 5000:  # Limit for API
                    try:
                        response = self.gemini_client.generate_content(f"Summarize this text: {text_content[:3000]}")
                        summary += f"\n\nAI Summary: {response.text}"
                    except Exception as e:
                        summary += f"\n\nAI summarization failed: {str(e)}"
                
                return summary
            else:
                return f"Document uploaded: {filename} (Content extraction requires additional libraries)"
                
        except Exception as e:
            return f"Document processing error: {str(e)}"
    
    def process_multimodal(self, text=None, image_data=None, doc_data=None, filename=None):
        """Process multi-modal input"""
        responses = []
        
        self.stats["total_queries"] += 1
        
        if image_data or doc_data:
            self.stats["multimodal_queries"] += 1
        
        # Process text
        if text:
            text_response = self.process_text(text)
            responses.append(f"Text: {text_response}")
        
        # Process image
        if image_data:
            image_response = self.process_image(image_data)
            responses.append(f"Image: {image_response}")
        
        # Process document
        if doc_data and filename:
            doc_response = self.process_document(doc_data, filename)
            responses.append(f"Document: {doc_response}")
        
        if not responses:
            responses.append("No content provided to process.")
        
        # Store in history
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "response": " | ".join(responses)
        })
        
        return " | ".join(responses)

# Initialize bot
bot = SimpleMultiModalBot()

@app.route('/')
def dashboard():
    return render_template_string('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Simple Multi-Modal AI Dashboard</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: 300px 1fr 280px;
            gap: 20px;
            height: calc(100vh - 40px);
        }
        
        .panel {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
        }
        
        .header {
            grid-column: 1 / -1;
            text-align: center;
            padding: 15px;
            margin-bottom: 20px;
        }
        
        .header h1 {
            color: white;
            font-size: 2rem;
            margin-bottom: 5px;
        }
        
        .header p {
            color: rgba(255, 255, 255, 0.9);
            font-size: 1.1rem;
        }
        
        .upload-panel h3, .stats-panel h3 {
            margin-bottom: 15px;
            color: #333;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .upload-item {
            background: #f8f9fa;
            border: 2px dashed #dee2e6;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            cursor: pointer;
            margin-bottom: 15px;
            transition: all 0.3s ease;
        }
        
        .upload-item:hover {
            border-color: #667eea;
            background: #f0f4ff;
        }
        
        .upload-item i {
            font-size: 2rem;
            margin-bottom: 8px;
            display: block;
        }
        
        .upload-item.image i { color: #007bff; }
        .upload-item.document i { color: #28a745; }
        
        .file-input { display: none; }
        
        .chat-area {
            display: flex;
            flex-direction: column;
        }
        
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 10px;
            margin-bottom: 15px;
            min-height: 400px;
        }
        
        .message {
            margin-bottom: 15px;
            display: flex;
            gap: 10px;
        }
        
        .message.user { flex-direction: row-reverse; }
        
        .message-content {
            max-width: 80%;
            padding: 12px 16px;
            border-radius: 15px;
            line-height: 1.4;
        }
        
        .message.user .message-content {
            background: #667eea;
            color: white;
        }
        
        .message.bot .message-content {
            background: white;
            color: #333;
            border: 1px solid #dee2e6;
        }
        
        .input-area {
            display: flex;
            gap: 10px;
        }
        
        .text-input {
            flex: 1;
            padding: 12px;
            border: 2px solid #dee2e6;
            border-radius: 10px;
            font-size: 14px;
            resize: none;
            min-height: 50px;
        }
        
        .text-input:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .send-btn {
            width: 50px;
            height: 50px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            font-size: 16px;
        }
        
        .send-btn:hover { background: #5a6fd8; }
        .send-btn:disabled { background: #ccc; cursor: not-allowed; }
        
        .stats-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-bottom: 20px;
        }
        
        .stat-item {
            text-align: center;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        
        .stat-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: #667eea;
        }
        
        .stat-label {
            font-size: 0.9rem;
            color: #666;
        }
        
        .status-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid #eee;
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #28a745;
        }
        
        .clear-btn {
            width: 100%;
            padding: 10px;
            background: #dc3545;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            margin-top: 15px;
        }
        
        .clear-btn:hover { background: #c82333; }
        
        .processing {
            text-align: center;
            padding: 10px;
            background: #e3f2fd;
            border-radius: 8px;
            margin-bottom: 15px;
            display: none;
        }
        
        @media (max-width: 768px) {
            .container {
                grid-template-columns: 1fr;
                gap: 15px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="panel header">
            <h1><i class="fas fa-brain"></i> Simple Multi-Modal AI</h1>
            <p>Working dashboard with minimal dependencies</p>
        </div>
        
        <div class="panel upload-panel">
            <h3><i class="fas fa-upload"></i> Upload Files</h3>
            
            <div class="upload-item image" onclick="document.getElementById('imageInput').click()">
                <i class="fas fa-image"></i>
                <div><strong>Images</strong></div>
                <small>PNG, JPG, GIF</small>
                <input type="file" id="imageInput" class="file-input" accept=".png,.jpg,.jpeg,.gif" onchange="handleFile(this, 'image')">
            </div>
            
            <div class="upload-item document" onclick="document.getElementById('docInput').click()">
                <i class="fas fa-file-text"></i>
                <div><strong>Documents</strong></div>
                <small>TXT files</small>
                <input type="file" id="docInput" class="file-input" accept=".txt" onchange="handleFile(this, 'document')">
            </div>
            
            <div id="uploadedFiles"></div>
            
            <button class="clear-btn" onclick="clearAll()">
                <i class="fas fa-trash"></i> Clear All
            </button>
        </div>
        
        <div class="panel chat-area">
            <div class="processing" id="processing">
                <i class="fas fa-spinner fa-spin"></i> Processing...
            </div>
            
            <div class="chat-messages" id="messages">
                <div class="message bot">
                    <div class="message-content">
                        Welcome! This is a simple multi-modal AI dashboard that works reliably.<br><br>
                        Features:<br>
                        • 📝 Text chat with AI<br>
                        • 🖼️ Image analysis<br>
                        • 📄 Basic document processing<br><br>
                        Try uploading a file or just chat with me!
                    </div>
                </div>
            </div>
            
            <div class="input-area">
                <textarea id="textInput" class="text-input" placeholder="Type your message here..." rows="2"></textarea>
                <button id="sendBtn" class="send-btn" onclick="sendMessage()">
                    <i class="fas fa-paper-plane"></i>
                </button>
            </div>
        </div>
        
        <div class="panel stats-panel">
            <h3><i class="fas fa-chart-bar"></i> Statistics</h3>
            
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-value" id="totalQueries">0</div>
                    <div class="stat-label">Total</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="multimodalQueries">0</div>
                    <div class="stat-label">Multi-modal</div>
                </div>
            </div>
            
            <h3><i class="fas fa-cogs"></i> System Status</h3>
            <div class="status-item">
                <span>Flask Server</span>
                <span class="status-dot"></span>
            </div>
            <div class="status-item">
                <span>Google Gemini</span>
                <span class="status-dot" id="geminiStatus"></span>
            </div>
            <div class="status-item">
                <span>Image Processing</span>
                <span class="status-dot" id="imageStatus"></span>
            </div>
        </div>
    </div>
    
    <script>
        let uploadedFiles = {};
        
        document.addEventListener('DOMContentLoaded', function() {
            updateStats();
            checkStatus();
            
            document.getElementById('textInput').addEventListener('keypress', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                }
            });
        });
        
        function handleFile(input, type) {
            const file = input.files[0];
            if (file) {
                const fileId = Date.now();
                uploadedFiles[fileId] = { file, type, name: file.name };
                
                const container = document.getElementById('uploadedFiles');
                const div = document.createElement('div');
                div.style.cssText = 'background: #e3f2fd; padding: 8px; border-radius: 5px; margin: 5px 0; font-size: 12px;';
                div.innerHTML = `${file.name} <button onclick="removeFile(${fileId})" style="float: right; background: none; border: none; color: #dc3545;">×</button>`;
                div.id = 'file_' + fileId;
                container.appendChild(div);
                
                input.value = '';
            }
        }
        
        function removeFile(fileId) {
            delete uploadedFiles[fileId];
            document.getElementById('file_' + fileId).remove();
        }
        
        async function sendMessage() {
            const textInput = document.getElementById('textInput');
            const sendBtn = document.getElementById('sendBtn');
            const processing = document.getElementById('processing');
            
            const text = textInput.value.trim();
            const hasFiles = Object.keys(uploadedFiles).length > 0;
            
            if (!text && !hasFiles) return;
            
            // Disable UI
            textInput.disabled = true;
            sendBtn.disabled = true;
            processing.style.display = 'block';
            
            // Add user message
            if (text) addMessage(text, 'user');
            
            // Show uploaded files
            Object.values(uploadedFiles).forEach(file => {
                addMessage(`📎 ${file.name}`, 'user');
            });
            
            try {
                const formData = new FormData();
                if (text) formData.append('text', text);
                
                Object.values(uploadedFiles).forEach(fileInfo => {
                    formData.append(fileInfo.type, fileInfo.file);
                    formData.append('filename', fileInfo.name);
                });
                
                const response = await fetch('/chat', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (result.success) {
                    addMessage(result.response, 'bot');
                    updateStats(result.stats);
                } else {
                    addMessage('Error: ' + result.error, 'bot');
                }
                
            } catch (error) {
                addMessage('Connection error. Please try again.', 'bot');
            }
            
            // Reset UI
            textInput.value = '';
            textInput.disabled = false;
            sendBtn.disabled = false;
            processing.style.display = 'none';
            uploadedFiles = {};
            document.getElementById('uploadedFiles').innerHTML = '';
        }
        
        function addMessage(text, sender) {
            const container = document.getElementById('messages');
            const div = document.createElement('div');
            div.className = `message ${sender}`;
            div.innerHTML = `<div class="message-content">${text}</div>`;
            container.appendChild(div);
            container.scrollTop = container.scrollHeight;
        }
        
        async function updateStats(stats = null) {
            if (!stats) {
                try {
                    const response = await fetch('/stats');
                    stats = await response.json();
                } catch (error) {
                    return;
                }
            }
            
            document.getElementById('totalQueries').textContent = stats.total_queries || 0;
            document.getElementById('multimodalQueries').textContent = stats.multimodal_queries || 0;
        }
        
        async function checkStatus() {
            try {
                const response = await fetch('/health');
                const status = await response.json();
                
                document.getElementById('geminiStatus').style.backgroundColor = status.gemini ? '#28a745' : '#dc3545';
                document.getElementById('imageStatus').style.backgroundColor = status.image_processing ? '#28a745' : '#dc3545';
            } catch (error) {
                console.error('Status check failed:', error);
            }
        }
        
        function clearAll() {
            if (confirm('Clear everything?')) {
                uploadedFiles = {};
                document.getElementById('uploadedFiles').innerHTML = '';
                document.getElementById('messages').innerHTML = `
                    <div class="message bot">
                        <div class="message-content">Chat cleared! How can I help you?</div>
                    </div>
                `;
                fetch('/clear', { method: 'POST' });
            }
        }
    </script>
</body>
</html>
    ''')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        text = request.form.get('text', '').strip()
        image_data = None
        doc_data = None
        filename = None
        
        # Handle image upload
        if 'image' in request.files:
            image_file = request.files['image']
            if image_file and image_file.filename:
                image_data = image_file.read()
        
        # Handle document upload
        if 'document' in request.files:
            doc_file = request.files['document']
            if doc_file and doc_file.filename:
                doc_data = doc_file.read()
                filename = doc_file.filename
        
        # Process with bot
        response = bot.process_multimodal(
            text=text if text else None,
            image_data=image_data,
            doc_data=doc_data,
            filename=filename
        )
        
        return jsonify({
            'success': True,
            'response': response,
            'stats': bot.stats
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    return jsonify(bot.stats)

@app.route('/clear', methods=['POST'])
def clear_chat():
    bot.conversation_history = []
    bot.stats = {"total_queries": 0, "multimodal_queries": 0}
    return jsonify({'success': True})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'gemini': bot.gemini_client is not None,
        'image_processing': HAS_PIL
    })

if __name__ == '__main__':
    print("🚀 Starting Simple Multi-Modal Dashboard...")
    print("📱 Access at: http://localhost:5000")
    print("✅ This version works with minimal dependencies!")
    
    app.run(host='0.0.0.0', port=5000, debug=True)