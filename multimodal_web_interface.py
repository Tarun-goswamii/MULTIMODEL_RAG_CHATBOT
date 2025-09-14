#!/usr/bin/env python3
"""
Multi-Modal RAG Chatbot Web Interface
Advanced web interface supporting all modalities
"""

import os
import base64
import json
import io
from flask import Flask, request, jsonify, render_template_string, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import asyncio
import threading
from datetime import datetime

from multimodal_rag_chatbot import MultiModalRAGChatbot, MultiModalInput, process_multimodal_sync

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {
    'audio': {'wav', 'mp3', 'ogg', 'm4a', 'flac'},
    'video': {'mp4', 'avi', 'mov', 'mkv', 'webm'},
    'image': {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'},
    'document': {'pdf', 'docx', 'doc', 'txt'}
}

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize chatbot
chatbot = MultiModalRAGChatbot()

def allowed_file(filename, file_type):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS.get(file_type, set())

@app.route('/')
def home():
    return render_template_string('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Modal RAG Chatbot</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            flex: 1;
            display: flex;
            flex-direction: column;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
            color: white;
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .header p {
            font-size: 1.2rem;
            opacity: 0.9;
        }
        
        .chat-interface {
            display: flex;
            gap: 20px;
            flex: 1;
            min-height: 600px;
        }
        
        .chat-area {
            flex: 2;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            display: flex;
            flex-direction: column;
        }
        
        .sidebar {
            flex: 1;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            padding: 20px;
            max-height: 600px;
            overflow-y: auto;
        }
        
        .chat-messages {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            border-bottom: 1px solid #eee;
        }
        
        .message {
            margin-bottom: 20px;
            display: flex;
            align-items: flex-start;
            gap: 10px;
        }
        
        .message.user {
            flex-direction: row-reverse;
        }
        
        .message-avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2rem;
            color: white;
        }
        
        .message.user .message-avatar {
            background: #667eea;
        }
        
        .message.bot .message-avatar {
            background: #764ba2;
        }
        
        .message-content {
            max-width: 70%;
            padding: 15px 20px;
            border-radius: 20px;
            line-height: 1.5;
        }
        
        .message.user .message-content {
            background: #667eea;
            color: white;
            border-bottom-right-radius: 5px;
        }
        
        .message.bot .message-content {
            background: #f5f5f5;
            color: #333;
            border-bottom-left-radius: 5px;
        }
        
        .input-area {
            padding: 20px;
            background: #f9f9f9;
            border-radius: 0 0 15px 15px;
        }
        
        .input-container {
            display: flex;
            gap: 10px;
            align-items: flex-end;
        }
        
        .text-input {
            flex: 1;
            min-height: 50px;
            padding: 15px;
            border: 2px solid #ddd;
            border-radius: 25px;
            font-size: 1rem;
            resize: vertical;
            max-height: 150px;
        }
        
        .text-input:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .send-button {
            width: 50px;
            height: 50px;
            border: none;
            border-radius: 50%;
            background: #667eea;
            color: white;
            font-size: 1.2rem;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .send-button:hover {
            background: #5a6fd8;
            transform: scale(1.05);
        }
        
        .send-button:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        
        .upload-section {
            margin-bottom: 20px;
        }
        
        .upload-section h3 {
            margin-bottom: 15px;
            color: #333;
            font-size: 1.1rem;
        }
        
        .upload-area {
            border: 2px dashed #ddd;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .upload-area:hover {
            border-color: #667eea;
            background: #f8f9ff;
        }
        
        .upload-area.dragover {
            border-color: #667eea;
            background: #f0f4ff;
        }
        
        .file-input {
            display: none;
        }
        
        .uploaded-files {
            margin-top: 15px;
        }
        
        .uploaded-file {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px;
            background: #f5f5f5;
            border-radius: 8px;
            margin-bottom: 8px;
        }
        
        .file-icon {
            width: 30px;
            height: 30px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 5px;
            color: white;
        }
        
        .file-icon.audio { background: #ff6b6b; }
        .file-icon.video { background: #4ecdc4; }
        .file-icon.image { background: #45b7d1; }
        .file-icon.document { background: #96ceb4; }
        
        .remove-file {
            margin-left: auto;
            background: none;
            border: none;
            color: #999;
            cursor: pointer;
            font-size: 1.1rem;
        }
        
        .remove-file:hover {
            color: #ff4757;
        }
        
        .stats-section {
            background: #f9f9f9;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }
        
        .stat-item {
            text-align: center;
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
        
        .processing-indicator {
            display: none;
            align-items: center;
            gap: 10px;
            padding: 10px;
            background: #e3f2fd;
            border-radius: 8px;
            margin-bottom: 15px;
        }
        
        .spinner {
            width: 20px;
            height: 20px;
            border: 2px solid #ddd;
            border-top: 2px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.5);
        }
        
        .modal-content {
            background: white;
            margin: 5% auto;
            padding: 20px;
            border-radius: 15px;
            width: 80%;
            max-width: 600px;
            max-height: 80vh;
            overflow-y: auto;
        }
        
        .close {
            color: #aaa;
            float: right;
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
        }
        
        .close:hover {
            color: #000;
        }
        
        @media (max-width: 768px) {
            .chat-interface {
                flex-direction: column;
            }
            
            .sidebar {
                order: -1;
                max-height: 300px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><i class="fas fa-robot"></i> Multi-Modal RAG Chatbot</h1>
            <p>Upload files, record audio, or chat with text - I understand it all!</p>
        </div>
        
        <div class="chat-interface">
            <div class="chat-area">
                <div class="chat-messages" id="chatMessages">
                    <div class="message bot">
                        <div class="message-avatar"><i class="fas fa-robot"></i></div>
                        <div class="message-content">
                            Hello! I'm your multi-modal AI assistant. I can process:
                            <br>📝 Text messages
                            <br>🎵 Audio files (speech-to-text)
                            <br>🎬 Video files (analysis)
                            <br>🖼️ Images (OCR & description)
                            <br>📄 Documents (PDF, Word, etc.)
                            <br><br>Try uploading a file or just ask me anything!
                        </div>
                    </div>
                </div>
                
                <div class="processing-indicator" id="processingIndicator">
                    <div class="spinner"></div>
                    <span>Processing your multi-modal input...</span>
                </div>
                
                <div class="input-area">
                    <div class="input-container">
                        <textarea 
                            class="text-input" 
                            id="messageInput" 
                            placeholder="Type your message here... You can also upload files using the sidebar!"
                            rows="2"
                        ></textarea>
                        <button class="send-button" id="sendButton" onclick="sendMessage()">
                            <i class="fas fa-paper-plane"></i>
                        </button>
                    </div>
                </div>
            </div>
            
            <div class="sidebar">
                <div class="stats-section">
                    <h3><i class="fas fa-chart-bar"></i> Statistics</h3>
                    <div class="stats-grid">
                        <div class="stat-item">
                            <div class="stat-value" id="totalQueries">0</div>
                            <div class="stat-label">Total Queries</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value" id="multimodalQueries">0</div>
                            <div class="stat-label">Multi-modal</div>
                        </div>
                    </div>
                </div>
                
                <div class="upload-section">
                    <h3><i class="fas fa-upload"></i> Upload Files</h3>
                    
                    <div class="upload-area" onclick="document.getElementById('audioInput').click()">
                        <i class="fas fa-microphone fa-2x" style="color: #ff6b6b;"></i>
                        <p>Audio Files</p>
                        <small>MP3, WAV, OGG, M4A, FLAC</small>
                        <input type="file" id="audioInput" class="file-input" accept=".mp3,.wav,.ogg,.m4a,.flac" onchange="handleFileUpload(this, 'audio')">
                    </div>
                    
                    <br>
                    
                    <div class="upload-area" onclick="document.getElementById('videoInput').click()">
                        <i class="fas fa-video fa-2x" style="color: #4ecdc4;"></i>
                        <p>Video Files</p>
                        <small>MP4, AVI, MOV, MKV, WEBM</small>
                        <input type="file" id="videoInput" class="file-input" accept=".mp4,.avi,.mov,.mkv,.webm" onchange="handleFileUpload(this, 'video')">
                    </div>
                    
                    <br>
                    
                    <div class="upload-area" onclick="document.getElementById('imageInput').click()">
                        <i class="fas fa-image fa-2x" style="color: #45b7d1;"></i>
                        <p>Image Files</p>
                        <small>PNG, JPG, GIF, BMP, TIFF</small>
                        <input type="file" id="imageInput" class="file-input" accept=".png,.jpg,.jpeg,.gif,.bmp,.tiff" onchange="handleFileUpload(this, 'image')">
                    </div>
                    
                    <br>
                    
                    <div class="upload-area" onclick="document.getElementById('documentInput').click()">
                        <i class="fas fa-file-alt fa-2x" style="color: #96ceb4;"></i>
                        <p>Documents</p>
                        <small>PDF, DOCX, DOC, TXT</small>
                        <input type="file" id="documentInput" class="file-input" accept=".pdf,.docx,.doc,.txt" onchange="handleFileUpload(this, 'document')">
                    </div>
                    
                    <div class="uploaded-files" id="uploadedFiles"></div>
                </div>
                
                <button onclick="clearAll()" style="width: 100%; padding: 10px; background: #ff4757; color: white; border: none; border-radius: 8px; cursor: pointer; margin-top: 15px;">
                    <i class="fas fa-trash"></i> Clear All
                </button>
            </div>
        </div>
    </div>
    
    <!-- History Modal -->
    <div id="historyModal" class="modal">
        <div class="modal-content">
            <span class="close" onclick="closeHistoryModal()">&times;</span>
            <h2>Conversation History</h2>
            <div id="historyContent"></div>
        </div>
    </div>
    
    <script>
        let uploadedFiles = {};
        let conversationHistory = [];
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            updateStats();
            
            // Enter key support
            document.getElementById('messageInput').addEventListener('keypress', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                }
            });
        });
        
        function handleFileUpload(input, type) {
            const file = input.files[0];
            if (file) {
                const fileId = Date.now() + '_' + file.name;
                uploadedFiles[fileId] = {
                    file: file,
                    type: type,
                    name: file.name
                };
                
                displayUploadedFile(fileId, file.name, type);
                input.value = ''; // Reset input
            }
        }
        
        function displayUploadedFile(fileId, fileName, type) {
            const container = document.getElementById('uploadedFiles');
            const fileDiv = document.createElement('div');
            fileDiv.className = 'uploaded-file';
            fileDiv.id = 'file_' + fileId;
            
            const iconMap = {
                'audio': 'fa-music',
                'video': 'fa-video',
                'image': 'fa-image',
                'document': 'fa-file-alt'
            };
            
            fileDiv.innerHTML = `
                <div class="file-icon ${type}">
                    <i class="fas ${iconMap[type]}"></i>
                </div>
                <div style="flex: 1;">
                    <div style="font-weight: bold; font-size: 0.9rem;">${fileName}</div>
                    <div style="font-size: 0.8rem; color: #666;">${type.toUpperCase()}</div>
                </div>
                <button class="remove-file" onclick="removeFile('${fileId}')">
                    <i class="fas fa-times"></i>
                </button>
            `;
            
            container.appendChild(fileDiv);
        }
        
        function removeFile(fileId) {
            delete uploadedFiles[fileId];
            const fileElement = document.getElementById('file_' + fileId);
            if (fileElement) {
                fileElement.remove();
            }
        }
        
        async function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            const sendButton = document.getElementById('sendButton');
            
            if (!message && Object.keys(uploadedFiles).length === 0) {
                return;
            }
            
            // Disable input
            input.disabled = true;
            sendButton.disabled = true;
            document.getElementById('processingIndicator').style.display = 'flex';
            
            // Add user message to chat
            if (message) {
                addMessage(message, 'user');
            }
            
            // Add file indicators
            Object.values(uploadedFiles).forEach(fileInfo => {
                addMessage(`📎 Uploaded ${fileInfo.type}: ${fileInfo.name}`, 'user');
            });
            
            try {
                // Prepare form data
                const formData = new FormData();
                if (message) {
                    formData.append('text', message);
                }
                
                // Add files
                Object.values(uploadedFiles).forEach(fileInfo => {
                    formData.append(fileInfo.type, fileInfo.file);
                });
                
                // Send request
                const response = await fetch('/chat', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (result.success) {
                    addMessage(result.response, 'bot');
                    
                    // Show processing summary
                    if (result.multimodal_summary && Object.keys(result.multimodal_summary).length > 0) {
                        const summaryText = Object.entries(result.multimodal_summary)
                            .map(([type, desc]) => `${type}: ${desc}`)
                            .join(' | ');
                        addMessage(`📊 Processing Summary: ${summaryText}`, 'bot', 'info');
                    }
                    
                    // Update stats
                    updateStats(result.stats);
                } else {
                    addMessage(`Error: ${result.error}`, 'bot', 'error');
                }
                
            } catch (error) {
                addMessage('Sorry, there was an error processing your request.', 'bot', 'error');
                console.error('Error:', error);
            }
            
            // Reset
            input.value = '';
            input.disabled = false;
            sendButton.disabled = false;
            document.getElementById('processingIndicator').style.display = 'none';
            uploadedFiles = {};
            document.getElementById('uploadedFiles').innerHTML = '';
        }
        
        function addMessage(text, sender, type = 'normal') {
            const container = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}`;
            
            const avatar = sender === 'user' ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>';
            
            let messageClass = 'message-content';
            if (type === 'info') {
                messageClass += ' info-message';
            } else if (type === 'error') {
                messageClass += ' error-message';
            }
            
            messageDiv.innerHTML = `
                <div class="message-avatar">${avatar}</div>
                <div class="${messageClass}">${text}</div>
            `;
            
            container.appendChild(messageDiv);
            container.scrollTop = container.scrollHeight;
        }
        
        async function updateStats(newStats = null) {
            try {
                let stats = newStats;
                if (!stats) {
                    const response = await fetch('/stats');
                    stats = await response.json();
                }
                
                document.getElementById('totalQueries').textContent = stats.total_queries || 0;
                document.getElementById('multimodalQueries').textContent = stats.multimodal_queries || 0;
            } catch (error) {
                console.error('Error updating stats:', error);
            }
        }
        
        function clearAll() {
            if (confirm('Clear all uploaded files and chat history?')) {
                uploadedFiles = {};
                document.getElementById('uploadedFiles').innerHTML = '';
                document.getElementById('chatMessages').innerHTML = `
                    <div class="message bot">
                        <div class="message-avatar"><i class="fas fa-robot"></i></div>
                        <div class="message-content">Chat cleared! How can I help you today?</div>
                    </div>
                `;
                
                // Clear server-side history
                fetch('/clear', { method: 'POST' });
            }
        }
        
        // Drag and drop support
        document.addEventListener('dragover', function(e) {
            e.preventDefault();
        });
        
        document.addEventListener('drop', function(e) {
            e.preventDefault();
            const files = e.dataTransfer.files;
            
            for (let file of files) {
                let type = 'document';
                
                if (file.type.startsWith('audio/')) type = 'audio';
                else if (file.type.startsWith('video/')) type = 'video';
                else if (file.type.startsWith('image/')) type = 'image';
                
                const fileId = Date.now() + '_' + file.name;
                uploadedFiles[fileId] = {
                    file: file,
                    type: type,
                    name: file.name
                };
                
                displayUploadedFile(fileId, file.name, type);
            }
        });
        
        // Add custom CSS for info and error messages
        const style = document.createElement('style');
        style.textContent = `
            .info-message {
                background: #e3f2fd !important;
                color: #1976d2 !important;
                font-size: 0.9rem;
                border-left: 4px solid #2196f3;
            }
            
            .error-message {
                background: #ffebee !important;
                color: #c62828 !important;
                border-left: 4px solid #f44336;
            }
        `;
        document.head.appendChild(style);
    </script>
</body>
</html>
    ''')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Get text input
        text_input = request.form.get('text', '').strip()
        
        # Process uploaded files
        multimodal_input = MultiModalInput(text=text_input if text_input else None)
        
        # Handle audio file
        if 'audio' in request.files:
            audio_file = request.files['audio']
            if audio_file and audio_file.filename:
                multimodal_input.audio_data = audio_file.read()
        
        # Handle video file
        if 'video' in request.files:
            video_file = request.files['video']
            if video_file and video_file.filename:
                multimodal_input.video_data = video_file.read()
        
        # Handle image file
        if 'image' in request.files:
            image_file = request.files['image']
            if image_file and image_file.filename:
                multimodal_input.image_data = image_file.read()
        
        # Handle document file
        if 'document' in request.files:
            document_file = request.files['document']
            if document_file and document_file.filename:
                multimodal_input.document_data = document_file.read()
                multimodal_input.file_type = document_file.filename.rsplit('.', 1)[1].lower()
        
        # Process the multimodal input
        result = process_multimodal_sync(chatbot, multimodal_input)
        
        # Get updated stats
        stats = chatbot.get_processing_stats()
        
        return jsonify({
            'success': True,
            'response': result['response'],
            'processing_time': result['processing_time'],
            'multimodal_summary': result['multimodal_summary'],
            'confidence': result['confidence'],
            'stats': stats
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    try:
        stats = chatbot.get_processing_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history', methods=['GET'])
def get_history():
    try:
        limit = request.args.get('limit', 10, type=int)
        history = chatbot.get_conversation_history(limit)
        return jsonify(history)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clear', methods=['POST'])
def clear_history():
    try:
        chatbot.conversation_history = []
        chatbot.processing_stats = {
            "total_queries": 0,
            "multimodal_queries": 0,
            "processing_time_avg": 0.0
        }
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'components': {
                'gemini': bool(chatbot.gemini_client),
                'qdrant': bool(chatbot.qdrant_client),
                'embeddings': bool(chatbot.embedding_model),
                'audio_processing': chatbot.audio_processor.recognizer is not None,
                'video_processing': chatbot.video_processor.face_cascade is not None,
                'image_processing': chatbot.image_processor.gemini_vision is not None,
                'document_processing': True
            },
            'stats': chatbot.get_processing_stats()
        }
        return jsonify(status)
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

if __name__ == '__main__':
    print("🌐 Starting Multi-Modal RAG Chatbot Web Interface...")
    print("📱 Access at: http://localhost:5000")
    print("🎯 Features: Text, Audio, Video, Image, Document processing")
    print("📋 Press Ctrl+C to stop")
    
    app.run(host='0.0.0.0', port=5000, debug=True)