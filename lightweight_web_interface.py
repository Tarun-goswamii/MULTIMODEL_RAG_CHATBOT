#!/usr/bin/env python3
"""
Lightweight Multi-Modal Web Interface
Works without problematic dependencies
"""

import os
import base64
import json
import io
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from werkzeug.utils import secure_filename
from datetime import datetime

from lightweight_multimodal_chatbot import LightweightMultiModalChatbot, MultiModalInput

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize chatbot
print("🤖 Initializing Lightweight Multi-Modal Chatbot...")
chatbot = LightweightMultiModalChatbot()

@app.route('/')
def home():
    return render_template_string('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lightweight Multi-Modal Chatbot</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        
        .container {
            max-width: 1200px;
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
            font-size: 1.1rem;
            opacity: 0.9;
        }
        
        .chat-container {
            display: flex;
            gap: 20px;
            flex: 1;
            min-height: 500px;
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
            max-height: 500px;
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
            background: #74b9ff;
        }
        
        .message.bot .message-avatar {
            background: #0984e3;
        }
        
        .message-content {
            max-width: 70%;
            padding: 15px 20px;
            border-radius: 20px;
            line-height: 1.5;
        }
        
        .message.user .message-content {
            background: #74b9ff;
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
            border-color: #74b9ff;
        }
        
        .send-button {
            width: 50px;
            height: 50px;
            border: none;
            border-radius: 50%;
            background: #74b9ff;
            color: white;
            font-size: 1.2rem;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .send-button:hover {
            background: #0984e3;
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
        
        .upload-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }
        
        .upload-area {
            border: 2px dashed #ddd;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            transition: all 0.3s ease;
            cursor: pointer;
            min-height: 80px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        
        .upload-area:hover {
            border-color: #74b9ff;
            background: #f8f9ff;
        }
        
        .upload-area i {
            font-size: 1.5rem;
            margin-bottom: 5px;
        }
        
        .upload-area p {
            font-size: 0.9rem;
            font-weight: bold;
            margin-bottom: 2px;
        }
        
        .upload-area small {
            font-size: 0.7rem;
            color: #666;
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
            padding: 8px;
            background: #f5f5f5;
            border-radius: 8px;
            margin-bottom: 8px;
            font-size: 0.9rem;
        }
        
        .file-icon {
            width: 25px;
            height: 25px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 5px;
            color: white;
            font-size: 0.8rem;
        }
        
        .file-icon.audio { background: #e17055; }
        .file-icon.image { background: #74b9ff; }
        .file-icon.document { background: #00b894; }
        
        .remove-file {
            margin-left: auto;
            background: none;
            border: none;
            color: #999;
            cursor: pointer;
            font-size: 1rem;
        }
        
        .remove-file:hover {
            color: #d63031;
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
            color: #74b9ff;
        }
        
        .stat-label {
            font-size: 0.8rem;
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
            border-top: 2px solid #74b9ff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .clear-btn {
            width: 100%;
            padding: 10px;
            background: #d63031;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            margin-top: 15px;
        }
        
        .clear-btn:hover {
            background: #b12a2a;
        }
        
        @media (max-width: 768px) {
            .chat-container {
                flex-direction: column;
            }
            
            .sidebar {
                order: -1;
                max-height: 250px;
            }
            
            .upload-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><i class="fas fa-robot"></i> Lightweight Multi-Modal Chatbot</h1>
            <p>Fast, reliable AI that understands text, audio, images, and documents!</p>
        </div>
        
        <div class="chat-container">
            <div class="chat-area">
                <div class="chat-messages" id="chatMessages">
                    <div class="message bot">
                        <div class="message-avatar"><i class="fas fa-robot"></i></div>
                        <div class="message-content">
                            Hello! I'm your lightweight multi-modal AI assistant. I can process:
                            <br>📝 Text messages
                            <br>🎵 Audio files (speech-to-text)  
                            <br>🖼️ Images (AI description)
                            <br>📄 Documents (PDF, TXT)
                            <br><br>Upload files or just chat with me!
                        </div>
                    </div>
                </div>
                
                <div class="processing-indicator" id="processingIndicator">
                    <div class="spinner"></div>
                    <span>Processing your input...</span>
                </div>
                
                <div class="input-area">
                    <div class="input-container">
                        <textarea 
                            class="text-input" 
                            id="messageInput" 
                            placeholder="Type your message here..."
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
                            <div class="stat-label">Total</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value" id="multimodalQueries">0</div>
                            <div class="stat-label">Multi-modal</div>
                        </div>
                    </div>
                </div>
                
                <div class="upload-section">
                    <h3><i class="fas fa-upload"></i> Upload Files</h3>
                    
                    <div class="upload-grid">
                        <div class="upload-area" onclick="document.getElementById('audioInput').click()">
                            <i class="fas fa-microphone" style="color: #e17055;"></i>
                            <p>Audio</p>
                            <small>MP3, WAV</small>
                            <input type="file" id="audioInput" class="file-input" accept=".mp3,.wav,.ogg" onchange="handleFileUpload(this, 'audio')">
                        </div>
                        
                        <div class="upload-area" onclick="document.getElementById('imageInput').click()">
                            <i class="fas fa-image" style="color: #74b9ff;"></i>
                            <p>Image</p>
                            <small>PNG, JPG</small>
                            <input type="file" id="imageInput" class="file-input" accept=".png,.jpg,.jpeg,.gif" onchange="handleFileUpload(this, 'image')">
                        </div>
                        
                        <div class="upload-area" onclick="document.getElementById('documentInput').click()">
                            <i class="fas fa-file-alt" style="color: #00b894;"></i>
                            <p>Document</p>
                            <small>PDF, TXT</small>
                            <input type="file" id="documentInput" class="file-input" accept=".pdf,.txt" onchange="handleFileUpload(this, 'document')">
                        </div>
                    </div>
                    
                    <div class="uploaded-files" id="uploadedFiles"></div>
                </div>
                
                <button onclick="clearAll()" class="clear-btn">
                    <i class="fas fa-trash"></i> Clear All
                </button>
            </div>
        </div>
    </div>
    
    <script>
        let uploadedFiles = {};
        
        document.addEventListener('DOMContentLoaded', function() {
            updateStats();
            
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
                input.value = '';
            }
        }
        
        function displayUploadedFile(fileId, fileName, type) {
            const container = document.getElementById('uploadedFiles');
            const fileDiv = document.createElement('div');
            fileDiv.className = 'uploaded-file';
            fileDiv.id = 'file_' + fileId;
            
            const iconMap = {
                'audio': 'fa-music',
                'image': 'fa-image',
                'document': 'fa-file-alt'
            };
            
            fileDiv.innerHTML = `
                <div class="file-icon ${type}">
                    <i class="fas ${iconMap[type]}"></i>
                </div>
                <div style="flex: 1;">
                    <div style="font-weight: bold;">${fileName}</div>
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
            
            input.disabled = true;
            sendButton.disabled = true;
            document.getElementById('processingIndicator').style.display = 'flex';
            
            if (message) {
                addMessage(message, 'user');
            }
            
            Object.values(uploadedFiles).forEach(fileInfo => {
                addMessage(`📎 ${fileInfo.type}: ${fileInfo.name}`, 'user');
            });
            
            try {
                const formData = new FormData();
                if (message) {
                    formData.append('text', message);
                }
                
                Object.values(uploadedFiles).forEach(fileInfo => {
                    formData.append(fileInfo.type, fileInfo.file);
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
                    addMessage(`Error: ${result.error}`, 'bot');
                }
                
            } catch (error) {
                addMessage('Sorry, there was an error processing your request.', 'bot');
                console.error('Error:', error);
            }
            
            input.value = '';
            input.disabled = false;
            sendButton.disabled = false;
            document.getElementById('processingIndicator').style.display = 'none';
            uploadedFiles = {};
            document.getElementById('uploadedFiles').innerHTML = '';
        }
        
        function addMessage(text, sender) {
            const container = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}`;
            
            const avatar = sender === 'user' ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>';
            
            messageDiv.innerHTML = `
                <div class="message-avatar">${avatar}</div>
                <div class="message-content">${text}</div>
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
        text_input = request.form.get('text', '').strip()
        multimodal_input = MultiModalInput(text=text_input if text_input else None)
        
        # Handle file uploads
        if 'audio' in request.files:
            audio_file = request.files['audio']
            if audio_file and audio_file.filename:
                multimodal_input.audio_data = audio_file.read()
        
        if 'image' in request.files:
            image_file = request.files['image']
            if image_file and image_file.filename:
                multimodal_input.image_data = image_file.read()
        
        if 'document' in request.files:
            document_file = request.files['document']
            if document_file and document_file.filename:
                multimodal_input.document_data = document_file.read()
                multimodal_input.file_type = document_file.filename.rsplit('.', 1)[1].lower()
        
        # Process the input
        result = chatbot.process_multimodal_input(multimodal_input)
        stats = chatbot.get_processing_stats()
        
        return jsonify({
            'success': True,
            'response': result['response'],
            'processing_time': result['processing_time'],
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
    try:
        status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'components': {
                'gemini': bool(chatbot.gemini_client),
                'qdrant': bool(chatbot.qdrant_client),
                'audio_processing': bool(chatbot.audio_processor.recognizer),
                'image_processing': True,
                'document_processing': True
            },
            'stats': chatbot.get_processing_stats()
        }
        return jsonify(status)
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

if __name__ == '__main__':
    print("🌐 Starting Lightweight Multi-Modal Web Interface...")
    print("📱 Access at: http://localhost:5000")
    print("🚀 Features: Text, Audio, Image, Document processing")
    print("📋 Press Ctrl+C to stop")
    
    app.run(host='0.0.0.0', port=5000, debug=True)