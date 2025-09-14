# Multi-Modal RAG Chatbot System

## Current Project Features:

### 1. Multi-Modal Content Processing
- **Text Processing**: Natural language understanding and response generation
- **Audio Processing**: Speech-to-text transcription using Google Speech Recognition
- **Image Processing**: AI-powered image description using Google Gemini Vision API
- **Document Processing**: PDF and text file content extraction and analysis
- **Video Processing**: Frame analysis and face detection capabilities (when OpenCV available)

### 2. Advanced Vector Search System
- **Qdrant Vector Database**: Professional vector database for semantic search
- **Semantic Skill Matching**: Vector-based candidate-to-job matching
- **Code Similarity Detection**: Plagiarism detection using semantic code analysis
- **Adaptive Question Generation**: AI-powered interview questions based on candidate profiles
- **Knowledge Base Integration**: Multi-modal document storage and retrieval

### 3. RAG (Retrieval-Augmented Generation) Engine
- **Context-Aware Responses**: AI responses enhanced with relevant knowledge base content
- **Multi-Modal Knowledge Base**: Support for text, audio, image, and document storage
- **Semantic Search**: Vector similarity search across all content types
- **Conversation History**: Persistent chat history with processing statistics
- **Real-time Processing**: Asynchronous multi-modal content processing

### 4. Flask Web Application
- **RESTful API**: Comprehensive API for all system functions
- **File Upload Support**: Multi-modal file upload and processing
- **Real-time Chat Interface**: Interactive chat with AI assistant
- **Knowledge Management**: CRUD operations for knowledge base
- **Vector Statistics**: Real-time monitoring of vector database status

### 5. Lightweight & Scalable Design
- **Dependency Management**: Graceful fallback when optional dependencies unavailable
- **Environment Configuration**: Flexible setup with multiple virtual environments
- **Error Handling**: Robust error handling and user feedback
- **Performance Monitoring**: Processing time tracking and optimization

## Technical Architecture:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Interface │    │   Flask Backend │    │   AI Services   │
│                 │    │                 │    │                 │
│ • File Upload   │◄──►│ • Multi-Modal   │◄──►│ • Google Gemini │
│ • Chat UI       │    │   Processing    │    │ • Speech-to-Text│
│ • Vector Search │    │ • RAG Engine    │    │ • Vision API    │
│ • Knowledge Mgmt│    │ • API Routes    │    │ • OCR Processing│
└─────────────────┘    └─────────────────┘    └─────────────────┘
          │                       │                       │
          └───────────────────────┼───────────────────────┘
                                  │
                         ┌─────────────────┐
                         │  Vector Storage │
                         │                 │
                         │ • Qdrant DB     │
                         │ • Embeddings    │
                         │ • Similarity    │
                         │ • Collections   │
                         └─────────────────┘
```

## Key Technologies Used:

- **Backend**: Flask, Python 3.13
- **AI/ML**: Google Gemini, SentenceTransformers, OpenCV, PIL
- **Vector DB**: Qdrant (cloud and local)
- **Audio**: SpeechRecognition, PyAudio
- **Documents**: PyPDF2, python-docx
- **Environment**: Multiple virtual environments (lightweight_env, multimodal_env, venv_backend)

## Current System Capabilities:

✅ **Multi-modal file processing** (text, audio, images, PDFs)  
✅ **Vector-based semantic search** with Qdrant  
✅ **AI-powered responses** using Google Gemini  
✅ **Knowledge base management** with CRUD operations  
✅ **Real-time chat interface** with context awareness  
✅ **Code similarity detection** for plagiarism prevention  
✅ **Adaptive question generation** for interviews  
✅ **Professional web interface** with file upload support  
✅ **Robust error handling** and graceful fallbacks  
✅ **Performance monitoring** and statistics tracking