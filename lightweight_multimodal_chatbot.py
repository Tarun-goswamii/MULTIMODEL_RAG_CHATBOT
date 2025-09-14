#!/usr/bin/env python3
"""
Lightweight Multi-Modal RAG Chatbot
Optimized version without problematic dependencies
"""

import os
import sys
import json
import base64
import tempfile
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any, Union
import asyncio
import threading
from dataclasses import dataclass
import logging

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

# Import core dependencies with error handling
try:
    import google.generativeai as genai
    HAS_GEMINI = True
    print("✅ Google Gemini imported")
except ImportError:
    print("⚠️ Google Gemini not installed")
    HAS_GEMINI = False

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    HAS_QDRANT = True
    print("✅ Qdrant imported")
except ImportError:
    print("⚠️ Qdrant not installed")
    HAS_QDRANT = False

try:
    from flask import Flask, request, jsonify, render_template_string
    from flask_cors import CORS
    HAS_FLASK = True
    print("✅ Flask imported")
except ImportError:
    print("⚠️ Flask not installed")
    HAS_FLASK = False

# Optional multi-modal processing imports (graceful fallback)
try:
    import cv2
    import numpy as np
    HAS_OPENCV = True
    print("✅ OpenCV imported")
except ImportError:
    print("⚠️ OpenCV not installed - video processing disabled")
    HAS_OPENCV = False

try:
    import speech_recognition as sr
    HAS_AUDIO = True
    print("✅ Audio processing imported")
except ImportError:
    print("⚠️ Audio processing not installed")
    HAS_AUDIO = False

try:
    from PIL import Image
    HAS_PIL = True
    print("✅ PIL imported")
except ImportError:
    print("⚠️ PIL not installed")
    HAS_PIL = False

try:
    import PyPDF2
    HAS_PDF = True
    print("✅ PDF processing imported")
except ImportError:
    print("⚠️ PDF processing not installed")
    HAS_PDF = False

@dataclass
class MultiModalInput:
    """Data structure for multi-modal input"""
    text: Optional[str] = None
    audio_data: Optional[bytes] = None
    video_data: Optional[bytes] = None
    image_data: Optional[bytes] = None
    document_data: Optional[bytes] = None
    file_type: Optional[str] = None
    metadata: Dict[str, Any] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}

@dataclass
class ProcessedContent:
    """Processed multi-modal content"""
    extracted_text: str = ""
    audio_transcript: str = ""
    image_description: str = ""
    video_summary: str = ""
    document_text: str = ""
    entities: List[str] = None
    sentiment: str = "neutral"
    confidence: float = 0.0
    processing_time: float = 0.0
    
    def __post_init__(self):
        if self.entities is None:
            self.entities = []

class SimpleTextProcessor:
    """Simple text processing without heavy dependencies"""
    
    def __init__(self):
        # Simple keyword-based similarity
        self.stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
            'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were'
        }
    
    def create_simple_embedding(self, text: str) -> List[float]:
        """Create simple word-based embedding"""
        words = text.lower().split()
        words = [w for w in words if w not in self.stopwords and len(w) > 2]
        
        # Simple bag of words representation
        vocab = set(words)
        embedding = []
        
        # Create a simple feature vector
        for word in sorted(vocab)[:50]:  # Limit to 50 features
            embedding.append(words.count(word) / len(words) if words else 0)
        
        # Pad or truncate to fixed size
        while len(embedding) < 50:
            embedding.append(0.0)
        
        return embedding[:50]
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Simple cosine similarity"""
        try:
            import math
            
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = math.sqrt(sum(a * a for a in vec1))
            magnitude2 = math.sqrt(sum(a * a for a in vec2))
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            
            return dot_product / (magnitude1 * magnitude2)
        except:
            return 0.0

class AudioProcessor:
    """Simple audio processing"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer() if HAS_AUDIO else None
        
    def process_audio(self, audio_data: bytes, file_format: str = "wav") -> Dict[str, Any]:
        """Process audio file and extract information"""
        if not HAS_AUDIO:
            return {"transcript": "Audio processing not available", "error": "Dependencies missing"}
        
        try:
            # Save temporary audio file
            with tempfile.NamedTemporaryFile(suffix=f".{file_format}", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_path = temp_file.name
            
            # Simple transcription
            with sr.AudioFile(temp_path) as source:
                audio = self.recognizer.record(source)
                
            try:
                transcript = self.recognizer.recognize_google(audio)
            except:
                transcript = "Could not transcribe audio"
            
            # Clean up
            os.unlink(temp_path)
            
            return {
                "transcript": transcript,
                "duration": len(audio_data) / 44100,  # Rough estimate
                "confidence": 0.8
            }
            
        except Exception as e:
            return {"transcript": "", "error": str(e)}

class ImageProcessor:
    """Simple image processing"""
    
    def __init__(self):
        self.gemini_vision = None
        if HAS_GEMINI:
            try:
                self.gemini_vision = genai.GenerativeModel('gemini-1.5-flash')
            except:
                pass
    
    def process_image(self, image_data: bytes) -> Dict[str, Any]:
        """Process image and extract information"""
        try:
            if not HAS_PIL:
                return {"description": "Image processing not available", "error": "PIL not installed"}
            
            # Convert to PIL Image
            import io
            image = Image.open(io.BytesIO(image_data))
            
            results = {
                "description": "",
                "text_content": "",
                "width": image.width,
                "height": image.height,
                "format": image.format or "Unknown",
                "confidence": 0.7
            }
            
            # AI-powered image description using Gemini
            if self.gemini_vision:
                try:
                    response = self.gemini_vision.generate_content([
                        "Describe this image in detail. What do you see?",
                        image
                    ])
                    results["description"] = response.text
                    results["confidence"] = 0.9
                except Exception as e:
                    results["description"] = f"Could not analyze image with AI: {str(e)}"
            else:
                results["description"] = f"Image detected: {image.width}x{image.height} pixels, format: {image.format}"
            
            return results
            
        except Exception as e:
            return {"description": "", "error": str(e)}

class DocumentProcessor:
    """Simple document processing"""
    
    def process_document(self, document_data: bytes, file_type: str) -> Dict[str, Any]:
        """Process document and extract text content"""
        try:
            text_content = ""
            
            if file_type.lower() == "pdf" and HAS_PDF:
                text_content = self._process_pdf(document_data)
            elif file_type.lower() == "txt":
                text_content = document_data.decode('utf-8', errors='ignore')
            else:
                return {"text": "", "error": f"Unsupported document type: {file_type}"}
            
            # Extract metadata
            word_count = len(text_content.split())
            char_count = len(text_content)
            
            return {
                "text": text_content,
                "word_count": word_count,
                "char_count": char_count,
                "file_type": file_type,
                "confidence": 0.9
            }
            
        except Exception as e:
            return {"text": "", "error": str(e)}
    
    def _process_pdf(self, pdf_data: bytes) -> str:
        """Extract text from PDF"""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(pdf_data)
            temp_path = temp_file.name
        
        try:
            with open(temp_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            os.unlink(temp_path)
            return text
        except:
            os.unlink(temp_path)
            return ""

class LightweightMultiModalChatbot:
    """Lightweight Multi-Modal RAG Chatbot without heavy dependencies"""
    
    def __init__(self):
        """Initialize the lightweight chatbot"""
        print("🤖 Initializing Lightweight Multi-Modal RAG Chatbot...")
        
        # Core components
        self.gemini_client = None
        self.qdrant_client = None
        self.text_processor = SimpleTextProcessor()
        self.knowledge_base = []
        
        # Multi-modal processors
        self.audio_processor = AudioProcessor()
        self.image_processor = ImageProcessor()
        self.document_processor = DocumentProcessor()
        
        # Initialize core components
        self._init_gemini()
        self._init_qdrant()
        self._load_knowledge_base()
        
        # Conversation history
        self.conversation_history = []
        self.processing_stats = {
            "total_queries": 0,
            "multimodal_queries": 0,
            "processing_time_avg": 0.0
        }
        
        print("✅ Lightweight Multi-Modal RAG Chatbot ready!")
    
    def _init_gemini(self):
        """Initialize Google Gemini client"""
        if not HAS_GEMINI:
            print("⚠️ Google Gemini library not installed")
            return
        
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            print("⚠️ Gemini API key not configured")
            return
        
        try:
            genai.configure(api_key=api_key)
            self.gemini_client = genai.GenerativeModel('gemini-1.5-flash')
            print("✅ Gemini initialized successfully")
        except Exception as e:
            print(f"❌ Gemini initialization failed: {e}")
            self.gemini_client = None
    
    def _init_qdrant(self):
        """Initialize Qdrant client"""
        qdrant_api_key = os.getenv('QDRANT_API_KEY')
        if qdrant_api_key and HAS_QDRANT:
            try:
                self.qdrant_client = QdrantClient(
                    url="https://35f8ce34-d42f-4d11-a6f8-376f05d9b152.us-east-1-1.aws.cloud.qdrant.io:6333",
                    api_key=qdrant_api_key,
                )
                collections = self.qdrant_client.get_collections()
                print("✅ Qdrant Cloud connected")
                return
            except Exception as e:
                print(f"⚠️ Qdrant Cloud failed: {e}")
        
        print("❌ No Qdrant available - using local search only")
        self.qdrant_client = None
    
    def _load_knowledge_base(self):
        """Load comprehensive knowledge base"""
        self.knowledge_base = [
            {
                "text": "This is a lightweight multi-modal RAG chatbot that can process text, audio, video, images, and documents. It uses simplified processing to avoid dependency conflicts.",
                "category": "system",
                "keywords": ["multimodal", "rag", "chatbot", "lightweight"]
            },
            {
                "text": "Audio processing includes speech-to-text transcription using Google's speech recognition API. Supports WAV, MP3, and other common audio formats.",
                "category": "audio",
                "keywords": ["audio", "speech", "transcription", "voice"]
            },
            {
                "text": "Image processing includes AI-powered description using Google Gemini Vision API. Can analyze photos, screenshots, diagrams, and other visual content.",
                "category": "image",
                "keywords": ["image", "visual", "photo", "description"]
            },
            {
                "text": "Document processing supports PDF and text files. Extracts content and provides intelligent analysis and summarization.",
                "category": "document",
                "keywords": ["document", "pdf", "text", "extraction"]
            },
            {
                "text": "The system uses Google Gemini for AI responses and can optionally connect to Qdrant for vector search capabilities.",
                "category": "ai",
                "keywords": ["gemini", "ai", "vector", "search"]
            }
        ]
        print(f"📚 Loaded {len(self.knowledge_base)} knowledge entries")
    
    def process_multimodal_input(self, multimodal_input: MultiModalInput) -> Dict[str, Any]:
        """Process multi-modal input"""
        start_time = datetime.now()
        processed_content = ProcessedContent()
        
        # Process different modalities
        if multimodal_input.text:
            processed_content.extracted_text = multimodal_input.text
        
        if multimodal_input.audio_data:
            audio_result = self.audio_processor.process_audio(multimodal_input.audio_data)
            processed_content.audio_transcript = audio_result.get('transcript', '')
        
        if multimodal_input.image_data:
            image_result = self.image_processor.process_image(multimodal_input.image_data)
            processed_content.image_description = image_result.get('description', '')
        
        if multimodal_input.document_data:
            doc_result = self.document_processor.process_document(
                multimodal_input.document_data, 
                multimodal_input.file_type or "txt"
            )
            processed_content.document_text = doc_result.get('text', '')
        
        # Combine all extracted text
        all_text = " ".join(filter(None, [
            processed_content.extracted_text,
            processed_content.audio_transcript,
            processed_content.image_description,
            processed_content.document_text
        ]))
        
        # Generate response
        response = self._generate_response(all_text, processed_content)
        
        # Update statistics
        processing_time = (datetime.now() - start_time).total_seconds()
        self.processing_stats["total_queries"] += 1
        if any([multimodal_input.audio_data, multimodal_input.image_data, multimodal_input.document_data]):
            self.processing_stats["multimodal_queries"] += 1
        
        # Store in conversation history
        conversation_entry = {
            "timestamp": start_time.isoformat(),
            "input": {
                "text": multimodal_input.text,
                "has_audio": bool(multimodal_input.audio_data),
                "has_image": bool(multimodal_input.image_data),
                "has_document": bool(multimodal_input.document_data)
            },
            "response": response,
            "processing_time": processing_time
        }
        self.conversation_history.append(conversation_entry)
        
        return {
            "response": response,
            "processed_content": processed_content,
            "processing_time": processing_time,
            "confidence": 0.8
        }
    
    def _generate_response(self, combined_text: str, processed_content: ProcessedContent) -> str:
        """Generate response using available AI"""
        if not combined_text.strip():
            return "I didn't receive any content to process. Please provide text, upload a file, or record audio."
        
        # Search knowledge base
        context = self.search_knowledge(combined_text)
        
        # Try Gemini first
        if self.gemini_client:
            try:
                context_text = "\n".join([item["text"] for item in context[:3]]) if context else "No specific context found."
                
                prompt = f"""You are a helpful multi-modal AI assistant. Analyze and respond to the following input:

INPUT: {combined_text}

CONTEXT: {context_text}

Provide a helpful, informative response. If multiple types of content were provided (text, audio, image, document), acknowledge them in your response."""

                response = self.gemini_client.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                print(f"⚠️ Gemini failed: {e}")
        
        # Fallback response
        if context:
            return f"Based on your input, here's what I found: {context[0]['text']}"
        else:
            return f"I processed your multi-modal input: {combined_text[:200]}{'...' if len(combined_text) > 200 else ''}"
    
    def search_knowledge(self, query: str, limit: int = 3) -> List[Dict]:
        """Simple keyword-based knowledge search"""
        if not query.strip():
            return []
        
        query_lower = query.lower()
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
            for word in query_lower.split():
                if len(word) > 2 and word in text_lower:
                    score += 2
            
            if score > 0:
                scored_results.append({"text": item["text"], "score": score})
        
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[:limit]
    
    def get_conversation_history(self, limit: int = 10) -> List[Dict]:
        """Get recent conversation history"""
        return self.conversation_history[-limit:]
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            **self.processing_stats,
            "multimodal_percentage": (self.processing_stats["multimodal_queries"] / max(1, self.processing_stats["total_queries"])) * 100,
            "conversation_length": len(self.conversation_history)
        }

if __name__ == "__main__":
    print("🚀 Starting Lightweight Multi-Modal RAG Chatbot System...")
    
    # Initialize chatbot
    chatbot = LightweightMultiModalChatbot()
    
    # Test basic functionality
    print("\n🧪 Testing lightweight capabilities...")
    
    test_input = MultiModalInput(
        text="Hello, this is a test of the lightweight multi-modal system.",
        metadata={"test": True}
    )
    
    try:
        result = chatbot.process_multimodal_input(test_input)
        print(f"✅ Test successful!")
        print(f"📝 Response: {result['response'][:200]}...")
        print(f"⏱️ Processing time: {result['processing_time']:.3f}s")
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    print(f"\n📊 System Status:")
    print(f"   Gemini: {'✅' if chatbot.gemini_client else '❌'}")
    print(f"   Qdrant: {'✅' if chatbot.qdrant_client else '❌'}")
    print(f"   Audio: {'✅' if HAS_AUDIO else '❌'}")
    print(f"   Images: {'✅' if HAS_PIL else '❌'}")
    print(f"   Documents: {'✅' if HAS_PDF else '❌'}")
    
    print(f"\n🎉 Lightweight Multi-Modal RAG Chatbot ready!")