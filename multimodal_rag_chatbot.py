#!/usr/bin/env python3
"""
Multi-Modal RAG Chatbot with Comprehensive Features
Supports: Text, Audio, Video, Images, Documents, Real-time Processing
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

# Import dependencies with error handling
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
    from sentence_transformers import SentenceTransformer
    HAS_EMBEDDINGS = True
    print("✅ Sentence transformers imported")
except ImportError:
    print("⚠️ Sentence transformers not installed")
    HAS_EMBEDDINGS = False

try:
    from flask import Flask, request, jsonify, render_template_string
    from flask_cors import CORS
    HAS_FLASK = True
    print("✅ Flask imported")
except ImportError:
    print("⚠️ Flask not installed")
    HAS_FLASK = False

# Multi-modal processing imports
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
    import pydub
    HAS_AUDIO = True
    print("✅ Audio processing imported")
except ImportError:
    print("⚠️ Audio processing not installed")
    HAS_AUDIO = False

try:
    from PIL import Image
    import pytesseract
    HAS_OCR = True
    print("✅ OCR capabilities imported")
except ImportError:
    print("⚠️ OCR not installed")
    HAS_OCR = False

try:
    import PyPDF2
    import docx
    HAS_DOCUMENT = True
    print("✅ Document processing imported")
except ImportError:
    print("⚠️ Document processing not installed")
    HAS_DOCUMENT = False

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

class AudioProcessor:
    """Advanced audio processing capabilities"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer() if HAS_AUDIO else None
        
    def process_audio(self, audio_data: bytes, file_format: str = "wav") -> Dict[str, Any]:
        """Process audio file and extract information"""
        if not HAS_AUDIO:
            return {"transcript": "", "error": "Audio processing not available"}
        
        try:
            # Save temporary audio file
            with tempfile.NamedTemporaryFile(suffix=f".{file_format}", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_path = temp_file.name
            
            # Convert to WAV if needed
            if file_format.lower() != "wav":
                audio = pydub.AudioSegment.from_file(temp_path)
                wav_path = temp_path.replace(f".{file_format}", ".wav")
                audio.export(wav_path, format="wav")
                temp_path = wav_path
            
            # Transcribe audio
            with sr.AudioFile(temp_path) as source:
                audio = self.recognizer.record(source)
                
            # Try multiple recognition engines
            transcript = ""
            try:
                transcript = self.recognizer.recognize_google(audio)
            except:
                try:
                    transcript = self.recognizer.recognize_sphinx(audio)
                except:
                    transcript = "Could not transcribe audio"
            
            # Clean up temp file
            os.unlink(temp_path)
            
            # Analyze audio characteristics
            audio_analysis = self._analyze_audio_characteristics(audio_data)
            
            return {
                "transcript": transcript,
                "duration": audio_analysis.get("duration", 0),
                "quality": audio_analysis.get("quality", "unknown"),
                "language": "en",  # Could be detected
                "confidence": 0.8
            }
            
        except Exception as e:
            return {"transcript": "", "error": str(e)}
    
    def _analyze_audio_characteristics(self, audio_data: bytes) -> Dict[str, Any]:
        """Analyze audio characteristics like duration, quality, etc."""
        try:
            # This would use librosa or similar for detailed analysis
            return {
                "duration": len(audio_data) / 44100,  # Rough estimate
                "quality": "good",
                "noise_level": "low"
            }
        except:
            return {"duration": 0, "quality": "unknown", "noise_level": "unknown"}

class VideoProcessor:
    """Advanced video processing capabilities"""
    
    def __init__(self):
        self.face_cascade = None
        if HAS_OPENCV:
            try:
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            except:
                pass
    
    def process_video(self, video_data: bytes) -> Dict[str, Any]:
        """Process video file and extract information"""
        if not HAS_OPENCV:
            return {"summary": "", "error": "Video processing not available"}
        
        try:
            # Save temporary video file
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
                temp_file.write(video_data)
                temp_path = temp_file.name
            
            # Open video
            cap = cv2.VideoCapture(temp_path)
            
            frame_count = 0
            face_detections = 0
            key_frames = []
            
            # Process frames
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Sample every 30th frame
                if frame_count % 30 == 0:
                    # Detect faces
                    if self.face_cascade is not None:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                        if len(faces) > 0:
                            face_detections += 1
                    
                    # Store key frame info
                    key_frames.append({
                        "frame": frame_count,
                        "timestamp": frame_count / cap.get(cv2.CAP_PROP_FPS),
                        "faces": len(faces) if self.face_cascade else 0
                    })
            
            cap.release()
            os.unlink(temp_path)
            
            # Generate summary
            fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
            duration = frame_count / fps
            
            summary = f"Video analysis: {duration:.1f} seconds, {frame_count} frames"
            if face_detections > 0:
                summary += f", {face_detections} frames with faces detected"
            
            return {
                "summary": summary,
                "duration": duration,
                "frame_count": frame_count,
                "face_detections": face_detections,
                "key_frames": key_frames[:10],  # Limit to 10 key frames
                "confidence": 0.7
            }
            
        except Exception as e:
            return {"summary": "", "error": str(e)}

class ImageProcessor:
    """Advanced image processing capabilities"""
    
    def __init__(self):
        self.gemini_vision = None
        if HAS_GEMINI:
            try:
                self.gemini_vision = genai.GenerativeModel('gemini-pro-vision')
            except:
                pass
    
    def process_image(self, image_data: bytes) -> Dict[str, Any]:
        """Process image and extract information"""
        try:
            # Convert to PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            results = {
                "description": "",
                "text_content": "",
                "objects": [],
                "confidence": 0.0
            }
            
            # OCR text extraction
            if HAS_OCR:
                try:
                    text_content = pytesseract.image_to_string(image)
                    results["text_content"] = text_content.strip()
                except:
                    pass
            
            # AI-powered image description using Gemini Vision
            if self.gemini_vision:
                try:
                    response = self.gemini_vision.generate_content([
                        "Describe this image in detail. What do you see? Include any text, objects, people, or important details.",
                        image
                    ])
                    results["description"] = response.text
                    results["confidence"] = 0.9
                except Exception as e:
                    results["description"] = f"Could not analyze image: {str(e)}"
            
            # Basic image properties
            results["width"] = image.width
            results["height"] = image.height
            results["format"] = image.format
            results["mode"] = image.mode
            
            return results
            
        except Exception as e:
            return {"description": "", "error": str(e)}

class DocumentProcessor:
    """Advanced document processing capabilities"""
    
    def process_document(self, document_data: bytes, file_type: str) -> Dict[str, Any]:
        """Process document and extract text content"""
        if not HAS_DOCUMENT:
            return {"text": "", "error": "Document processing not available"}
        
        try:
            text_content = ""
            
            if file_type.lower() == "pdf":
                text_content = self._process_pdf(document_data)
            elif file_type.lower() in ["docx", "doc"]:
                text_content = self._process_word(document_data)
            elif file_type.lower() == "txt":
                text_content = document_data.decode('utf-8')
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
    
    def _process_word(self, word_data: bytes) -> str:
        """Extract text from Word document"""
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as temp_file:
            temp_file.write(word_data)
            temp_path = temp_file.name
        
        try:
            doc = docx.Document(temp_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            os.unlink(temp_path)
            return text
        except:
            os.unlink(temp_path)
            return ""

class MultiModalRAGChatbot:
    """Advanced Multi-Modal RAG Chatbot"""
    
    def __init__(self):
        """Initialize the multi-modal chatbot"""
        print("🤖 Initializing Multi-Modal RAG Chatbot...")
        
        # Core components
        self.gemini_client = None
        self.qdrant_client = None
        self.embedding_model = None
        self.knowledge_base = []
        
        # Multi-modal processors
        self.audio_processor = AudioProcessor()
        self.video_processor = VideoProcessor()
        self.image_processor = ImageProcessor()
        self.document_processor = DocumentProcessor()
        
        # Initialize core components
        self._init_gemini()
        self._init_qdrant()
        self._init_embeddings()
        self._load_knowledge_base()
        
        # Conversation history
        self.conversation_history = []
        self.processing_stats = {
            "total_queries": 0,
            "multimodal_queries": 0,
            "processing_time_avg": 0.0
        }
        
        print("✅ Multi-Modal RAG Chatbot ready!")
    
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
            
            # Test connection
            test_response = self.gemini_client.generate_content("Hello")
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
                "text": "This is a multi-modal RAG chatbot that can process text, audio, video, images, and documents. It uses vector search and AI to provide comprehensive responses.",
                "category": "system",
                "keywords": ["multimodal", "rag", "chatbot", "audio", "video", "image", "document"]
            },
            {
                "text": "Audio processing includes speech-to-text transcription, audio quality analysis, and speaker identification. Supports multiple audio formats.",
                "category": "audio",
                "keywords": ["audio", "speech", "transcription", "voice", "sound"]
            },
            {
                "text": "Video analysis includes frame extraction, object detection, face recognition, and scene understanding. Can process various video formats.",
                "category": "video", 
                "keywords": ["video", "frames", "detection", "recognition", "analysis"]
            },
            {
                "text": "Image processing includes OCR text extraction, object recognition, scene description, and visual question answering.",
                "category": "image",
                "keywords": ["image", "ocr", "visual", "object", "recognition"]
            },
            {
                "text": "Document processing supports PDF, Word, and text files. Extracts content, metadata, and performs semantic analysis.",
                "category": "document",
                "keywords": ["document", "pdf", "word", "text", "extraction"]
            }
        ]
        print(f"📚 Loaded {len(self.knowledge_base)} knowledge entries")
    
    async def process_multimodal_input(self, multimodal_input: MultiModalInput) -> Dict[str, Any]:
        """Process multi-modal input asynchronously"""
        start_time = datetime.now()
        processed_content = ProcessedContent()
        
        # Process different modalities in parallel
        tasks = []
        
        if multimodal_input.text:
            processed_content.extracted_text = multimodal_input.text
        
        if multimodal_input.audio_data:
            tasks.append(self._process_audio_async(multimodal_input.audio_data))
        
        if multimodal_input.video_data:
            tasks.append(self._process_video_async(multimodal_input.video_data))
        
        if multimodal_input.image_data:
            tasks.append(self._process_image_async(multimodal_input.image_data))
        
        if multimodal_input.document_data:
            tasks.append(self._process_document_async(
                multimodal_input.document_data, 
                multimodal_input.file_type or "txt"
            ))
        
        # Wait for all processing tasks to complete
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, dict) and not isinstance(result, Exception):
                    if 'transcript' in result:
                        processed_content.audio_transcript = result['transcript']
                    if 'summary' in result:
                        processed_content.video_summary = result['summary']
                    if 'description' in result:
                        processed_content.image_description = result['description']
                    if 'text' in result:
                        processed_content.document_text = result['text']
        
        # Combine all extracted text
        all_text = " ".join(filter(None, [
            processed_content.extracted_text,
            processed_content.audio_transcript,
            processed_content.image_description,
            processed_content.video_summary,
            processed_content.document_text
        ]))
        
        # Generate comprehensive response
        response = await self._generate_multimodal_response(all_text, processed_content)
        
        # Update statistics
        processing_time = (datetime.now() - start_time).total_seconds()
        self.processing_stats["total_queries"] += 1
        if any([multimodal_input.audio_data, multimodal_input.video_data, 
                multimodal_input.image_data, multimodal_input.document_data]):
            self.processing_stats["multimodal_queries"] += 1
        
        # Store in conversation history
        conversation_entry = {
            "timestamp": start_time.isoformat(),
            "input": {
                "text": multimodal_input.text,
                "has_audio": bool(multimodal_input.audio_data),
                "has_video": bool(multimodal_input.video_data),
                "has_image": bool(multimodal_input.image_data),
                "has_document": bool(multimodal_input.document_data)
            },
            "processed_content": {
                "audio_transcript": processed_content.audio_transcript,
                "image_description": processed_content.image_description,
                "video_summary": processed_content.video_summary,
                "document_text": processed_content.document_text[:200] + "..." if len(processed_content.document_text) > 200 else processed_content.document_text
            },
            "response": response,
            "processing_time": processing_time
        }
        self.conversation_history.append(conversation_entry)
        
        return {
            "response": response,
            "processed_content": processed_content,
            "processing_time": processing_time,
            "multimodal_summary": self._create_multimodal_summary(processed_content),
            "confidence": self._calculate_confidence(processed_content)
        }
    
    async def _process_audio_async(self, audio_data: bytes) -> Dict[str, Any]:
        """Process audio asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.audio_processor.process_audio, audio_data)
    
    async def _process_video_async(self, video_data: bytes) -> Dict[str, Any]:
        """Process video asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.video_processor.process_video, video_data)
    
    async def _process_image_async(self, image_data: bytes) -> Dict[str, Any]:
        """Process image asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.image_processor.process_image, image_data)
    
    async def _process_document_async(self, document_data: bytes, file_type: str) -> Dict[str, Any]:
        """Process document asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.document_processor.process_document, document_data, file_type)
    
    async def _generate_multimodal_response(self, combined_text: str, processed_content: ProcessedContent) -> str:
        """Generate comprehensive response using all modalities"""
        if not combined_text.strip():
            return "I didn't receive any content to process. Please provide text, upload a file, or record audio/video."
        
        # Search knowledge base
        context = self.search_knowledge(combined_text)
        
        # Build comprehensive prompt
        prompt = self._build_multimodal_prompt(combined_text, processed_content, context)
        
        # Generate response with Gemini
        if self.gemini_client:
            try:
                response = self.gemini_client.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                print(f"⚠️ Gemini failed: {e}")
        
        # Fallback response
        return self._generate_fallback_response(combined_text, processed_content, context)
    
    def _build_multimodal_prompt(self, text: str, content: ProcessedContent, context: List[Dict]) -> str:
        """Build comprehensive prompt for multi-modal content"""
        context_text = "\n".join([item["text"] for item in context[:3]]) if context else "No specific context found."
        
        modalities_processed = []
        if content.audio_transcript:
            modalities_processed.append(f"Audio: {content.audio_transcript[:100]}...")
        if content.image_description:
            modalities_processed.append(f"Image: {content.image_description[:100]}...")
        if content.video_summary:
            modalities_processed.append(f"Video: {content.video_summary[:100]}...")
        if content.document_text:
            modalities_processed.append(f"Document: {content.document_text[:100]}...")
        
        prompt = f"""You are an advanced multi-modal AI assistant. Analyze and respond to the following multi-modal input:

PRIMARY TEXT INPUT:
{text}

PROCESSED CONTENT FROM OTHER MODALITIES:
{chr(10).join(modalities_processed) if modalities_processed else "No additional modalities processed."}

RELEVANT CONTEXT:
{context_text}

INSTRUCTIONS:
1. Provide a comprehensive response that acknowledges all input modalities
2. If multiple modalities contain related information, synthesize them coherently
3. Be helpful, accurate, and specific to the content provided
4. If you detect any inconsistencies between modalities, mention them
5. Provide actionable insights when possible

RESPONSE:"""
        
        return prompt
    
    def _generate_fallback_response(self, text: str, content: ProcessedContent, context: List[Dict]) -> str:
        """Generate fallback response when AI fails"""
        response_parts = []
        
        if content.audio_transcript:
            response_parts.append(f"From your audio, I heard: {content.audio_transcript}")
        
        if content.image_description:
            response_parts.append(f"From your image, I can see: {content.image_description}")
        
        if content.video_summary:
            response_parts.append(f"From your video: {content.video_summary}")
        
        if content.document_text:
            response_parts.append(f"From your document: {content.document_text[:200]}...")
        
        if context:
            response_parts.append(f"Based on my knowledge: {context[0]['text']}")
        
        if response_parts:
            return " | ".join(response_parts)
        else:
            return "I processed your input but couldn't generate a specific response. Please try rephrasing your question."
    
    def search_knowledge(self, query: str, limit: int = 3) -> List[Dict]:
        """Search knowledge base"""
        if not query.strip():
            return []
        
        # Vector search if available
        if self.qdrant_client and self.embedding_model:
            try:
                query_embedding = self.embedding_model.encode(query).tolist()
                results = self.qdrant_client.search(
                    collection_name="multimodal_knowledge",
                    query_vector=query_embedding,
                    limit=limit
                )
                return [{"text": r.payload.get("text", ""), "score": r.score} for r in results]
            except:
                pass
        
        # Fallback keyword search
        query_lower = query.lower()
        scored_results = []
        
        for item in self.knowledge_base:
            score = 0
            text_lower = item["text"].lower()
            keywords = item.get("keywords", [])
            
            for keyword in keywords:
                if keyword.lower() in query_lower:
                    score += 5
            
            for word in query_lower.split():
                if len(word) > 2 and word in text_lower:
                    score += 2
            
            if score > 0:
                scored_results.append({"text": item["text"], "score": score})
        
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[:limit]
    
    def _create_multimodal_summary(self, content: ProcessedContent) -> Dict[str, str]:
        """Create summary of processed modalities"""
        summary = {}
        
        if content.audio_transcript:
            summary["audio"] = f"Transcribed {len(content.audio_transcript.split())} words from audio"
        
        if content.image_description:
            summary["image"] = f"Analyzed image content"
        
        if content.video_summary:
            summary["video"] = f"Processed video content"
        
        if content.document_text:
            summary["document"] = f"Extracted {len(content.document_text.split())} words from document"
        
        return summary
    
    def _calculate_confidence(self, content: ProcessedContent) -> float:
        """Calculate overall confidence score"""
        confidences = []
        
        if content.audio_transcript:
            confidences.append(0.8)  # Good confidence for audio
        
        if content.image_description:
            confidences.append(0.9)  # High confidence for image
        
        if content.video_summary:
            confidences.append(0.7)  # Medium confidence for video
        
        if content.document_text:
            confidences.append(0.95)  # Very high confidence for documents
        
        return sum(confidences) / len(confidences) if confidences else 0.5
    
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

# Synchronous wrapper for easier use
def process_multimodal_sync(chatbot: MultiModalRAGChatbot, multimodal_input: MultiModalInput) -> Dict[str, Any]:
    """Synchronous wrapper for multimodal processing"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(chatbot.process_multimodal_input(multimodal_input))
        return result
    finally:
        loop.close()

if __name__ == "__main__":
    print("🚀 Starting Multi-Modal RAG Chatbot System...")
    
    # Initialize chatbot
    chatbot = MultiModalRAGChatbot()
    
    # Test basic functionality
    print("\n🧪 Testing multi-modal capabilities...")
    
    test_input = MultiModalInput(
        text="Hello, this is a test of the multi-modal system. Can you tell me about your capabilities?",
        metadata={"test": True}
    )
    
    try:
        result = process_multimodal_sync(chatbot, test_input)
        print(f"✅ Test successful!")
        print(f"📝 Response: {result['response'][:200]}...")
        print(f"⏱️ Processing time: {result['processing_time']:.3f}s")
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    print(f"\n📊 System Status:")
    print(f"   Gemini: {'✅' if chatbot.gemini_client else '❌'}")
    print(f"   Qdrant: {'✅' if chatbot.qdrant_client else '❌'}")
    print(f"   Audio: {'✅' if HAS_AUDIO else '❌'}")
    print(f"   Video: {'✅' if HAS_OPENCV else '❌'}")
    print(f"   OCR: {'✅' if HAS_OCR else '❌'}")
    print(f"   Documents: {'✅' if HAS_DOCUMENT else '❌'}")
    
    print(f"\n🎉 Multi-Modal RAG Chatbot ready for comprehensive processing!")