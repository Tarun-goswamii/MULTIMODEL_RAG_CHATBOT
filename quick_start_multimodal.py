#!/usr/bin/env python3
"""
Quick Start Script for Multi-Modal RAG Chatbot
Tests all components and launches the web interface
"""

import os
import sys
import subprocess
from pathlib import Path

def test_imports():
    """Test all required imports"""
    print("🧪 Testing Multi-Modal Components...")
    print("=" * 50)
    
    tests = [
        ("Flask", "import flask"),
        ("Google Gemini", "import google.generativeai"),
        ("Qdrant", "import qdrant_client"),
        ("Sentence Transformers", "import sentence_transformers"),
        ("NumPy", "import numpy"),
        ("OpenCV (Video)", "import cv2"),
        ("PIL (Images)", "import PIL"),
        ("Speech Recognition", "import speech_recognition"),
        ("PyPDF2", "import PyPDF2"),
        ("Python-docx", "import docx"),
    ]
    
    passed = 0
    total = len(tests)
    
    for name, import_statement in tests:
        try:
            exec(import_statement)
            print(f"✅ {name}")
            passed += 1
        except ImportError:
            print(f"❌ {name} - Not installed")
        except Exception as e:
            print(f"⚠️ {name} - Error: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} components working")
    return passed >= (total * 0.7)  # 70% success rate

def check_environment():
    """Check environment configuration"""
    print("\n🔍 Checking Environment...")
    print("=" * 30)
    
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file found")
        
        with open(env_file, 'r') as f:
            content = f.read()
            
        if "GEMINI_API_KEY" in content and "your-gemini-api-key-here" not in content:
            print("✅ Gemini API key configured")
        else:
            print("⚠️ Gemini API key needs configuration")
            
        if "QDRANT_API_KEY" in content and "your-qdrant-api-key-here" not in content:
            print("✅ Qdrant API key configured")
        else:
            print("⚠️ Qdrant API key needs configuration (optional)")
    else:
        print("❌ .env file not found")
        return False
    
    return True

def test_multimodal_chatbot():
    """Test the multi-modal chatbot"""
    print("\n🤖 Testing Multi-Modal Chatbot...")
    print("=" * 35)
    
    try:
        # Test import
        from multimodal_rag_chatbot import MultiModalRAGChatbot, MultiModalInput
        print("✅ Multi-modal chatbot imported")
        
        # Test initialization
        chatbot = MultiModalRAGChatbot()
        print("✅ Chatbot initialized")
        
        # Test basic functionality
        test_input = MultiModalInput(text="Hello, test message")
        from multimodal_rag_chatbot import process_multimodal_sync
        
        result = process_multimodal_sync(chatbot, test_input)
        print("✅ Basic processing test passed")
        print(f"📝 Response preview: {result['response'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Chatbot test failed: {e}")
        return False

def launch_web_interface():
    """Launch the web interface"""
    print("\n🌐 Launching Web Interface...")
    print("=" * 30)
    
    try:
        # Check if web interface file exists
        web_file = Path("multimodal_web_interface.py")
        if not web_file.exists():
            print("❌ Web interface file not found")
            return False
        
        print("✅ Web interface file found")
        print("🚀 Starting Flask server...")
        print("📱 Open your browser to: http://localhost:5000")
        print("📋 Press Ctrl+C to stop the server")
        print()
        
        # Launch the web interface
        subprocess.run([sys.executable, "multimodal_web_interface.py"])
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Failed to launch web interface: {e}")
        return False
    
    return True

def create_demo_files():
    """Create demo files for testing"""
    print("\n📁 Creating Demo Files...")
    print("=" * 25)
    
    demo_dir = Path("demo_files")
    demo_dir.mkdir(exist_ok=True)
    
    # Create demo text file
    demo_text = """Multi-Modal RAG Chatbot Demo Document

This is a sample document to test the document processing capabilities of the multi-modal chatbot.

Key Features:
1. Text processing and understanding
2. Audio transcription and analysis  
3. Video frame extraction and analysis
4. Image OCR and description
5. Document parsing (PDF, Word, TXT)
6. Vector search with Qdrant
7. AI-powered responses with Google Gemini

The system can handle multiple input modalities simultaneously and provide comprehensive responses based on all the processed content.
"""
    
    with open(demo_dir / "demo_document.txt", "w") as f:
        f.write(demo_text)
    
    print("✅ Demo text document created")
    print(f"📄 Location: {demo_dir / 'demo_document.txt'}")
    
    # Create instructions
    instructions = """Multi-Modal Chatbot Demo Instructions

1. Open http://localhost:5000 in your browser
2. Try these test scenarios:

Text Chat:
- "What can you do?"
- "Explain vector search"
- "How do you process multiple modalities?"

File Upload Tests:
- Upload demo_document.txt (Document processing)
- Upload any image file (OCR + AI description)
- Upload any audio file (Speech-to-text)
- Upload any video file (Frame analysis)

Multi-Modal Tests:
- Upload an image + ask "What do you see?"
- Upload a document + ask "Summarize this"
- Combine text + file uploads for comprehensive analysis

The chatbot will process all inputs and provide intelligent responses!
"""
    
    with open(demo_dir / "demo_instructions.txt", "w") as f:
        f.write(instructions)
    
    print("✅ Demo instructions created")
    print(f"📋 Location: {demo_dir / 'demo_instructions.txt'}")

def main():
    """Main execution function"""
    print("🚀 Multi-Modal RAG Chatbot Quick Start")
    print("=" * 50)
    print()
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    # Run tests
    imports_ok = test_imports()
    env_ok = check_environment()
    
    if not imports_ok:
        print("\n❌ Some dependencies are missing!")
        print("💡 Run setup_multimodal.bat to install dependencies")
        input("Press Enter to continue anyway...")
    
    if not env_ok:
        print("\n⚠️ Environment configuration issues detected")
        print("💡 Update your .env file with proper API keys")
        input("Press Enter to continue anyway...")
    
    # Test chatbot
    chatbot_ok = test_multimodal_chatbot()
    
    if not chatbot_ok:
        print("\n❌ Chatbot test failed!")
        print("💡 Check your configuration and try again")
        input("Press Enter to continue anyway...")
    
    # Create demo files
    create_demo_files()
    
    # Launch web interface
    print("\n" + "=" * 50)
    print("🎉 READY TO LAUNCH!")
    print("=" * 50)
    
    choice = input("\nStart the web interface? (y/n): ").lower().strip()
    
    if choice in ['y', 'yes', '']:
        launch_web_interface()
    else:
        print("\n📋 To start manually:")
        print("   python multimodal_web_interface.py")
        print("\n📱 Then open: http://localhost:5000")

if __name__ == "__main__":
    main()