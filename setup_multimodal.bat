@echo off
echo ===============================================
echo   Multi-Modal RAG Chatbot Setup
echo ===============================================
echo.

REM Change to the script directory
cd /d "%~dp0"

echo 🔧 Setting up Multi-Modal RAG Chatbot...
echo Current directory: %CD%
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Create virtual environment if it doesn't exist
if not exist "multimodal_env" (
    echo 📦 Creating virtual environment...
    python -m venv multimodal_env
    echo ✅ Virtual environment created
) else (
    echo ✅ Virtual environment already exists
)

echo.
echo 🔄 Activating virtual environment...
call multimodal_env\Scripts\activate.bat

echo.
echo 📥 Upgrading pip...
python -m pip install --upgrade pip

echo.
echo 📦 Installing core dependencies...
pip install flask==2.3.3
pip install flask-cors==4.0.0
pip install python-dotenv==1.0.0

echo.
echo 🤖 Installing AI/ML libraries...
pip install google-generativeai==0.3.2
pip install sentence-transformers==2.2.2
pip install numpy==1.24.3
pip install scikit-learn==1.3.2
pip install qdrant-client==1.6.9

echo.
echo 🎵 Installing audio processing...
pip install SpeechRecognition==3.10.0
pip install pydub==0.25.1
echo Note: PyAudio installation may require Microsoft Visual C++ Build Tools
pip install pyaudio==0.2.11

echo.
echo 🎬 Installing video processing...
pip install opencv-python==4.8.1.78

echo.
echo 🖼️ Installing image processing...
pip install Pillow==10.1.0
pip install pytesseract==0.3.10

echo.
echo 📄 Installing document processing...
pip install PyPDF2==3.0.1
pip install python-docx==1.1.0

echo.
echo 🌐 Installing web framework...
pip install werkzeug==2.3.7

echo.
echo ✅ All dependencies installed successfully!
echo.

REM Create uploads directory
if not exist "uploads" mkdir uploads
echo ✅ Created uploads directory

echo.
echo 🔑 Checking environment configuration...
if exist ".env" (
    echo ✅ .env file found
) else (
    echo ⚠️ .env file not found
    echo Creating sample .env file...
    echo # Multi-Modal RAG Chatbot Configuration > .env
    echo GEMINI_API_KEY=your-gemini-api-key-here >> .env
    echo QDRANT_API_KEY=your-qdrant-api-key-here >> .env
    echo QDRANT_HOST=localhost >> .env
    echo QDRANT_PORT=6333 >> .env
    echo MAX_FILE_SIZE=100 >> .env
    echo UPLOAD_FOLDER=uploads >> .env
    echo ✅ Sample .env file created - please update with your API keys
)

echo.
echo 🧪 Testing installation...
python -c "
try:
    import flask
    import google.generativeai as genai
    import qdrant_client
    import sentence_transformers
    import numpy
    import cv2
    import PIL
    import speech_recognition
    import PyPDF2
    import docx
    print('✅ All core modules imported successfully!')
except ImportError as e:
    print(f'❌ Import error: {e}')
"

echo.
echo ===============================================
echo   Setup Complete! 🎉
echo ===============================================
echo.
echo 🚀 To start the Multi-Modal RAG Chatbot:
echo    1. Activate environment: multimodal_env\Scripts\activate.bat
echo    2. Update your .env file with API keys
echo    3. Run: python multimodal_web_interface.py
echo    4. Open browser: http://localhost:5000
echo.
echo 📋 Features available:
echo    ✅ Text Chat
echo    ✅ Audio Upload (Speech-to-Text)
echo    ✅ Video Upload (Analysis)
echo    ✅ Image Upload (OCR + AI Description)
echo    ✅ Document Upload (PDF, Word, TXT)
echo    ✅ Vector Search with Qdrant
echo    ✅ Google Gemini AI Integration
echo.
echo 💡 Don't forget to:
echo    - Add your GEMINI_API_KEY to .env file
echo    - Add your QDRANT_API_KEY to .env file (optional)
echo    - Install Tesseract OCR for image text extraction
echo.

pause