@echo off
echo ==========================================
echo   SIMPLE SETUP - Multi-Modal Dashboard
echo ==========================================
echo.

cd /d "%~dp0"

echo 🔧 Installing only essential packages...
echo.

REM Use existing environment or create new one
if exist "lightweight_env\Scripts\activate.bat" (
    call lightweight_env\Scripts\activate.bat
    echo ✅ Using existing environment
) else (
    echo 📦 Creating new environment...
    python -m venv simple_env
    call simple_env\Scripts\activate.bat
)

echo.
echo 📥 Installing core packages...
pip install flask==2.3.3
pip install flask-cors==4.0.0
pip install python-dotenv==1.0.0

echo.
echo 📥 Installing Google Gemini (may take a moment)...
pip install google-generativeai==0.3.2

echo.
echo 📥 Installing Pillow for images...
pip install Pillow==10.1.0

echo.
echo 🧪 Testing installation...
python -c "
import flask
import google.generativeai
from PIL import Image
print('✅ All packages working!')
"

echo.
echo ========================================
echo   🎉 SETUP COMPLETE!
echo ========================================
echo.
echo 🚀 To start the dashboard:
echo    python simple_dashboard.py
echo.
echo 📱 Then open: http://localhost:5000
echo.

choice /C YN /M "Start the simple dashboard now? (Y/N)"
if errorlevel 2 goto end
if errorlevel 1 goto start

:start
echo 🌐 Starting simple dashboard...
python simple_dashboard.py

:end
echo.
echo 👋 Ready to go!
pause