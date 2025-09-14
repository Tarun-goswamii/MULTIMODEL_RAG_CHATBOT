@echo off
echo ========================================
echo   QUICK START - Enhanced Interview System
echo ========================================
echo.

cd /d "%~dp0"

REM Check if we're in the right directory
if not exist "backend" (
    echo ❌ Error: backend directory not found
    echo Make sure you're running this from the project root directory
    pause
    exit /b 1
)

echo 📂 Project directory: %CD%
echo.

REM Activate environment if it exists
if exist "lightweight_env\Scripts\activate.bat" (
    call lightweight_env\Scripts\activate.bat
    echo ✅ Environment activated
) else (
    echo ⚠️ No virtual environment found, using system Python
)

echo.
echo 📦 Installing required packages...

REM Install only essential packages
pip install flask==2.3.3
pip install flask-cors==4.0.0
pip install python-dotenv==1.0.0

echo.
echo 🧪 Testing the application...

REM Check if enhanced_app.py exists in backend
if exist "backend\enhanced_app.py" (
    echo ✅ Enhanced app found in backend directory
) else (
    echo ❌ Enhanced app not found
    echo Please make sure enhanced_app.py is in the backend directory
    pause
    exit /b 1
)

echo.
echo 🚀 Starting Enhanced Interview System...
echo.
echo 📋 Features available:
echo   ✅ Interview chat interface
echo   ✅ Knowledge base upload (text/audio/image)
echo   ✅ Semantic search across content
echo   ✅ Enhanced chat with context retrieval
echo.

cd backend
python enhanced_app.py

echo.
echo 📱 If the app started successfully, visit:
echo    http://localhost:8000
echo.
pause