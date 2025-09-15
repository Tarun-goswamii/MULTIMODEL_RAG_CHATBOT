# MULTIMODEL_RAG_CHATBOT
🚀 Complete Command Prompt Setup

You now have everything you need! Here's exactly what to do:

Option 1: Automated Setup (Easiest)

Right-click Command Prompt and select "Run as administrator"
Navigate to your project:
cd /d "C:\Users\tarun\OneDrive\Desktop\MULTI_TASK_LLM"
Run the complete setup:
setup_complete_system.bat
Option 3: Setup WITHOUT Docker (Recommended for Docker Issues)

Since you're getting Docker errors, use this approach:

REM 1. Navigate to project
cd /d "C:\Users\tarun\OneDrive\Desktop\MULTI_TASK_LLM"

REM 2. Create Python environment
python -m venv venv_backend
call venv_backend\Scripts\activate.bat
python -m pip install --upgrade pip

REM 3. Install basic packages (skip problematic ones)
pip install flask flask-cors numpy scikit-learn requests python-dotenv joblib

REM 4. Setup mock vector database (no Docker needed)
cd backend
python qdrant_setup_mock.py
cd ..
Option 2: Quick Manual Commands

If you prefer step-by-step:

REM 1. Navigate to project
cd /d "C:\Users\tarun\OneDrive\Desktop\MULTI_TASK_LLM"

REM 2. Create Python environment
python -m venv venv_backend
call venv_backend\Scripts\activate.bat
python -m pip install --upgrade pip

REM 3. Install Python packages
pip install flask flask-cors flask-socketio numpy scikit-learn requests python-dotenv joblib
pip install qdrant-client sentence-transformers

REM 4. Start Docker (if available) - Skip if getting Docker errors
REM docker run -d --name qdrant-interview -p 6333:6333 -p 6334:6334 qdrant/qdrant
REM If Docker error, use mock setup instead:
REM cd backend
REM python qdrant_setup_mock.py
REM cd ..

REM 5. Setup database
cd backend
python qdrant_setup.py
cd ..

REM 6. Install frontend (if Node.js available)
cd venv\frontend\MULTI_TASK_LLM
npm install
cd ..\..\..
Start Your System:

Complete system (recommended):

start_system.bat
Or start individually:

REM Backend only
start_backend.bat

REM Frontend only  
start_frontend.bat
Manual start commands:

Terminal 1 - Backend:

cd /d "C:\Users\tarun\OneDrive\Desktop\MULTI_TASK_LLM"
call venv_backend\Scripts\activate.bat
cd backend
python app.py
Terminal 2 - Frontend:

cd /d "C:\Users\tarun\OneDrive\Desktop\MULTI_TASK_LLM\venv\frontend\MULTI_TASK_LLM"
npm run dev
Access Your System:

Interview Interface: http://localhost:3000
Backend API: http://localhost:8000
Qdrant Dashboard: http://localhost:6333/dashboard
Check Status:

cd /d "C:\Users\tarun\OneDrive\Desktop\MULTI_TASK_LLM\backend"
python check_setup_status.py
Troubleshooting:

If Docker doesn't work:

cd backend
python qdrant_setup_mock.py
If you get permission errors:

Run Command Prompt as Administrator
If packages fail to install:

pip install --upgrade pip
pip install flask flask-cors numpy scikit-learn requests --no-cache-dir
The automated setup script will:

✅ Create all directories
✅ Set up Python virtual environment
✅ Install all dependencies
✅ Start Qdrant (or use mock if Docker issues)
✅ Create collections and populate data
✅ Set up frontend dependencies
✅ Create startup scripts
✅ Test the entire system
Just run setup_complete_system.bat and it will handle everything automatically!

Would you like to try the automated setup first, or do you prefer the manual step-by-step approach?