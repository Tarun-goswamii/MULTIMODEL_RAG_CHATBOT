# Quick Setup Guide for Multi-Task LLM Interview System

## Option 1: Automated Setup (Recommended)

### Run the Complete Setup Script:
1. Right-click PowerShell and select "Run as Administrator"
2. Navigate to your project:
   ```powershell
   cd "C:\Users\tarun\OneDrive\Desktop\MULTI_TASK_LLM"
   ```
3. Run the setup script:
   ```powershell
   .\setup_complete_system.ps1
   ```

### If Docker Issues:
```powershell
.\setup_complete_system.ps1 -MockQdrant
```

## Option 2: Manual Setup

### Step 1: Setup Docker & Qdrant
```powershell
# Start Docker Desktop first!
docker run -d --name qdrant-interview -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### Step 2: Setup Python Backend
```powershell
cd "C:\Users\tarun\OneDrive\Desktop\MULTI_TASK_LLM"
python -m venv venv_backend
.\venv_backend\Scripts\Activate.ps1
pip install -r backend\requirements.txt
cd backend
python qdrant_setup.py
```

### Step 3: Setup Frontend
```powershell
cd ..\venv\frontend\MULTI_TASK_LLM
npm install
```

### Step 4: Start System
```powershell
# Terminal 1 - Backend
cd C:\Users\tarun\OneDrive\Desktop\MULTI_TASK_LLM\backend
python app.py

# Terminal 2 - Frontend  
cd C:\Users\tarun\OneDrive\Desktop\MULTI_TASK_LLM\venv\frontend\MULTI_TASK_LLM
npm run dev
```

## Access Points:
- **Interview System**: http://localhost:3000
- **Backend API**: http://localhost:8000  
- **Qdrant Dashboard**: http://localhost:6333/dashboard

## Troubleshooting:
1. **Docker Issues**: Use `-MockQdrant` flag
2. **Permission Issues**: Run PowerShell as Administrator
3. **Port Conflicts**: Stop other services on ports 3000, 8000, 6333

## Quick Commands:
```powershell
# Check setup status
python backend\check_setup_status.py

# Start only backend
.\start_backend.bat

# Start only frontend  
.\start_frontend.bat

# Start complete system
.\start_system.bat
```