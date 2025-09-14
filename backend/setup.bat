@echo off
echo Setting up Multi-Task LLM Interview System...

REM Create virtual environment
python -m venv interview_env

REM Activate virtual environment
call interview_env\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip

REM Install requirements
pip install -r requirements.txt

echo.
echo Setup completed successfully!
echo.
echo To start the system:
echo 1. Run: interview_env\Scripts\activate.bat
echo 2. Run: python app.py
echo.
pause
