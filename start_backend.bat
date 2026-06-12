@echo off
title HR Policy Chatbot — Backend
echo.
echo  ======================================================
echo   HR Policy Chatbot  ^|  FastAPI Backend
echo  ======================================================
echo.

REM Check for .env file
if not exist ".env" (
    echo  [WARN] .env file not found!
    echo  Copy .env.example to .env and add your OPENAI_API_KEY.
    echo.
    pause
    exit /b 1
)

cd backend
echo  Starting FastAPI on http://localhost:8000 ...
echo  Press Ctrl+C to stop.
echo.
python main.py

pause
