@echo off
title HR Policy Chatbot — Install Dependencies
echo.
echo  ======================================================
echo   HR Policy Chatbot  ^|  Installing Dependencies
echo  ======================================================
echo.

pip install -r requirements.txt

echo.
echo  ======================================================
echo   Installation complete!
echo.
echo   Next steps:
echo     1. Copy .env.example to .env
echo     2. Add your OPENAI_API_KEY in .env
echo     3. Run start_backend.bat   (in one terminal)
echo     4. Run start_frontend.bat  (in another terminal)
echo     5. Open http://localhost:8501 in your browser
echo  ======================================================
echo.
pause
