@echo off
title HR Policy Chatbot — Frontend
echo.
echo  ======================================================
echo   HR Policy Chatbot  ^|  Streamlit Frontend
echo  ======================================================
echo.

cd frontend
echo  Starting Streamlit on http://localhost:8501 ...
echo  Press Ctrl+C to stop.
echo.
streamlit run app.py --server.port 8501 --server.headless false

pause
