@echo off
cd /d "%~dp0"

echo ================================
echo Launching KP Cricket Dashboard
echo ================================
echo.

"C:\Users\ashis\anaconda3\Scripts\streamlit.exe" run app/app.py

pause