@echo off
echo ============================================
echo   US Small Caps Screener
echo ============================================
echo.
cd /d "%~dp0"
C:\Users\Ahmad\miniconda3\python.exe main.py

echo.
echo ============================================
echo   Done. Check the output folder for:
echo     - stage1_shortlist.xlsx (data)
echo     - ai_analysis_prompt.txt (paste into AI)
echo ============================================
pause
