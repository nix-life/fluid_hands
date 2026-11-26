@echo off
python -m venv .venv
if errorlevel 1 ( echo VENV FAILED & exit /b 1 )
.venv\Scripts\python -m pip install -r requirements.txt
if errorlevel 1 ( echo PIP FAILED & exit /b 1 )
.venv\Scripts\python tracker.py --download-only
if errorlevel 1 ( echo MODEL DOWNLOAD FAILED & exit /b 1 )
echo SETUP OK
