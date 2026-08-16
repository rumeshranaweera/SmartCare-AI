@echo off
title SmartCare AI - Appointment No-Show Predictor
echo Installing/checking requirements...
python -m pip install -r requirements.txt
echo.
echo Starting SmartCare AI...
python -m streamlit run streamlit_app.py
pause
