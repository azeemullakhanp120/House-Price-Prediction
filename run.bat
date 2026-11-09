@echo off
title EstateIQ House Price Prediction Web App
echo Starting EstateIQ House Price Prediction Web App...
cd /d "%~dp0"
if exist house_price_app (
    cd house_price_app
)
python app/app.py
pause

