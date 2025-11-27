@echo off
title Trading System
echo ============================================
echo   SISTEMA DE TRADING COMPLETO
echo ============================================
echo.
echo Iniciando Dashboard (http://localhost:5000)...
start "Dashboard" cmd /c "python dashboard.py"

timeout /t 3 /nobreak >nul

echo Iniciando Bot de Trading...
start "Trading Bot" cmd /c "python bot.py"

echo.
echo ============================================
echo   Sistema iniciado!
echo   - Dashboard: http://localhost:5000
echo   - Bot: Executando em background
echo ============================================
echo.
echo Pressione qualquer tecla para fechar esta janela...
pause >nul
