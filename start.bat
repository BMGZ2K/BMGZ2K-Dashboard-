@echo off
title Trading System
cd /d "%~dp0"

echo ============================================
echo   TRADING SYSTEM
echo ============================================
echo.
echo [1] Iniciar Dashboard (localhost:5000)
echo [2] Iniciar Bot
echo [3] Iniciar Ambos
echo [4] Git Push
echo [5] Git Pull
echo [6] Evoluir Estrategia
echo [0] Sair
echo.
set /p choice="Escolha: "

if "%choice%"=="1" goto dashboard
if "%choice%"=="2" goto bot
if "%choice%"=="3" goto both
if "%choice%"=="4" goto push
if "%choice%"=="5" goto pull
if "%choice%"=="6" goto evolve
if "%choice%"=="0" goto end
goto end

:dashboard
start "Dashboard" cmd /k python dashboard.py
goto end

:bot
start "Bot" cmd /k python bot.py
goto end

:both
start "Dashboard" cmd /k python dashboard.py
timeout /t 3 /nobreak > nul
start "Bot" cmd /k python bot.py
goto end

:push
set /p msg="Mensagem do commit: "
python scripts/git_push.py "%msg%"
pause
goto end

:pull
python scripts/git_pull.py
pause
goto end

:evolve
python auto_evolve.py
pause
goto end

:end
