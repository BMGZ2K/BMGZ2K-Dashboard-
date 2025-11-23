@echo off
title Binance Bot Controller
color 0A
echo ==================================================
echo       🚀 INICIANDO SISTEMA DE TRADING AUTOMATICO
echo ==================================================
echo.

echo 1. Iniciando Bot de Trading (Minimizado)...
start /min "Trading Bot" cmd /c run_bot.bat

echo 2. Iniciando Motor de Evolucao (Minimizado)...
start /min "Evolution Engine" cmd /c run_evolution.bat

echo 3. Iniciando Dashboard (Minimizado)...
start /min "Dashboard" cmd /c run_dashboard.bat

echo.
echo ✅ SISTEMA ONLINE!
echo.
echo O Dashboard vai abrir no seu navegador em 5 segundos...
timeout /t 5 >nul
start http://localhost:8501

echo.
echo ==================================================
echo    TUDO PRONTO! AGORA ACOMPANHE PELO NAVEGADOR.
echo    (Mantenha esta janela aberta ou minimizada)
echo ==================================================
echo.
echo Para desligar tudo, execute o arquivo 'stop_all.bat'
pause
