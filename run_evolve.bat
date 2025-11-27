@echo off
echo ============================================================
echo    SISTEMA DE EVOLUCAO AUTOMATICA
echo ============================================================
echo.

cd /d "%~dp0"

REM Verificar Python
python --version 2>nul
if errorlevel 1 (
    echo ERRO: Python nao encontrado!
    pause
    exit /b 1
)

echo.
echo Escolha uma opcao:
echo   1. Teste rapido (5 simbolos, 30 dias, 2 geracoes)
echo   2. Evolucao completa (15 simbolos, 90 dias, 5 geracoes)
echo   3. Apenas WFO Backtest (sem evolucao)
echo.

set /p opcao="Opcao [1/2/3]: "

if "%opcao%"=="1" (
    echo.
    echo Iniciando teste rapido...
    python auto_evolve.py --quick
) else if "%opcao%"=="2" (
    echo.
    echo Iniciando evolucao completa...
    python auto_evolve.py --full
) else if "%opcao%"=="3" (
    echo.
    echo Iniciando WFO Backtest...
    python portfolio_wfo.py
) else (
    echo Opcao invalida!
)

echo.
echo ============================================================
echo    PROCESSO FINALIZADO
echo ============================================================
pause
