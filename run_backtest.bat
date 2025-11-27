@echo off
echo ============================================
echo   Backtest Rapido
echo ============================================
echo.

cd /d "%~dp0"

echo Executando Backtest...
python backtest.py --symbol BTC/USDT --timeframe 1h --days 60

pause
