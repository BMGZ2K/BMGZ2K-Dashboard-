import ccxt
import pandas as pd
import config
from data import MarketData
from strategy import MomentumStrategy
from risk import RiskManager
from backtester import Backtester
import itertools

def run_optimization():
    print("🧬 Starting Genetic Optimization...")
    
    # 1. Fetch Historical Data (Larger set for backtest)
    exchange = ccxt.binance({'options': {'defaultType': 'future'}}) # Public data doesn't need keys
    market_data = MarketData(exchange)
    print(f"📥 Fetching 1000 candles for {config.SYMBOL}...")
    df = market_data.fetch_ohlcv(config.SYMBOL, timeframe=config.TIMEFRAME, limit=1000)
    
    if df.empty:
        print("❌ Failed to fetch data.")
        return

    # 2. Define Parameter Grid
    ema_shorts = [5, 9, 12]
    ema_longs = [21, 26, 50]
    rsi_periods = [14]
    
    best_pnl = -float('inf')
    best_params = None
    best_metrics = None
    
    risk_manager = RiskManager(risk_per_trade=0.02, leverage=10)
    
    print(f"🧪 Testing {len(ema_shorts) * len(ema_longs) * len(rsi_periods)} combinations...")
    
    for short, long, rsi in itertools.product(ema_shorts, ema_longs, rsi_periods):
        if short >= long: continue # Skip invalid combinations
        
        strategy = MomentumStrategy(ema_short=short, ema_long=long, rsi_period=rsi)
        backtester = Backtester(strategy, risk_manager, initial_balance=10000)
        
        metrics = backtester.run(df.copy())
        
        print(f"   Params: {short}/{long}/{rsi} -> PnL: {metrics['Total PnL']} | WinRate: {metrics['Win Rate']}%")
        
        if metrics['Total PnL'] > best_pnl:
            best_pnl = metrics['Total PnL']
            best_params = (short, long, rsi)
            best_metrics = metrics

    print("\n🏆 OPTIMIZATION RESULTS 🏆")
    print(f"Best Parameters: EMA_S={best_params[0]}, EMA_L={best_params[1]}, RSI={best_params[2]}")
    print(f"Metrics: {best_metrics}")
    
    # Save best params to a file or just print for now
    with open('best_params.txt', 'w') as f:
        f.write(f"EMA_SHORT={best_params[0]}\n")
        f.write(f"EMA_LONG={best_params[1]}\n")
        f.write(f"RSI_PERIOD={best_params[2]}\n")

if __name__ == "__main__":
    run_optimization()
