import ccxt
import pandas as pd
import config
import json
import time
import itertools
import logging
import sys
from data import MarketData
from strategy import HybridStrategy
from risk import RiskManager
from backtester import Backtester

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evolution.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()

def run_evolution():
    logger.info("🧬 Starting Evolution Engine...")
    
    exchange = ccxt.binance({'options': {'defaultType': 'future'}})
    market_data = MarketData(exchange)
    risk_manager = RiskManager(risk_per_trade=0.02, leverage=10)
    
    while True:
        try:
            logger.info(f"📥 Fetching 1000 candles for optimization...")
            df = market_data.fetch_ohlcv(config.SYMBOL, timeframe=config.TIMEFRAME, limit=1000)
            
            if df.empty:
                logger.warning("❌ Failed to fetch data. Retrying in 60s...")
                time.sleep(60)
                continue

            # Parameter Grid (Expanded)
            ema_shorts = [5, 9, 12]
            ema_longs = [21, 34]
            rsi_periods = [14]
            adx_thresholds = [20, 25]
            bb_windows = [20, 30]
            
            best_pnl = -float('inf')
            best_params = None
            
            # Run Optimization
            for short, long, rsi, adx, bb in itertools.product(ema_shorts, ema_longs, rsi_periods, adx_thresholds, bb_windows):
                if short >= long: continue
                
                strategy = HybridStrategy(
                    ema_short=short, 
                    ema_long=long, 
                    rsi_period=rsi, 
                    adx_threshold=adx,
                    bb_window=bb
                )
                backtester = Backtester(strategy, risk_manager, initial_balance=10000)
                
                # Run quick backtest
                metrics = backtester.run(df.copy())
                
                if metrics['Total PnL'] > best_pnl:
                    best_pnl = metrics['Total PnL']
                    best_params = {
                        'ema_short': short, 
                        'ema_long': long, 
                        'rsi_period': rsi, 
                        'adx_threshold': adx,
                        'bb_window': bb
                    }
            
            logger.info(f"🏆 Best Params Found: {best_params} | PnL: {best_pnl:.2f}")
            
            # Update params.json if valid
            if best_params:
                current_params = {}
                try:
                    with open('params.json', 'r') as f:
                        current_params = json.load(f)
                except:
                    pass
                
                if current_params != best_params:
                    with open('params.json', 'w') as f:
                        json.dump(best_params, f, indent=4)
                    logger.info("💾 Updated params.json with new evolutionary winner!")
                else:
                    logger.info("💤 Current params are still optimal.")
            
            # Sleep for a while before next evolution cycle
            logger.info("⏳ Sleeping for 5 minutes...")
            time.sleep(300)
            
        except Exception as e:
            logger.error(f"❌ Evolution Error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    run_evolution()
