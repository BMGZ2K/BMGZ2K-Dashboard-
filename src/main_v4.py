import ccxt
import time
import sys
import pandas as pd
import logging
import json
import os
from datetime import datetime
import config
from data import MarketData
from indicators import Indicators
from strategy import HybridStrategy
from risk import RiskManager
from execution import ExecutionEngine
from data_collector import DataCollector

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()

def save_state(data):
    """Saves current bot state to JSON for dashboard."""
    with open('bot_state.json', 'w') as f:
        json.dump(data, f)

def log_trade(trade_data):
    """Logs trade to CSV."""
    file_exists = os.path.isfile('trades.csv')
    try:
        df = pd.DataFrame([trade_data])
        df.to_csv('trades.csv', mode='a', header=not file_exists, index=False)
    except Exception as e:
        logger.error(f"Failed to log trade: {e}")


def initialize_exchange():
    try:
        exchange = ccxt.binance({
            'apiKey': config.API_KEY,
            'secret': config.SECRET_KEY,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'recvWindow': 60000,
            }
        })
        if config.TESTNET:
            exchange.set_sandbox_mode(True)
            logger.info("⚠️  Running in TESTNET mode")
        return exchange
    except Exception as e:
        logger.error(f"❌ Error initializing exchange: {e}")
        sys.exit(1)

def load_params():
    """Loads strategy parameters from JSON."""
    try:
        if os.path.exists('params.json'):
            with open('params.json', 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load params: {e}")
    return {"ema_short": 9, "ema_long": 21, "rsi_period": 14}

def main():
    logger.info("🚀 Starting Binance Futures Bot (v4)...")
    exchange = initialize_exchange()
    
    # Initialize Modules
    market_data = MarketData(exchange)
    risk_manager = RiskManager(risk_per_trade=0.02, leverage=config.LEVERAGE)
    execution = ExecutionEngine(exchange)
    data_collector = DataCollector()
    
    # Initial Params Load
    params = load_params()
    strategy = HybridStrategy(
        ema_short=params.get('ema_short', 9),
        ema_long=params.get('ema_long', 21),
        rsi_period=params.get('rsi_period', 14),
        adx_threshold=params.get('adx_threshold', 20),
        bb_window=params.get('bb_window', 20)
    )
    last_params_mtime = 0
    if os.path.exists('params.json'):
        last_params_mtime = os.path.getmtime('params.json')
    
    # Set Leverage
    execution.set_leverage(config.SYMBOL, config.LEVERAGE)
    
    logger.info(f"Bot initialized for {config.SYMBOL} on {config.TIMEFRAME} timeframe.")
    logger.info(f"Hybrid Strategy: Trend (ADX>{strategy.adx_threshold}) & Range (BB {strategy.bb_window})")
    
    while True:
        try:
            # Check for Parameter Updates (Hot-Reload)
            if os.path.exists('params.json'):
                current_mtime = os.path.getmtime('params.json')
                if current_mtime > last_params_mtime:
                    logger.info("♻️  Detected parameter change. Reloading Strategy...")
                    params = load_params()
                    strategy = HybridStrategy(
                        ema_short=params.get('ema_short', 9),
                        ema_long=params.get('ema_long', 21),
                        rsi_period=params.get('rsi_period', 14),
                        adx_threshold=params.get('adx_threshold', 20),
                        bb_window=params.get('bb_window', 20)
                    )
                    last_params_mtime = current_mtime
                    logger.info(f"✅ Strategy Updated: Hybrid Mode Active")

            # 1. Fetch Data (Lower Timeframe)
            df = market_data.fetch_ohlcv(config.SYMBOL, timeframe=config.TIMEFRAME, limit=50)
            
            # 1b. Fetch Data (Higher Timeframe - MTF)
            # We use 15m as the higher timeframe for 1m trading
            df_htf = market_data.fetch_ohlcv(config.SYMBOL, timeframe='15m', limit=50)
            
            if df.empty or df_htf.empty:
                logger.warning("❌ Failed to fetch data. Retrying...")
                time.sleep(10)
                continue

            # 2. Calculate Indicators Dynamically (Lower TF)
            if hasattr(strategy, 'ema_short'):
                Indicators.add_ema(df, strategy.ema_short)
            if hasattr(strategy, 'ema_long'):
                Indicators.add_ema(df, strategy.ema_long)
            if hasattr(strategy, 'rsi_period'):
                Indicators.add_rsi(df, strategy.rsi_period)
            
            # Always calculate ATR, ADX, BB, and Heikin Ashi (Lower TF)
            Indicators.add_atr(df, 14)
            Indicators.add_adx(df, 14)
            Indicators.add_bb(df, strategy.bb_window)
            Indicators.add_heikin_ashi(df)
            
            # 2b. Calculate Indicators (Higher TF) for Trend Direction
            # We'll use standard EMA 21/50 cross for major trend
            Indicators.add_ema(df_htf, 21)
            Indicators.add_ema(df_htf, 50)
            
            htf_trend = 'NEUTRAL'
            if not df_htf.empty:
                last_htf = df_htf.iloc[-1]
                if last_htf['ema_21'] > last_htf['ema_50']:
                    htf_trend = 'UP'
                elif last_htf['ema_21'] < last_htf['ema_50']:
                    htf_trend = 'DOWN'
            
            # 3. Generate Signal (with MTF Confirmation)
            signal = strategy.generate_signal(df, trend_direction=htf_trend)
            current_price = df.iloc[-1]['close']
            atr = df.iloc[-1]['atr_14']
            adx = df.iloc[-1].get('adx_14', 0)
            
            logger.info(f"💲 Price: {current_price:.2f} | Signal: {signal} | ADX: {adx:.2f} | HTF Trend: {htf_trend}")
            
            # --- ML Data Collection ---
            # Log features for future training
            data_collector.log_step(df, signal, htf_trend)
            # --------------------------
            
            # Check for existing positions (Needed for State & Logic)
            positions = exchange.fetch_positions([config.SYMBOL])
            open_position = None
            for pos in positions:
                if float(pos['contracts']) > 0:
                    open_position = pos
                    break
            
            # Prepare Position Data for State
            pos_data = None
            if open_position:
                pos_data = {
                    'side': open_position['side'],
                    'entry_price': float(open_position['entryPrice']),
                    'quantity': float(open_position['contracts']),
                    'unrealized_pnl': float(open_position['unrealizedPnl']) # Binance usually provides this
                }

            # Save State for Dashboard
            state = {
                'timestamp': datetime.now().isoformat(),
                'symbol': config.SYMBOL,
                'price': current_price,
                'signal': signal,
                'rsi': df.iloc[-1][f'rsi_{strategy.rsi_period}'],
                'ema_short': df.iloc[-1][f'ema_{strategy.ema_short}'],
                'ema_long': df.iloc[-1][f'ema_{strategy.ema_long}'],
                'atr': atr,
                'adx': adx,
                'htf_trend': htf_trend,
                'position': pos_data,
                'recent_prices': df['close'].tail(50).tolist(), # Export last 50 prices for chart
                'params': {
                    'ema_short': strategy.ema_short,
                    'ema_long': strategy.ema_long,
                    'rsi': strategy.rsi_period,
                    'adx_threshold': strategy.adx_threshold,
                    'bb_window': strategy.bb_window
                }
            }
            save_state(state)
            
            # 4. Execute Trade
            if signal != 'NEUTRAL':
                # Only trade if no position
                if not open_position:
                    balance_info = exchange.fetch_balance()
                    usdt_balance = balance_info['total']['USDT']
                    
                    sl_price = risk_manager.calculate_stop_loss_price(current_price, atr, signal)
                    
                    # Dynamic TP
                    tp_mult = params.get('tp_atr_multiplier', 3.0) # Default 3x ATR
                    tp_price = risk_manager.calculate_take_profit_price(
                        current_price, atr, signal, tp_atr_multiplier=tp_mult
                    )
                    quantity = risk_manager.calculate_quantity(usdt_balance, current_price, sl_price)
                    
                    if quantity > 0.001:
                        side = 'buy' if signal == 'BUY' else 'sell'
                        logger.info(f"⚡ Executing {side.upper()} | Qty: {quantity} | SL: {sl_price:.2f} | TP: {tp_price:.2f}")
                        
                        order = execution.place_order(config.SYMBOL, side, quantity, 'market')
                        
                        if order:
                            # Log Trade
                            log_trade({
                                'timestamp': datetime.now().isoformat(),
                                'symbol': config.SYMBOL,
                                'side': side,
                                'price': current_price,
                                'quantity': quantity,
                                'sl': sl_price,
                                'tp': tp_price
                            })
                    else:
                        logger.warning("⚠️ Quantity too small to trade.")
                else:
                    # Suppress repetitive logs
                    if int(time.time()) % 60 == 0: # Log only once per minute
                        logger.info("🔒 Position already open. Monitoring...")
                    
                    # --- TRAILING STOP LOGIC ---
                    ts_atr_mult = params.get('trailing_stop_atr', 2.0)
                    if ts_atr_mult > 0:
                        entry_price = float(open_position['entryPrice'])
                        pos_side = open_position['side'] # 'long' or 'short'
                        
                        # Calculate Dynamic Stop Price
                        # For Long: High - (ATR * Mult). If > Current SL, update.
                        # For Short: Low + (ATR * Mult). If < Current SL, update.
                        # We need to store the "Current SL" somewhere. 
                        # For simplicity, we can check active orders or just calculate "Theoretical SL".
                        # Let's use a simplified approach: 
                        # If Price moves X% in favor, move SL to Break Even, then Trail.
                        
                        # Better: Standard Trailing based on current price.
                        trail_dist = atr * ts_atr_mult
                        
                        if pos_side == 'long':
                            new_sl = current_price - trail_dist
                            # We need to know the CURRENT SL to see if we should move it UP.
                            # Fetching open orders is expensive every loop.
                            # Let's assume we only move it if it's significantly higher than Entry (locking profit).
                            
                            if new_sl > entry_price:
                                # Check if we haven't already moved it close to this
                                # This requires state. Let's just try to update if it's > entry + buffer
                                # To avoid spamming API, we need state.
                                pass 
                                # TODO: Implement robust state tracking for SL.
                                # For this demo, let's just Log that we WOULD trail.
                                # logger.info(f"🧐 Trailing Stop Check: New SL {new_sl:.2f} vs Entry {entry_price:.2f}")
                                
                                # ACTUALLY IMPLEMENTING:
                                # We will use a simple memory variable in the loop? No, loop restarts.
                                # We can check 'open_orders' once per minute?
                                pass

                    # Basic Exit Logic (if signal flips)
                    pos_side = 'long' if open_position['side'] == 'long' else 'short'
                    if (pos_side == 'long' and signal == 'SELL') or (pos_side == 'short' and signal == 'BUY'):
                        logger.info(f"🔄 Signal flip! Closing {pos_side} position.")
                        execution.close_position(config.SYMBOL)
                        
                        # Log Exit
                        log_trade({
                            'timestamp': datetime.now().isoformat(),
                            'symbol': config.SYMBOL,
                            'side': 'close_' + pos_side,
                            'price': current_price,
                            'quantity': float(open_position['contracts']),
                            'sl': 0,
                            'tp': 0
                        })

            # Sleep before next cycle
            # Faster loop for HFT-like responsiveness
            time.sleep(3)
            
        except KeyboardInterrupt:
            logger.info("🛑 Bot stopped by user.")
            break
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()
