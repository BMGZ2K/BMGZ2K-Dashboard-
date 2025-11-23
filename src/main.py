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
        logging.FileHandler("bot_clean.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()

def save_state(data):
    """Saves current bot state to JSON for dashboard."""
    try:
        with open('bot_state.json', 'w') as f:
            json.dump(data, f)
    except Exception as e:
        logger.error(f"Failed to save state: {e}")

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
            logger.info("[TESTNET] Running in TESTNET mode")
        return exchange
    except Exception as e:
        logger.error(f"[ERROR] Error initializing exchange: {e}")
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
    logger.info("[START] Starting Binance Futures Bot (Multi-Symbol AI)...")
    exchange = initialize_exchange()
    
    # Initialize Modules
    market_data = MarketData(exchange)
    risk_manager = RiskManager(risk_per_trade=0.02, leverage=config.LEVERAGE)
    execution = ExecutionEngine(exchange)
    data_collector = DataCollector()
    
    # Initial Params Load
    params = load_params()
    
    # Initialize Strategy per Symbol (or shared if stateless)
    # HybridStrategy is mostly stateless but stores 'last_ml_confidence'
    strategies = {}
    for sym in config.SYMBOLS:
        strategies[sym] = HybridStrategy(
            ema_short=params.get('ema_short', 9),
            ema_long=params.get('ema_long', 21),
            rsi_period=params.get('rsi_period', 14),
            adx_threshold=params.get('adx_threshold', 20),
            bb_window=params.get('bb_window', 20)
        )
        # Set Initial Leverage
        execution.set_leverage(sym, config.LEVERAGE)
    
    last_params_mtime = 0
    if os.path.exists('params.json'):
        last_params_mtime = os.path.getmtime('params.json')
    
    logger.info(f"Bot initialized for {config.SYMBOLS} on {config.TIMEFRAME} timeframe.")
    
    # Global State Dictionary
    bot_state = {}

    while True:
        try:
            # Check for Parameter Updates (Hot-Reload)
            if os.path.exists('params.json'):
                current_mtime = os.path.getmtime('params.json')
                if current_mtime > last_params_mtime:
                    logger.info("[UPDATE] Detected parameter change. Reloading Strategies...")
                    params = load_params()
                    for sym in config.SYMBOLS:
                        strategies[sym] = HybridStrategy(
                            ema_short=params.get('ema_short', 9),
                            ema_long=params.get('ema_long', 21),
                            rsi_period=params.get('rsi_period', 14),
                            adx_threshold=params.get('adx_threshold', 20),
                            bb_window=params.get('bb_window', 20)
                        )
                    last_params_mtime = current_mtime
                    logger.info(f"[OK] Strategies Updated")

            # --- MULTI-SYMBOL LOOP ---
            active_positions_count = 0
            
            # First pass to count positions (optional, but good for risk check)
            # We can do it inside the loop or fetch all positions once if API allows.
            # For simplicity, we'll check inside the loop or rely on local tracking.
            
            for symbol in config.SYMBOLS:
                try:
                    strategy = strategies[symbol]
                    
                    # 1. Fetch Data (Lower Timeframe)
                    df = market_data.fetch_ohlcv(symbol, timeframe=config.TIMEFRAME, limit=60)
                    
                    # 1b. Fetch Data (Higher Timeframe - MTF)
                    df_htf = market_data.fetch_ohlcv(symbol, timeframe='15m', limit=50)
                    
                    if df.empty or df_htf.empty:
                        logger.warning(f"[WARN] Failed to fetch data for {symbol}. Skipping...")
                        continue

                    # 2. Calculate Indicators
                    if hasattr(strategy, 'ema_short'): Indicators.add_ema(df, strategy.ema_short)
                    if hasattr(strategy, 'ema_long'): Indicators.add_ema(df, strategy.ema_long)
                    if hasattr(strategy, 'rsi_period'): Indicators.add_rsi(df, strategy.rsi_period)
                    
                    Indicators.add_atr(df, 14)
                    Indicators.add_adx(df, 14)
                    Indicators.add_bb(df, strategy.bb_window)
                    Indicators.add_heikin_ashi(df)

                    # ML Features
                    Indicators.add_ema(df, 9)
                    Indicators.add_ema(df, 21)
                    Indicators.add_bb(df, 20)
                    Indicators.add_rsi(df, 14)
                    
                    # 2b. HTF Indicators
                    Indicators.add_ema(df_htf, 21)
                    Indicators.add_ema(df_htf, 50)
                    
                    htf_trend = 'NEUTRAL'
                    if not df_htf.empty:
                        last_htf = df_htf.iloc[-1]
                        if last_htf['ema_21'] > last_htf['ema_50']:
                            htf_trend = 'UP'
                        elif last_htf['ema_21'] < last_htf['ema_50']:
                            htf_trend = 'DOWN'
                    
                    # 3. Generate Signal
                    signal = strategy.generate_signal(df, trend_direction=htf_trend)
                    current_price = df.iloc[-1]['close']
                    atr = df.iloc[-1]['atr_14']
                    
                    # --- ML Data Collection ---
                    data_collector.log_data(symbol, df)
                    
                    # 4. Check Position
                    position = execution.fetch_position(symbol)
                    # logger.info(f"[DEBUG] Raw Position for {symbol}: {position}") # Uncomment to debug PnL
                    amt = float(position.get('info', {}).get('positionAmt', 0))
                    unrealized_pnl = float(position.get('unrealizedProfit', 0))
                    
                    # Fallback: Try to get PnL from raw info if 0 (Binance specific)
                    if unrealized_pnl == 0 and 'info' in position and 'unRealizedProfit' in position['info']:
                        unrealized_pnl = float(position['info']['unRealizedProfit'])
                    
                    if amt != 0:
                        active_positions_count += 1

                    # Update State for this Symbol
                    bot_state[symbol] = {
                        'price': current_price,
                        'signal': signal,
                        'position': amt,
                        'pnl': unrealized_pnl,
                        'ml_confidence': getattr(strategy, 'last_ml_confidence', 0.0),
                        'htf_trend': htf_trend
                    }
                    
                    logger.info(f"[{symbol}] Price: {current_price:.2f} | Sig: {signal} | Pos: {amt} | PnL: {unrealized_pnl:.2f}")

                    # 5. Execution Logic
                    if amt == 0:
                        # Entry Logic
                        if signal in ['BUY', 'SELL']:
                            # Check Max Positions
                            # We need a global count. 
                            # Ideally we fetch all positions first, but let's trust our loop count for next iteration
                            # or just check balance.
                            
                            balance_info = exchange.fetch_balance()
                            usdt_balance = balance_info['total']['USDT']
                            
                            # Risk Check: Don't open if too many positions (approximate check)
                            # For strict safety, we should fetch all positions from exchange.
                            # But let's proceed with per-trade allocation.
                            
                            sl_price = risk_manager.calculate_stop_loss_price(current_price, atr, signal)
                            
                            # Dynamic Leverage
                            volatility_pct = (atr / current_price) * 100
                            rec_leverage = risk_manager.recommend_leverage(volatility_pct)
                            execution.set_leverage(symbol, rec_leverage)
                            
                            # Dynamic TP
                            tp_mult = params.get('tp_atr_multiplier', 3.0)
                            tp_price = risk_manager.calculate_take_profit_price(
                                current_price, atr, signal, tp_atr_multiplier=tp_mult
                            )
                            
                            # Quantity Calculation (Fixed Amount or % of Balance)
                            # Use config.USDT_AMOUNT if defined, else risk calc
                            quantity = (config.USDT_AMOUNT * config.LEVERAGE) / current_price
                            
                            # ML Scaling
                            conf = getattr(strategy, 'last_ml_confidence', 0)
                            if conf >= 0.7: quantity *= 1.5
                            elif conf < 0.4: quantity *= 0.5
                            
                            if quantity * current_price > 5: # Min notional approx $5
                                side = 'buy' if signal == 'BUY' else 'sell'
                                execution.place_order(symbol, side, quantity, 'market')
                                log_trade({
                                    'timestamp': datetime.now().isoformat(),
                                    'symbol': symbol,
                                    'side': side,
                                    'price': current_price,
                                    'quantity': quantity,
                                    'confidence': conf
                                })
                    
                    else:
                        # Exit Logic
                        pos_side = 'long' if amt > 0 else 'short'
                        if (pos_side == 'long' and signal == 'SELL') or (pos_side == 'short' and signal == 'BUY'):
                            logger.info(f"[{symbol}] Reversal! Closing {pos_side}.")
                            execution.close_position(symbol)
                            log_trade({
                                'timestamp': datetime.now().isoformat(),
                                'symbol': symbol,
                                'side': 'close_' + pos_side,
                                'price': current_price,
                                'quantity': abs(amt),
                                'confidence': 0
                            })

                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    continue
            
            # Save State
            save_state(bot_state)
            
            # Wait before next loop
            time.sleep(10)
            
        except KeyboardInterrupt:
            logger.info("[STOP] Bot stopped by user.")
            break
        except Exception as e:
            logger.error(f"[ERROR] Unexpected error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()
