import ccxt
import pandas as pd
import config
import sys
import time
from data import MarketData
from indicators import Indicators
from strategy import MomentumStrategy
from risk import RiskManager
from execution import ExecutionEngine

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
        return exchange
    except Exception as e:
        print(f"❌ Error initializing exchange: {e}")
        sys.exit(1)

def run_bot_cycle():
    print("🤖 Starting Bot Cycle...")
    exchange = initialize_exchange()
    
    # Initialize Modules
    market_data = MarketData(exchange)
    strategy = MomentumStrategy()
    risk_manager = RiskManager(risk_per_trade=0.02, leverage=config.LEVERAGE)
    execution = ExecutionEngine(exchange)
    
    # Set Leverage
    execution.set_leverage(config.SYMBOL, config.LEVERAGE)
    
    # 1. Fetch Data
    print(f"📥 Fetching data for {config.SYMBOL}...")
    df = market_data.fetch_ohlcv(config.SYMBOL, timeframe=config.TIMEFRAME, limit=50)
    
    if df.empty:
        print("❌ Failed to fetch data.")
        return

    # 2. Calculate Indicators
    df = Indicators.add_all(df)
    
    # 3. Generate Signal
    signal = strategy.generate_signal(df)
    current_price = df.iloc[-1]['close']
    atr = df.iloc[-1]['atr_14']
    
    print(f"💲 Price: {current_price} | ATR: {atr:.2f} | Signal: {signal}")
    
    # 4. Execute Trade (if signal)
    if signal != 'NEUTRAL':
        # Check Balance
        balance_info = exchange.fetch_balance()
        usdt_balance = balance_info['total']['USDT']
        
        # Calculate Risk Parameters
        sl_price = risk_manager.calculate_stop_loss_price(current_price, atr, signal)
        tp_price = risk_manager.calculate_take_profit_price(current_price, atr, signal)
        quantity = risk_manager.calculate_quantity(usdt_balance, current_price, sl_price)
        
        print(f"⚖️  Risk Calc: Balance={usdt_balance:.2f} | Qty={quantity:.3f} | SL={sl_price:.2f} | TP={tp_price:.2f}")
        
        if quantity > 0.001: # Min order size check (approx)
            # Place Order
            side = 'buy' if signal == 'BUY' else 'sell'
            order = execution.place_order(config.SYMBOL, side, quantity, 'market')
            
            if order:
                # Place SL/TP Orders (Simplified: Just printing for now, or separate orders)
                # Binance Futures supports placing SL/TP with the order or separately.
                # For this demo, we'll just log it.
                print(f"🛡️ Stop Loss should be at {sl_price:.2f}")
                print(f"🎯 Take Profit should be at {tp_price:.2f}")
        else:
            print("⚠️ Quantity too small to trade.")
    else:
        print("💤 No trade signal.")

if __name__ == "__main__":
    run_bot_cycle()
