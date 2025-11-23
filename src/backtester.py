import pandas as pd
from indicators import Indicators

class Backtester:
    def __init__(self, strategy, risk_manager, initial_balance=10000):
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.trades = []
        self.position = None # {'side': 'BUY'/'SELL', 'entry_price': float, 'quantity': float, 'sl': float, 'tp': float}

    def run(self, df):
        """
        Runs the backtest on the provided DataFrame.
        """
        # Pre-calculate indicators dynamically based on strategy
        if hasattr(self.strategy, 'ema_short'):
            Indicators.add_ema(df, self.strategy.ema_short)
        if hasattr(self.strategy, 'ema_long'):
            Indicators.add_ema(df, self.strategy.ema_long)
        if hasattr(self.strategy, 'rsi_period'):
            Indicators.add_rsi(df, self.strategy.rsi_period)
            
        # Ensure ATR is calculated for RiskManager (default 14)
        Indicators.add_atr(df, 14)
        
        for i in range(50, len(df)):
            # Slice the dataframe to simulate real-time (up to current candle)
            # In a fast backtest, we can just look at the current row if indicators are pre-calculated correctly without lookahead.
            # Indicators.add_all uses rolling windows, so row 'i' depends only on 0..i.
            
            current_candle = df.iloc[i]
            previous_candles = df.iloc[:i+1] # Pass context if needed, but strategy uses last row
            
            timestamp = current_candle['timestamp']
            price = current_candle['close']
            atr = current_candle['atr_14']
            
            # Check for Exits
            if self.position:
                self._check_exit(price, timestamp)
            
            # Check for Entries (only if no position)
            if not self.position:
                signal = self.strategy.generate_signal(previous_candles)
                if signal != 'NEUTRAL':
                    self._open_position(signal, price, atr, timestamp)

        self._calculate_metrics()
        return self.metrics

    def _open_position(self, side, price, atr, timestamp):
        sl = self.risk_manager.calculate_stop_loss_price(price, atr, side)
        tp = self.risk_manager.calculate_take_profit_price(price, atr, side)
        quantity = self.risk_manager.calculate_quantity(self.balance, price, sl)
        
        if quantity > 0:
            self.position = {
                'side': side,
                'entry_price': price,
                'quantity': quantity,
                'sl': sl,
                'tp': tp,
                'entry_time': timestamp
            }

    def _check_exit(self, price, timestamp):
        pos = self.position
        pnl = 0
        reason = None
        
        if pos['side'] == 'BUY':
            if price <= pos['sl']:
                pnl = (pos['sl'] - pos['entry_price']) * pos['quantity']
                reason = 'Stop Loss'
            elif price >= pos['tp']:
                pnl = (pos['tp'] - pos['entry_price']) * pos['quantity']
                reason = 'Take Profit'
        elif pos['side'] == 'SELL':
            if price >= pos['sl']:
                pnl = (pos['entry_price'] - pos['sl']) * pos['quantity']
                reason = 'Stop Loss'
            elif price <= pos['tp']:
                pnl = (pos['entry_price'] - pos['tp']) * pos['quantity']
                reason = 'Take Profit'
                
        if reason:
            self.balance += pnl
            self.trades.append({
                'entry_time': pos['entry_time'],
                'exit_time': timestamp,
                'side': pos['side'],
                'pnl': pnl,
                'reason': reason
            })
            self.position = None

    def _calculate_metrics(self):
        total_trades = len(self.trades)
        if total_trades == 0:
            self.metrics = {'Total Trades': 0, 'Final Balance': self.balance, 'Win Rate': 0, 'Total PnL': 0.0}
            return

        wins = [t for t in self.trades if t['pnl'] > 0]
        win_rate = len(wins) / total_trades
        total_pnl = self.balance - self.initial_balance
        
        self.metrics = {
            'Total Trades': total_trades,
            'Final Balance': round(self.balance, 2),
            'Total PnL': round(total_pnl, 2),
            'Win Rate': round(win_rate * 100, 2)
        }
