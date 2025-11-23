import ccxt
import time
import logging

class ExecutionEngine:
    def __init__(self, exchange):
        self.exchange = exchange

    def set_leverage(self, symbol, leverage):
        try:
            self.exchange.set_leverage(leverage, symbol)
            print(f"[CONFIG] Leverage set to {leverage}x for {symbol}")
        except Exception as e:
            print(f"[ERROR] Failed to set leverage: {e}")

    def place_order(self, symbol, side, quantity, order_type='market', price=None, params={}):
        """
        Places an order on the exchange.
        """
        try:
            # Ensure quantity is precise enough for the exchange
            # In a real bot, we would use exchange.amount_to_precision(symbol, quantity)
            # For now, we'll round to 3 decimals for BTC
            quantity = round(quantity, 3)
            
            if order_type == 'market':
                order = self.exchange.create_order(symbol, order_type, side, quantity, params=params)
            elif order_type == 'limit':
                order = self.exchange.create_order(symbol, order_type, side, quantity, price, params=params)
            
            print(f"[ORDER] Order Placed: {side} {quantity} {symbol} @ {order_type.upper()}")
            return order
        except Exception as e:
            print(f"[ERROR] Order Placement Failed: {e}")
            return None

    def close_position(self, symbol):
        try:
            # Get current position
            positions = self.exchange.fetch_positions([symbol])
            position = positions[0] if positions else None
            
            if position and float(position['contracts']) > 0:
                side = 'sell' if position['side'] == 'long' else 'buy'
                amount = float(position['contracts'])
                self.exchange.create_market_order(symbol, side, amount)
                print(f"Position closed for {symbol}")
                
                # Cancel all open orders (SL/TP)
                self.exchange.cancel_all_orders(symbol)
        except Exception as e:
            print(f"Error closing position: {e}")

    def update_stop_loss(self, symbol, side, new_stop_price):
        """Updates the Stop Loss order for an open position."""
        try:
            # 1. Cancel existing open orders (assuming only SL/TP are open)
            # In a real scenario, we should find the specific SL order ID.
            # For simplicity, we cancel all and re-place if needed.
            self.exchange.cancel_all_orders(symbol)
            
            # 2. Place new Stop Market order
            # Side for SL is opposite to position
            sl_side = 'sell' if side == 'long' else 'buy'
            
            self.exchange.create_order(
                symbol=symbol,
                type='STOP_MARKET',
                side=sl_side,
                amount=None, # Close entire position? We need amount.
                # If we don't pass amount, some exchanges default to close position, but ccxt usually needs it.
                # We need to know the position size.
                params={'stopPrice': new_stop_price, 'closePosition': True} # Binance specific for closing
            )
            print(f"[UPDATE] Stop Loss updated to {new_stop_price}")
            return True
        except Exception as e:
            print(f"[ERROR] Error updating Stop Loss: {e}")
            return False

    def fetch_position(self, symbol):
        """Fetches the current position for a symbol."""
        try:
            positions = self.exchange.fetch_positions([symbol])
            if positions:
                # Debug: Print raw position data to understand PnL issue
                logging.info(f"[DEBUG] Position for {symbol}: {positions[0]}") 
                return positions[0]
        except Exception as e:
            print(f"Error fetching position: {e}")
        # Return a default empty position structure if not found or error
        return {'info': {'positionAmt': 0}, 'unrealizedProfit': 0, 'contracts': 0, 'side': 'none', 'unrealizedProfit': 0}
