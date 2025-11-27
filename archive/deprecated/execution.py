"""
Execution Module - Order Management and Trade Execution
Robust order execution with retries and error handling
"""
import ccxt
import time
import csv
import os
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass

from .config import (
    API_KEY, SECRET_KEY, USE_TESTNET,
    LEVERAGE_CAP, EXECUTION_CONFIG, LOG_FILE
)


@dataclass
class OrderResult:
    """Order execution result."""
    success: bool
    order_id: str = ""
    symbol: str = ""
    side: str = ""
    amount: float = 0.0
    price: float = 0.0
    status: str = ""
    error: str = ""
    realized_pnl: float = 0.0


class ExchangeManager:
    """
    Manage exchange connection and market data.
    """
    
    def __init__(self):
        self.exchange = None
        self.markets = {}
        self._initialize()
    
    def _initialize(self):
        """Initialize exchange connection."""
        try:
            self.exchange = ccxt.binance({
                'apiKey': API_KEY,
                'secret': SECRET_KEY,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',
                    'recvWindow': 60000,
                }
            })
            
            if USE_TESTNET:
                self.exchange.set_sandbox_mode(True)
                print("[TESTNET] Running in sandbox mode")
            
            self.exchange.load_markets()
            print(f"Exchange initialized. {len(self.exchange.markets)} markets loaded.")
            
        except Exception as e:
            print(f"Exchange initialization failed: {e}")
            raise
    
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = '5m',
        limit: int = 500
    ) -> Optional[List]:
        """Fetch OHLCV data."""
        try:
            return self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        except Exception as e:
            print(f"Failed to fetch OHLCV for {symbol}: {e}")
            return None
    
    def fetch_balance(self) -> Dict[str, float]:
        """Fetch account balance."""
        try:
            balance = self.exchange.fetch_balance()
            return {
                'total': float(balance['total'].get('USDT', 0)),
                'free': float(balance['free'].get('USDT', 0)),
                'used': float(balance['used'].get('USDT', 0))
            }
        except Exception as e:
            print(f"Failed to fetch balance: {e}")
            return {'total': 0, 'free': 0, 'used': 0}
    
    def fetch_positions(self) -> List[Dict]:
        """Fetch all open positions."""
        try:
            positions = self.exchange.fetch_positions()
            return [p for p in positions if float(p.get('contracts', 0)) != 0]
        except Exception as e:
            print(f"Failed to fetch positions: {e}")
            return []
    
    def fetch_position(self, symbol: str) -> Optional[Dict]:
        """Fetch position for specific symbol."""
        try:
            positions = self.exchange.fetch_positions([symbol])
            for p in positions:
                if p['symbol'] == symbol:
                    return p
            return None
        except Exception as e:
            print(f"Failed to fetch position for {symbol}: {e}")
            return None
    
    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for symbol."""
        try:
            self.exchange.set_leverage(leverage, symbol)
            return True
        except Exception as e:
            if 'No need to change' not in str(e):
                print(f"Failed to set leverage for {symbol}: {e}")
            return False
    
    def fetch_funding_rate(self, symbol: str) -> float:
        """Fetch current funding rate."""
        try:
            if hasattr(self.exchange, 'fetch_funding_rate'):
                rate = self.exchange.fetch_funding_rate(symbol)
                return float(rate.get('fundingRate', 0))
        except:
            pass
        return 0.0


class ExecutionEngine:
    """
    Order execution with retry logic and error handling.
    """
    
    def __init__(self, exchange_manager: ExchangeManager):
        self.exchange_manager = exchange_manager
        self.exchange = exchange_manager.exchange
        
        self.retry_attempts = EXECUTION_CONFIG.get('retry_attempts', 5)
        self.retry_delay_base = EXECUTION_CONFIG.get('retry_delay_base', 0.5)
        self.min_notional = EXECUTION_CONFIG.get('min_notional', 6.0)
        
        self.blacklist = set()
    
    def execute_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        reduce_only: bool = False,
        reason: str = ""
    ) -> OrderResult:
        """
        Execute a market order with retry logic.
        
        Args:
            symbol: Trading pair
            side: 'buy' or 'sell'
            amount: Order quantity
            price: Current market price (for logging)
            reduce_only: Close position only
            reason: Order reason for logging
        
        Returns:
            OrderResult with execution details
        """
        attempts = 0
        final_amount = amount
        last_error = ""
        
        while attempts < self.retry_attempts:
            try:
                # Check min notional
                notional = final_amount * price
                if notional < self.min_notional:
                    if not reduce_only:
                        final_amount = self.min_notional / price * 1.1
                        print(f"Adjusted amount to meet min notional: {final_amount:.4f}")
                
                # Get market precision
                market = self.exchange.market(symbol)
                min_amount = market['limits']['amount']['min']
                
                if final_amount < min_amount:
                    if reduce_only:
                        # For close orders, try the actual position size
                        final_amount = min_amount
                    else:
                        return OrderResult(
                            success=False,
                            symbol=symbol,
                            side=side,
                            error=f"Amount {final_amount} < min {min_amount}"
                        )
                
                # Format quantity
                qty_str = self.exchange.amount_to_precision(symbol, final_amount)
                
                # Execute via raw API for better control
                params = {
                    'symbol': symbol.replace('/', ''),
                    'side': side.upper(),
                    'type': 'MARKET',
                    'quantity': qty_str,
                }
                
                if reduce_only:
                    params['reduceOnly'] = 'true'
                
                order = self.exchange.fapiPrivatePostOrder(params)
                
                # Log trade
                self._log_trade(
                    symbol=symbol,
                    side=side,
                    amount=float(qty_str),
                    price=price,
                    reason=reason,
                    status='FILLED'
                )
                
                return OrderResult(
                    success=True,
                    order_id=str(order.get('orderId', '')),
                    symbol=symbol,
                    side=side,
                    amount=float(qty_str),
                    price=price,
                    status='FILLED'
                )
                
            except Exception as e:
                error_msg = str(e).lower()
                last_error = error_msg
                
                if 'margin' in error_msg or 'insufficient' in error_msg:
                    final_amount *= 0.5
                    print(f"Insufficient margin. Retrying with {final_amount:.4f}")
                
                elif '-2022' in error_msg or 'reduceonly' in error_msg:
                    # Position already closed or size mismatch
                    try:
                        self.exchange.cancel_all_orders(symbol)
                        time.sleep(0.5)
                        
                        pos = self.exchange_manager.fetch_position(symbol)
                        if pos:
                            actual_amt = abs(float(pos.get('contracts', 0)))
                            if actual_amt == 0:
                                return OrderResult(success=True, status='ALREADY_CLOSED')
                            final_amount = actual_amt
                        else:
                            return OrderResult(success=True, status='ALREADY_CLOSED')
                    except:
                        pass
                
                elif 'precision' in error_msg or '-1111' in error_msg:
                    self.exchange.load_markets(reload=True)
                
                elif 'invalid symbol' in error_msg or '-4140' in error_msg:
                    self.blacklist.add(symbol)
                    return OrderResult(
                        success=False,
                        symbol=symbol,
                        error='Invalid symbol - blacklisted'
                    )
                
                attempts += 1
                time.sleep(self.retry_delay_base * (attempts + 1))
        
        # Log failed trade
        self._log_trade(
            symbol=symbol,
            side=side,
            amount=amount,
            price=price,
            reason=reason,
            status=f'FAILED: {last_error}'
        )
        
        return OrderResult(
            success=False,
            symbol=symbol,
            side=side,
            amount=amount,
            error=last_error
        )
    
    def close_position(self, symbol: str, reason: str = "CLOSE") -> OrderResult:
        """Close entire position for a symbol."""
        pos = self.exchange_manager.fetch_position(symbol)
        
        if not pos:
            return OrderResult(success=True, status='NO_POSITION')
        
        amt = float(pos.get('contracts', 0))
        if amt == 0:
            return OrderResult(success=True, status='NO_POSITION')
        
        side = 'sell' if amt > 0 else 'buy'
        price = float(pos.get('markPrice', 0))
        
        return self.execute_order(
            symbol=symbol,
            side=side,
            amount=abs(amt),
            price=price,
            reduce_only=True,
            reason=reason
        )
    
    def close_all_positions(self, reason: str = "PANIC_CLOSE") -> List[OrderResult]:
        """Close all open positions."""
        results = []
        positions = self.exchange_manager.fetch_positions()
        
        for pos in positions:
            symbol = pos['symbol']
            result = self.close_position(symbol, reason)
            results.append(result)
        
        return results
    
    def _log_trade(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        reason: str,
        status: str,
        pnl: float = 0.0
    ):
        """Log trade to CSV file."""
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        
        file_exists = os.path.exists(LOG_FILE)
        
        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    'timestamp', 'symbol', 'side', 'amount', 
                    'price', 'reason', 'status', 'pnl'
                ])
            
            writer.writerow([
                datetime.now().isoformat(),
                symbol, side, amount, price, reason, status, pnl
            ])


class PositionTracker:
    """
    Track and manage active positions locally.
    """
    
    def __init__(self, exchange_manager: ExchangeManager):
        self.exchange_manager = exchange_manager
        self.positions: Dict[str, Dict] = {}
    
    def sync_with_exchange(self):
        """Sync local positions with exchange."""
        exchange_positions = self.exchange_manager.fetch_positions()
        
        synced = {}
        for pos in exchange_positions:
            symbol = pos['symbol']
            amt = float(pos.get('contracts', 0))
            
            if amt == 0:
                continue
            
            # Preserve local state if exists
            local = self.positions.get(symbol, {})
            
            synced[symbol] = {
                'amt': amt,
                'entry': float(pos.get('entryPrice', 0)),
                'pnl': float(pos.get('unrealizedPnl', 0)),
                'entry_time': local.get('entry_time', datetime.now().isoformat()),
                'max_price': max(local.get('max_price', 0), float(pos.get('markPrice', 0))),
                'min_price': min(local.get('min_price', float('inf')), float(pos.get('markPrice', 0))),
                'trail_stop': local.get('trail_stop', 0),
                'dca_count': local.get('dca_count', 0),
                'tp_count': local.get('tp_count', 0)
            }
        
        self.positions = synced
    
    def update_position(
        self,
        symbol: str,
        price: float,
        pnl: float
    ):
        """Update position with current price."""
        if symbol not in self.positions:
            return
        
        pos = self.positions[symbol]
        pos['pnl'] = pnl
        
        if pos['amt'] > 0:  # Long
            pos['max_price'] = max(pos.get('max_price', price), price)
        else:  # Short
            pos['min_price'] = min(pos.get('min_price', price), price)
    
    def add_position(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        amount: float
    ):
        """Add new position."""
        if symbol in self.positions:
            # DCA
            old = self.positions[symbol]
            old_cost = abs(old['amt']) * old['entry']
            new_cost = amount * entry_price
            total_qty = abs(old['amt']) + amount
            
            self.positions[symbol] = {
                'amt': total_qty if side == 'buy' else -total_qty,
                'entry': (old_cost + new_cost) / total_qty,
                'pnl': 0,
                'entry_time': old.get('entry_time', datetime.now().isoformat()),
                'max_price': entry_price,
                'min_price': entry_price,
                'trail_stop': 0,
                'dca_count': old.get('dca_count', 0) + 1,
                'tp_count': old.get('tp_count', 0)
            }
        else:
            self.positions[symbol] = {
                'amt': amount if side == 'buy' else -amount,
                'entry': entry_price,
                'pnl': 0,
                'entry_time': datetime.now().isoformat(),
                'max_price': entry_price,
                'min_price': entry_price,
                'trail_stop': 0,
                'dca_count': 0,
                'tp_count': 0
            }
    
    def remove_position(self, symbol: str):
        """Remove closed position."""
        if symbol in self.positions:
            del self.positions[symbol]
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get position for symbol."""
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, Dict]:
        """Get all positions."""
        return self.positions.copy()


def create_exchange() -> ExchangeManager:
    """Factory function to create exchange manager."""
    return ExchangeManager()


def create_execution_engine(exchange_manager: ExchangeManager) -> ExecutionEngine:
    """Factory function to create execution engine."""
    return ExecutionEngine(exchange_manager)
