"""
Risk Management Module
Advanced Position Sizing, Trailing Stops, and Portfolio Risk

AVISO: Este módulo NÃO está atualmente integrado ao bot.py.
O risk management básico está em core/trader.py.
Funcionalidades avançadas disponíveis aqui:
- recommend_leverage(): Leverage dinâmico baseado em volatilidade
- check_portfolio_risk(): Avaliação de risco do portfolio
- get_cleanup_actions(): Identificação de posições tóxicas
- update_trailing_stop(): Trailing stop dinâmico

TODO: Integrar ao bot.py para uso completo das funcionalidades.
"""
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class Position:
    """Active position data."""
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    quantity: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    trailing_stop: float = 0.0
    max_price: float = 0.0
    min_price: float = 0.0
    dca_count: int = 0
    tp_count: int = 0
    
    @property
    def unrealized_pnl(self) -> float:
        return 0.0  # Updated externally
    
    @property
    def roi_pct(self) -> float:
        return 0.0  # Updated externally


class RiskManager:
    """
    Comprehensive risk management system.
    Handles position sizing, stops, and portfolio-level risk.
    """
    
    def __init__(
        self,
        risk_per_trade: float = 0.02,
        max_portfolio_risk: float = 0.20,
        leverage_cap: int = 12,
        max_positions: int = 15,
        circuit_breaker_drawdown: float = 0.25
    ):
        self.risk_per_trade = risk_per_trade
        self.max_portfolio_risk = max_portfolio_risk
        self.leverage_cap = leverage_cap
        self.max_positions = max_positions
        self.circuit_breaker_drawdown = circuit_breaker_drawdown

        # CORRIGIDO: Inicializar com valor default razoável (será atualizado)
        self.high_water_mark = 10000.0  # Era 0.0, causava divisão por zero
        self.is_halted = False

    def set_initial_balance(self, balance: float):
        """Define o balance inicial e atualiza high water mark."""
        if balance > 0:
            self.high_water_mark = max(self.high_water_mark, balance)
    
    # =========================================================================
    # POSITION SIZING
    # =========================================================================
    
    def calculate_position_size(
        self,
        balance: float,
        entry_price: float,
        stop_loss_price: float,
        signal_score: float = 5.0,
        volatility_pct: float = 2.0
    ) -> Tuple[float, float]:
        """
        Calculate optimal position size based on risk.
        
        Returns:
            (quantity, risk_amount)
        """
        # Base risk adjusted by signal quality
        base_risk = self.risk_per_trade
        
        # Scale risk by signal strength (0.5x to 1.5x)
        if signal_score >= 8.0:
            risk_mult = 1.5
        elif signal_score >= 6.0:
            risk_mult = 1.2
        elif signal_score >= 4.0:
            risk_mult = 1.0
        else:
            risk_mult = 0.5
        
        # Reduce risk in high volatility
        if volatility_pct > 5.0:
            risk_mult *= 0.5
        elif volatility_pct > 3.0:
            risk_mult *= 0.75
        
        adjusted_risk = base_risk * risk_mult
        risk_amount = balance * adjusted_risk
        
        # Calculate stop distance
        stop_distance = abs(entry_price - stop_loss_price)
        if stop_distance == 0:
            return 0.0, 0.0
        
        # Position size based on risk
        quantity_by_risk = risk_amount / stop_distance
        
        # Position size based on leverage cap
        max_notional = balance * self.leverage_cap
        quantity_by_leverage = max_notional / entry_price
        
        # Take minimum
        quantity = min(quantity_by_risk, quantity_by_leverage)
        
        return quantity, risk_amount
    
    def calculate_stop_loss(
        self,
        entry_price: float,
        atr: float,
        side: str,
        multiplier: float = 1.5
    ) -> float:
        """Calculate initial stop loss price."""
        stop_distance = atr * multiplier
        
        if side == 'long':
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance
    
    def calculate_take_profit(
        self,
        entry_price: float,
        atr: float,
        side: str,
        multiplier: float = 3.5
    ) -> float:
        """Calculate take profit price."""
        tp_distance = atr * multiplier
        
        if side == 'long':
            return entry_price + tp_distance
        else:
            return entry_price - tp_distance
    
    # =========================================================================
    # TRAILING STOP LOGIC
    # =========================================================================
    
    def update_trailing_stop(
        self,
        position: Dict,
        current_price: float,
        atr: float,
        params: Dict
    ) -> Tuple[float, bool]:
        """
        Update trailing stop based on price action.
        
        Returns:
            (new_trail_stop, should_exit)
        """
        side = position['side']
        entry_price = position['entry']
        prev_trail = position.get('trail_stop', 0.0)
        max_price = position.get('max_price', entry_price)
        min_price = position.get('min_price', entry_price)
        
        # Calculate ROI
        if side == 'long':
            pnl_per_unit = current_price - entry_price
            max_price = max(max_price, current_price)
            peak_pnl = max_price - entry_price
        else:
            pnl_per_unit = entry_price - current_price
            min_price = min(min_price, current_price)
            peak_pnl = entry_price - min_price
        
        roi_pct = pnl_per_unit / entry_price if entry_price > 0 else 0
        
        # Dynamic ATR multiplier based on profit
        if peak_pnl > (atr * 4.0):
            atr_mult = 0.5
        elif peak_pnl > (atr * 2.0):
            atr_mult = 1.0
        elif peak_pnl > (atr * 1.0):
            atr_mult = 1.5
        else:
            atr_mult = 2.0
        
        # Calculate new trail stop
        if side == 'long':
            new_trail = max_price - (atr * atr_mult)
            
            # Move to breakeven when profitable
            if roi_pct > 0.005 and new_trail < entry_price:
                new_trail = entry_price * 1.001  # BE + fees
            
            # Never lower the stop
            if prev_trail > 0:
                new_trail = max(new_trail, prev_trail)
            
            # Check exit
            should_exit = current_price < new_trail and new_trail > 0
            
        else:  # short
            new_trail = min_price + (atr * atr_mult)
            
            # Move to breakeven
            if roi_pct > 0.005 and new_trail > entry_price:
                new_trail = entry_price * 0.999
            
            # Never raise the stop
            if prev_trail > 0:
                new_trail = min(new_trail, prev_trail)
            
            # Check exit
            should_exit = current_price > new_trail and new_trail > 0
        
        return new_trail, should_exit
    
    # =========================================================================
    # CIRCUIT BREAKER
    # =========================================================================
    
    def check_circuit_breaker(
        self,
        initial_balance: float,
        current_balance: float
    ) -> Tuple[bool, float, float]:
        """
        Check if circuit breaker should trigger.
        
        Returns:
            (is_triggered, drawdown_pct, new_high_water_mark)
        """
        if initial_balance <= 0:
            return False, 0.0, 0.0
        
        # Update high water mark
        self.high_water_mark = max(self.high_water_mark, current_balance)
        
        # Calculate drawdown from peak
        if self.high_water_mark > 0:
            drawdown = (self.high_water_mark - current_balance) / self.high_water_mark
        else:
            drawdown = 0.0
        
        is_triggered = drawdown > self.circuit_breaker_drawdown
        
        if is_triggered:
            self.is_halted = True
        
        return is_triggered, drawdown, self.high_water_mark
    
    # =========================================================================
    # PORTFOLIO RISK
    # =========================================================================
    
    def check_portfolio_risk(
        self,
        active_positions: Dict[str, Dict],
        balance: float
    ) -> Dict[str, Any]:
        """
        Assess overall portfolio risk.
        
        Returns:
            Dict with risk metrics and warnings
        """
        total_exposure = 0.0
        total_risk = 0.0
        longs = 0
        shorts = 0
        
        for symbol, pos in active_positions.items():
            amt = abs(pos.get('amt', 0))
            entry = pos.get('entry', 0)
            side = 'long' if pos.get('amt', 0) > 0 else 'short'
            
            exposure = amt * entry
            total_exposure += exposure
            
            if side == 'long':
                longs += 1
            else:
                shorts += 1
        
        leverage_used = total_exposure / balance if balance > 0 else 0
        
        warnings = []
        
        if leverage_used > self.leverage_cap * 0.8:
            warnings.append(f"High leverage: {leverage_used:.1f}x")
        
        if longs >= 10:
            warnings.append(f"Too many longs: {longs}")
        
        if shorts >= 10:
            warnings.append(f"Too many shorts: {shorts}")
        
        if len(active_positions) >= self.max_positions:
            warnings.append(f"Max positions reached: {len(active_positions)}")
        
        return {
            'total_exposure': total_exposure,
            'leverage_used': leverage_used,
            'longs': longs,
            'shorts': shorts,
            'position_count': len(active_positions),
            'warnings': warnings,
            'can_open_long': longs < 10 and len(active_positions) < self.max_positions,
            'can_open_short': shorts < 10 and len(active_positions) < self.max_positions,
        }
    
    # =========================================================================
    # RISK CLEANUP
    # =========================================================================
    
    def get_cleanup_actions(
        self,
        active_positions: Dict[str, Dict],
        global_sentiment: float,
        current_time: datetime = None
    ) -> List[Dict]:
        """
        Identify positions that should be closed for risk management.
        
        Returns:
            List of close actions
        """
        if current_time is None:
            current_time = datetime.now()
        
        actions = []
        
        for symbol, pos in active_positions.items():
            amt = pos.get('amt', 0)
            entry = pos.get('entry', 0)
            pnl = pos.get('pnl', 0)
            
            if amt == 0 or entry == 0:
                continue
            
            roi = pnl / (abs(amt) * entry) if entry > 0 else 0
            side = 'long' if amt > 0 else 'short'
            
            # Sentiment mismatch
            if global_sentiment < 0.25 and side == 'long' and roi < -0.015:
                actions.append({
                    'symbol': symbol,
                    'side': 'sell',
                    'amount': abs(amt),
                    'reason': f'SENTIMENT_MISMATCH_BEAR (Sent: {global_sentiment:.2f})',
                    'reduceOnly': True,
                    'priority': 10
                })
                continue
            
            if global_sentiment > 0.75 and side == 'short' and roi < -0.015:
                actions.append({
                    'symbol': symbol,
                    'side': 'buy',
                    'amount': abs(amt),
                    'reason': f'SENTIMENT_MISMATCH_BULL (Sent: {global_sentiment:.2f})',
                    'reduceOnly': True,
                    'priority': 10
                })
                continue
            
            # Toxic asset (rapid loss)
            entry_time = pos.get('entry_time')
            if entry_time:
                try:
                    if isinstance(entry_time, str):
                        entry_dt = datetime.fromisoformat(entry_time)
                    else:
                        entry_dt = entry_time
                    
                    duration_minutes = (current_time - entry_dt).total_seconds() / 60
                    
                    # Fast loss
                    if duration_minutes < 30 and roi < -0.05:
                        actions.append({
                            'symbol': symbol,
                            'side': 'sell' if amt > 0 else 'buy',
                            'amount': abs(amt),
                            'reason': f'TOXIC_ASSET (ROI: {roi*100:.1f}% in {duration_minutes:.0f}m)',
                            'reduceOnly': True,
                            'priority': 9
                        })
                        continue
                    
                    # Time stop (stagnant position)
                    duration_hours = duration_minutes / 60
                    if duration_hours > 4 and -0.01 < roi < 0.01:
                        actions.append({
                            'symbol': symbol,
                            'side': 'sell' if amt > 0 else 'buy',
                            'amount': abs(amt),
                            'reason': f'STAGNATION ({duration_hours:.1f}h, ROI: {roi*100:.2f}%)',
                            'reduceOnly': True,
                            'priority': 5
                        })
                        continue
                        
                except:
                    pass
        
        return sorted(actions, key=lambda x: x['priority'], reverse=True)
    
    # =========================================================================
    # DYNAMIC LEVERAGE
    # =========================================================================
    
    def recommend_leverage(self, volatility_pct: float, adx: float = 20) -> int:
        """
        Recommend leverage based on market conditions.
        
        Args:
            volatility_pct: ATR as percentage of price
            adx: Current ADX value
        
        Returns:
            Recommended leverage (capped)
        """
        # Base leverage
        if volatility_pct > 5:
            base_lev = 3
        elif volatility_pct > 3:
            base_lev = 6
        elif volatility_pct > 2:
            base_lev = 10
        else:
            base_lev = self.leverage_cap
        
        # Adjust for trend strength
        if adx > 40:
            base_lev = min(base_lev + 2, self.leverage_cap)
        elif adx < 20:
            base_lev = max(base_lev - 2, 3)
        
        return min(base_lev, self.leverage_cap)


class PositionManager:
    """Manage active positions with advanced features."""
    
    def __init__(self, risk_manager: RiskManager):
        self.risk_manager = risk_manager
        self.positions: Dict[str, Dict] = {}
    
    def add_position(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        stop_loss: float,
        take_profit: float
    ):
        """Add or update a position."""
        if symbol in self.positions:
            # DCA - Average entry
            old = self.positions[symbol]
            old_cost = abs(old['amt']) * old['entry']
            new_cost = quantity * entry_price
            total_qty = abs(old['amt']) + quantity
            
            avg_entry = (old_cost + new_cost) / total_qty
            
            self.positions[symbol] = {
                'amt': total_qty if side == 'long' else -total_qty,
                'entry': avg_entry,
                'pnl': old.get('pnl', 0),
                'entry_time': old.get('entry_time', datetime.now().isoformat()),
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'trail_stop': old.get('trail_stop', 0),
                'max_price': max(old.get('max_price', entry_price), entry_price),
                'min_price': min(old.get('min_price', entry_price), entry_price),
                'dca_count': old.get('dca_count', 0) + 1,
                'tp_count': old.get('tp_count', 0)
            }
        else:
            self.positions[symbol] = {
                'amt': quantity if side == 'long' else -quantity,
                'entry': entry_price,
                'pnl': 0,
                'entry_time': datetime.now().isoformat(),
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'trail_stop': 0,
                'max_price': entry_price,
                'min_price': entry_price,
                'dca_count': 0,
                'tp_count': 0
            }
    
    def close_position(self, symbol: str, partial_pct: float = 1.0) -> Optional[Dict]:
        """Close position (full or partial)."""
        if symbol not in self.positions:
            return None
        
        pos = self.positions[symbol]
        
        if partial_pct >= 1.0:
            # Full close
            closed = self.positions.pop(symbol)
            return closed
        else:
            # Partial close
            close_qty = abs(pos['amt']) * partial_pct
            remaining_qty = abs(pos['amt']) - close_qty
            
            if remaining_qty < 0.0001:
                return self.close_position(symbol)
            
            pos['amt'] = remaining_qty if pos['amt'] > 0 else -remaining_qty
            pos['tp_count'] = pos.get('tp_count', 0) + 1
            
            return {'symbol': symbol, 'closed_qty': close_qty}
    
    def update_pnl(self, symbol: str, current_price: float):
        """Update unrealized PnL for a position."""
        if symbol not in self.positions:
            return
        
        pos = self.positions[symbol]
        entry = pos['entry']
        amt = pos['amt']
        
        if amt > 0:  # Long
            pos['pnl'] = (current_price - entry) * amt
            pos['max_price'] = max(pos.get('max_price', entry), current_price)
        else:  # Short
            pos['pnl'] = (entry - current_price) * abs(amt)
            pos['min_price'] = min(pos.get('min_price', entry), current_price)
