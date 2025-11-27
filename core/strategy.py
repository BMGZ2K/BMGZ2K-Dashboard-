"""
Strategy Module - Multi-Strategy Signal Generation
Combines trend following, mean reversion, and momentum strategies
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime

from .indicators import calculate_indicators, calculate_htf_trend


@dataclass
class Signal:
    """Trading signal with metadata."""
    symbol: str
    direction: str  # 'long', 'short', 'neutral'
    score: float  # 0-10 signal strength
    reason: str
    entry_price: float
    stop_loss: float
    take_profit: float
    indicators: Dict[str, float]


class StrategyEngine:
    """
    Multi-strategy signal generation engine.
    Combines multiple strategies with ML confirmation.
    """
    
    def __init__(self, params: Dict[str, Any], ml_model=None):
        self.params = params
        self.ml_model = ml_model
        self.last_ml_confidence = 0.0
    
    def generate_signal(
        self,
        df: pd.DataFrame,
        df_htf: Optional[pd.DataFrame] = None,
        position: Optional[Dict] = None,
        global_sentiment: float = 0.5,
        funding_rate: float = 0.0
    ) -> Signal:
        """
        Generate trading signal by combining multiple strategies.
        
        Args:
            df: Primary timeframe OHLCV data
            df_htf: Higher timeframe data for trend filter
            position: Current position if any
            global_sentiment: Market sentiment (0=bearish, 1=bullish)
            funding_rate: Current funding rate
        
        Returns:
            Signal object with trade details
        """
        # Calculate indicators
        inds = calculate_indicators(df, self.params)
        
        # Higher timeframe trend
        htf_trend = 0
        if df_htf is not None and len(df_htf) > 50:
            htf_trend = calculate_htf_trend(df_htf, self.params)
        
        current_price = inds['current_price']
        atr = inds['atr']
        
        # Initialize signal
        direction = 'neutral'
        score = 0.0
        reasons = []
        
        # Check for exit signals first if in position
        if position and position.get('amt', 0) != 0:
            exit_signal = self._check_exit_signals(inds, position)
            if exit_signal:
                return exit_signal
        
        # Only look for entries if no position
        if not position or position.get('amt', 0) == 0:
            # Strategy 1: Trend Following
            trend_signal, trend_score, trend_reason = self._trend_following(inds, htf_trend)
            
            # Strategy 2: Momentum Breakout
            mom_signal, mom_score, mom_reason = self._momentum_breakout(inds)
            
            # Strategy 3: Mean Reversion (only in ranging markets)
            mr_signal, mr_score, mr_reason = self._mean_reversion(inds)
            
            # Combine signals
            signals = [
                (trend_signal, trend_score, trend_reason, 0.5),  # 50% weight
                (mom_signal, mom_score, mom_reason, 0.3),        # 30% weight
                (mr_signal, mr_score, mr_reason, 0.2),           # 20% weight
            ]
            
            # Calculate weighted score
            long_score = 0.0
            short_score = 0.0
            
            for sig, sc, reason, weight in signals:
                if sig == 'long':
                    long_score += sc * weight
                    if sc > 0:
                        reasons.append(reason)
                elif sig == 'short':
                    short_score += sc * weight
                    if sc > 0:
                        reasons.append(reason)
            
            # Apply filters
            long_score = self._apply_filters(long_score, 'long', inds, htf_trend, global_sentiment, funding_rate)
            short_score = self._apply_filters(short_score, 'short', inds, htf_trend, global_sentiment, funding_rate)
            
            # Determine direction
            min_score = 4.0  # Minimum score to trade
            
            if long_score >= min_score and long_score > short_score:
                direction = 'long'
                score = long_score
            elif short_score >= min_score and short_score > long_score:
                direction = 'short'
                score = short_score
            
            # ML Confirmation
            if direction != 'neutral' and self.ml_model is not None:
                direction, score = self._ml_filter(direction, score, inds, df)
        
        # Calculate stop loss and take profit
        sl_mult = self.params.get('sl_atr_multiplier', 1.5)
        tp_mult = self.params.get('tp_atr_multiplier', 3.5)
        
        if direction == 'long':
            stop_loss = current_price - (atr * sl_mult)
            take_profit = current_price + (atr * tp_mult)
        elif direction == 'short':
            stop_loss = current_price + (atr * sl_mult)
            take_profit = current_price - (atr * tp_mult)
        else:
            stop_loss = 0
            take_profit = 0
        
        return Signal(
            symbol='',  # Set externally
            direction=direction,
            score=score,
            reason=' | '.join(reasons) if reasons else 'No signal',
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            indicators=inds
        )
    
    def _trend_following(self, inds: Dict, htf_trend: int) -> Tuple[str, float, str]:
        """
        Trend following strategy using SuperTrend and EMA.
        """
        supertrend_dir = inds['supertrend_dir']
        supertrend_slow_dir = inds['supertrend_slow_dir']
        ema_fast = inds['ema_fast']
        ema_slow = inds['ema_slow']
        ema_200 = inds['ema_200']
        price = inds['current_price']
        adx = inds['adx']
        
        adx_threshold = self.params.get('adx_threshold', 20)
        
        signal = 'neutral'
        score = 0.0
        reason = ''
        
        # No trend - no trade
        if adx < adx_threshold:
            return signal, score, 'ADX too low'
        
        # Long conditions
        if supertrend_dir == 1 and ema_fast > ema_slow and price > ema_200:
            signal = 'long'
            score = 5.0
            reason = 'TREND_LONG'
            
            # Bonus for alignment
            if supertrend_slow_dir == 1:
                score += 1.5
                reason += ' (Slow ST aligned)'
            if htf_trend == 1:
                score += 1.0
                reason += ' (HTF aligned)'
            
            # ADX strength bonus
            if adx > 30:
                score += 0.5
            if adx > 40:
                score += 0.5
        
        # Short conditions
        elif supertrend_dir == -1 and ema_fast < ema_slow and price < ema_200:
            signal = 'short'
            score = 5.0
            reason = 'TREND_SHORT'
            
            if supertrend_slow_dir == -1:
                score += 1.5
                reason += ' (Slow ST aligned)'
            if htf_trend == -1:
                score += 1.0
                reason += ' (HTF aligned)'
            
            if adx > 30:
                score += 0.5
            if adx > 40:
                score += 0.5
        
        return signal, score, reason
    
    def _momentum_breakout(self, inds: Dict) -> Tuple[str, float, str]:
        """
        Momentum breakout strategy using RSI, MACD, and volume.
        """
        rsi = inds['rsi']
        rsi_prev = inds['rsi_prev']
        macd_hist = inds['macd_hist']
        macd_hist_prev = inds['macd_hist_prev']
        volume_ratio = inds['volume_ratio']
        stoch_k = inds['stoch_k']
        stoch_d = inds['stoch_d']
        stoch_k_prev = inds['stoch_k_prev']
        stoch_d_prev = inds['stoch_d_prev']
        adx = inds['adx']
        adx_slope = inds['adx_slope']
        
        signal = 'neutral'
        score = 0.0
        reason = ''
        
        # Long breakout
        if (rsi > 50 and rsi_prev <= 50 and  # RSI crossing above 50
            macd_hist > 0 and macd_hist_prev <= 0 and  # MACD histogram turning positive
            volume_ratio > 1.2):  # Above average volume
            
            signal = 'long'
            score = 4.5
            reason = 'MOM_BREAKOUT_LONG'
            
            # Stochastic confirmation
            if stoch_k > stoch_d and stoch_k_prev <= stoch_d_prev:
                score += 1.0
            
            # ADX rising
            if adx_slope > 0:
                score += 0.5
        
        # Short breakout
        elif (rsi < 50 and rsi_prev >= 50 and
              macd_hist < 0 and macd_hist_prev >= 0 and
              volume_ratio > 1.2):
            
            signal = 'short'
            score = 4.5
            reason = 'MOM_BREAKOUT_SHORT'
            
            if stoch_k < stoch_d and stoch_k_prev >= stoch_d_prev:
                score += 1.0
            
            if adx_slope > 0:
                score += 0.5
        
        return signal, score, reason
    
    def _mean_reversion(self, inds: Dict) -> Tuple[str, float, str]:
        """
        Mean reversion strategy for ranging markets.
        """
        price = inds['current_price']
        bb_lower = inds['bb_lower']
        bb_upper = inds['bb_upper']
        bb_mid = inds['bb_mid']
        rsi = inds['rsi']
        chop = inds['chop']
        adx = inds['adx']
        stoch_k = inds['stoch_k']
        
        signal = 'neutral'
        score = 0.0
        reason = ''
        
        # Only in ranging markets
        if adx > 25 or chop < 40:
            return signal, score, 'Market trending, skip MR'
        
        rsi_oversold = self.params.get('rsi_oversold', 30)
        rsi_overbought = self.params.get('rsi_overbought', 70)
        
        # Long: Price at lower BB + RSI oversold
        if price <= bb_lower and rsi < rsi_oversold and stoch_k < 20:
            signal = 'long'
            score = 4.0
            reason = f'MR_LONG (RSI={rsi:.0f}, BB touch)'
            
            # Extra oversold
            if rsi < 25:
                score += 0.5
            if stoch_k < 10:
                score += 0.5
        
        # Short: Price at upper BB + RSI overbought
        elif price >= bb_upper and rsi > rsi_overbought and stoch_k > 80:
            signal = 'short'
            score = 4.0
            reason = f'MR_SHORT (RSI={rsi:.0f}, BB touch)'
            
            if rsi > 75:
                score += 0.5
            if stoch_k > 90:
                score += 0.5
        
        return signal, score, reason
    
    def _apply_filters(
        self,
        score: float,
        direction: str,
        inds: Dict,
        htf_trend: int,
        sentiment: float,
        funding_rate: float
    ) -> float:
        """Apply filters to adjust score."""
        
        # HTF trend filter
        if direction == 'long' and htf_trend == -1:
            score *= 0.5  # Reduce score for counter-trend
        elif direction == 'short' and htf_trend == 1:
            score *= 0.5
        
        # Sentiment filter
        if direction == 'long' and sentiment < 0.3:
            score *= 0.7
        elif direction == 'short' and sentiment > 0.7:
            score *= 0.7
        
        # Funding rate filter
        if direction == 'long':
            if funding_rate < -0.01:  # Shorts paying - good for longs
                score *= 1.1
            elif funding_rate > 0.03:  # Crowded longs - danger
                score *= 0.7
        elif direction == 'short':
            if funding_rate > 0.01:  # Longs paying - good for shorts
                score *= 1.1
            elif funding_rate < -0.03:  # Crowded shorts - danger
                score *= 0.7
        
        # Heikin Ashi confirmation
        if direction == 'long' and not inds.get('ha_bullish', True):
            score *= 0.8
        elif direction == 'short' and inds.get('ha_bullish', False):
            score *= 0.8
        
        return score
    
    def _ml_filter(
        self,
        direction: str,
        score: float,
        inds: Dict,
        df: pd.DataFrame
    ) -> Tuple[str, float]:
        """Apply ML model filter."""
        if self.ml_model is None:
            return direction, score
        
        try:
            # Prepare features
            features = self._prepare_ml_features(inds, df)
            if features is None:
                return direction, score
            
            # Predict
            prediction = self.ml_model.predict(features)[0]
            probabilities = self.ml_model.predict_proba(features)[0]
            
            classes = self.ml_model.classes_
            pred_idx = list(classes).index(prediction)
            confidence = probabilities[pred_idx]
            
            self.last_ml_confidence = confidence
            
            # Filter based on ML
            if direction == 'long' and prediction != 1:
                if confidence > 0.7:
                    return 'neutral', 0
                score *= 0.7
            elif direction == 'short' and prediction != -1:
                if confidence > 0.7:
                    return 'neutral', 0
                score *= 0.7
            else:
                # ML confirms
                if confidence > 0.8:
                    score *= 1.3
                elif confidence > 0.6:
                    score *= 1.1
            
            return direction, score
            
        except Exception as e:
            return direction, score
    
    def _prepare_ml_features(self, inds: Dict, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Prepare features for ML prediction."""
        if len(df) < 3:
            return None
        
        try:
            features = {
                'rsi': inds['rsi'],
                'atr': inds['atr'],
                'adx': inds['adx'],
                'ema_fast': inds['ema_fast'],
                'ema_slow': inds['ema_slow'],
                'bb_upper': inds['bb_upper'],
                'bb_lower': inds['bb_lower'],
                'volume_ratio': inds['volume_ratio'],
                'stoch_k': inds['stoch_k'],
                'macd_hist': inds['macd_hist'],
                'momentum': inds['momentum'],
                'rsi_prev': inds['rsi_prev'],
                'adx_slope': inds['adx_slope'],
            }
            
            return pd.DataFrame([features])
        except:
            return None
    
    def _check_exit_signals(
        self,
        inds: Dict,
        position: Dict
    ) -> Optional[Signal]:
        """Check for exit signals on existing position."""
        
        amt = position.get('amt', 0)
        entry = position.get('entry', 0)
        
        if amt == 0 or entry == 0:
            return None
        
        current_price = inds['current_price']
        atr = inds['atr']
        rsi = inds['rsi']
        supertrend_dir = inds['supertrend_dir']
        adx = inds['adx']
        volume_ratio = inds['volume_ratio']
        
        side = 'long' if amt > 0 else 'short'
        pnl_per_unit = (current_price - entry) if side == 'long' else (entry - current_price)
        roi = pnl_per_unit / entry if entry > 0 else 0
        
        exit_direction = 'neutral'
        exit_reason = ''
        exit_score = 0
        
        # 1. Volume Climax Exit
        if volume_ratio > 3.0:
            if side == 'long' and rsi > 80:
                exit_direction = 'short'  # Close long
                exit_reason = 'EXIT_CLIMAX_PUMP'
                exit_score = 10
            elif side == 'short' and rsi < 20:
                exit_direction = 'long'  # Close short
                exit_reason = 'EXIT_CLIMAX_DUMP'
                exit_score = 10
        
        # 2. Trend Reversal Exit
        if exit_score == 0:
            if (side == 'long' and supertrend_dir == -1) or \
               (side == 'short' and supertrend_dir == 1):
                if adx > 20:
                    exit_direction = 'short' if side == 'long' else 'long'
                    exit_reason = 'EXIT_TREND_REVERSAL'
                    exit_score = 9
        
        # 3. RSI Extreme Exit (in weak trend)
        if exit_score == 0 and roi > 0.01 and adx < 30:
            if side == 'long' and rsi > 75:
                exit_direction = 'short'
                exit_reason = f'EXIT_RSI_EXTREME ({rsi:.0f})'
                exit_score = 7
            elif side == 'short' and rsi < 25:
                exit_direction = 'long'
                exit_reason = f'EXIT_RSI_EXTREME ({rsi:.0f})'
                exit_score = 7
        
        # 4. Dynamic Take Profit
        tp_mult = self.params.get('tp_atr_multiplier', 3.5)
        if adx > 50:
            tp_mult = 6.0
        elif adx > 30:
            tp_mult = 4.5
        
        tp_distance = atr * tp_mult
        
        if exit_score == 0 and pnl_per_unit > tp_distance:
            # Check for momentum fading
            if (side == 'long' and rsi < inds['rsi_prev']) or \
               (side == 'short' and rsi > inds['rsi_prev']):
                exit_direction = 'short' if side == 'long' else 'long'
                exit_reason = f'EXIT_TP_DYNAMIC ({tp_mult:.1f}x ATR)'
                exit_score = 8
        
        if exit_score > 0:
            return Signal(
                symbol='',
                direction=exit_direction,
                score=exit_score,
                reason=exit_reason,
                entry_price=current_price,
                stop_loss=0,
                take_profit=0,
                indicators=inds
            )
        
        return None


def analyze_symbol(
    symbol: str,
    df: pd.DataFrame,
    df_htf: Optional[pd.DataFrame],
    position: Optional[Dict],
    params: Dict,
    global_sentiment: float = 0.5,
    funding_rate: float = 0.0,
    ml_model=None
) -> Dict[str, Any]:
    """
    Analyze a single symbol and return trade recommendation.
    
    Returns:
        Dict with analysis results and potential action
    """
    engine = StrategyEngine(params, ml_model)
    
    signal = engine.generate_signal(
        df=df,
        df_htf=df_htf,
        position=position,
        global_sentiment=global_sentiment,
        funding_rate=funding_rate
    )
    
    signal.symbol = symbol
    
    # Build action if signal is valid
    action = None
    
    if signal.direction in ['long', 'short']:
        if position and position.get('amt', 0) != 0:
            # Exit action
            current_side = 'long' if position['amt'] > 0 else 'short'
            if (signal.direction == 'short' and current_side == 'long') or \
               (signal.direction == 'long' and current_side == 'short'):
                action = {
                    'symbol': symbol,
                    'side': 'sell' if current_side == 'long' else 'buy',
                    'amount': abs(position['amt']),
                    'price': signal.entry_price,
                    'reason': signal.reason,
                    'score': signal.score,
                    'reduceOnly': True
                }
        else:
            # Entry action
            action = {
                'symbol': symbol,
                'side': 'buy' if signal.direction == 'long' else 'sell',
                'price': signal.entry_price,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'reason': signal.reason,
                'score': signal.score,
                'reduceOnly': False
            }
    
    return {
        'symbol': symbol,
        'signal': signal,
        'action': action,
        'indicators': signal.indicators,
        'ml_confidence': engine.last_ml_confidence
    }
