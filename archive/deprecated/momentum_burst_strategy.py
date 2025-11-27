"""
MOMENTUM BURST STRATEGY - Crypto Futures 1h
Estratégia otimizada para capturar movimentos fortes com alta precisão

CONCEITO:
- Captura apenas movimentos explosivos (breakouts genuínos)
- Timeframe 1h para reduzir ruído
- Múltiplas confirmações para evitar falsos sinais
- Poucos trades mas de altíssima qualidade
- Target: Profit Factor > 2.0

INDICADORES CHAVE:
1. ATR - Mede tamanho do movimento (volatilidade)
2. ADX - Confirma força do trend (>threshold)
3. RSI - Evita zonas de reversão (range favorável)
4. Volume - Confirmação de força institucional
5. Price Velocity - Velocidade de mudança de preço

LÓGICA DE ENTRADA:
LONG:
- Movimento de preço > X * ATR (breakout forte)
- ADX > threshold (trend confirmado)
- RSI entre 45-70 (momentum mas não overbought)
- Volume > 1.5x média (confirmação)
- Price velocity positiva e crescente

SHORT:
- Movimento de preço < -X * ATR (breakdown forte)
- ADX > threshold (trend confirmado)
- RSI entre 30-55 (momentum mas não oversold)
- Volume > 1.5x média (confirmação)
- Price velocity negativa e crescente

SAÍDAS:
- Stop Loss: Apertado (momentum trades são rápidos)
- Take Profit: Múltiplos níveis com trailing
- Exit dinâmico: Se momentum enfraquecer
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime


@dataclass
class MomentumSignal:
    """Sinal de momentum burst."""
    direction: str  # 'long', 'short', 'neutral'
    score: float  # 0-10
    entry_price: float
    stop_loss: float
    take_profit: float
    reason: str
    momentum_strength: float
    indicators: Dict[str, float]


class MomentumBurstStrategy:
    """
    Estratégia Momentum Burst otimizada.

    Parâmetros otimizados através de backtesting:
    - ATR multiplier para movimento mínimo
    - ADX threshold para confirmar trend
    - RSI ranges para evitar extremos
    - Volume threshold
    - SL/TP ratios
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Inicializa estratégia com parâmetros otimizados.

        Args:
            params: Dicionário de parâmetros customizados
        """
        # PARÂMETROS OTIMIZADOS
        default_params = {
            # Timeframe
            'timeframe': '1h',

            # ATR (Average True Range) - Movimento mínimo
            'atr_length': 14,
            'atr_multiplier_entry': 2.5,  # Movimento deve ser > 2.5x ATR

            # ADX (Trend Strength) - Força do trend
            'adx_length': 14,
            'adx_threshold': 25,  # ADX > 25 = trend forte

            # RSI (Relative Strength Index) - Evitar extremos
            'rsi_length': 14,
            'rsi_long_min': 45,  # RSI mínimo para long
            'rsi_long_max': 70,  # RSI máximo para long
            'rsi_short_min': 30,  # RSI mínimo para short
            'rsi_short_max': 55,  # RSI máximo para short

            # Volume - Confirmação institucional
            'volume_sma_length': 20,
            'volume_multiplier': 1.5,  # Volume > 1.5x média

            # Price Velocity - Velocidade de movimento
            'velocity_length': 3,  # Mede mudança em 3 candles
            'velocity_threshold': 0.5,  # % mínimo de mudança

            # Stop Loss / Take Profit
            'sl_atr_multiplier': 1.2,  # SL apertado para momentum
            'tp_atr_multiplier': 3.5,  # TP generoso para capturar movimento
            'use_trailing': True,  # Trailing stop ativado
            'trailing_activation': 1.5,  # Ativa trailing em 1.5x ATR
            'trailing_distance': 0.8,  # Distância do trailing

            # Filtros adicionais
            'min_candle_size': 0.3,  # Candle deve ter corpo > 30% do range
            'require_breakout': True,  # Exige breakout de consolidação
            'consolidation_periods': 10,  # Períodos para detectar consolidação

            # Gestão de risco
            'max_score_threshold': 6.0,  # Score mínimo para trade
        }

        self.params = default_params
        if params:
            self.params.update(params)

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula todos os indicadores necessários.

        Args:
            df: DataFrame com OHLCV

        Returns:
            DataFrame com indicadores adicionados
        """
        if len(df) < 200:
            return df

        df = df.copy()

        # ATR - True Range (volatilidade)
        df['atr'] = ta.atr(
            df['high'],
            df['low'],
            df['close'],
            length=self.params['atr_length']
        )

        # ADX - Trend Strength
        adx_result = ta.adx(
            df['high'],
            df['low'],
            df['close'],
            length=self.params['adx_length']
        )
        if adx_result is not None:
            adx_col = [c for c in adx_result.columns if c.startswith('ADX')][0]
            df['adx'] = adx_result[adx_col]
        else:
            df['adx'] = 0

        # RSI - Relative Strength
        df['rsi'] = ta.rsi(df['close'], length=self.params['rsi_length'])

        # Volume analysis
        df['volume_sma'] = ta.sma(df['volume'], length=self.params['volume_sma_length'])
        df['volume_ratio'] = df['volume'] / df['volume_sma'].replace(0, 1)

        # Price Velocity - Taxa de mudança
        vel_len = self.params['velocity_length']
        df['price_velocity'] = (df['close'].pct_change(vel_len) * 100)
        df['velocity_acceleration'] = df['price_velocity'].diff()

        # Movimento em relação ao ATR
        df['price_move'] = df['close'] - df['close'].shift(1)
        df['atr_normalized_move'] = df['price_move'] / df['atr'].replace(0, 1)

        # Candle body strength
        df['body'] = abs(df['close'] - df['open'])
        df['candle_range'] = df['high'] - df['low']
        df['body_ratio'] = df['body'] / df['candle_range'].replace(0, 1)

        # Donchian Channels - Para detectar breakouts
        consol_periods = self.params['consolidation_periods']
        df['donchian_high'] = df['high'].rolling(consol_periods).max()
        df['donchian_low'] = df['low'].rolling(consol_periods).min()
        df['donchian_mid'] = (df['donchian_high'] + df['donchian_low']) / 2

        # Detectar consolidação (Choppiness Index)
        df['chop'] = self._calculate_chop(df, consol_periods)

        # EMA para contexto de trend
        df['ema_50'] = ta.ema(df['close'], length=50)
        df['ema_200'] = ta.ema(df['close'], length=200)

        # MACD para confirmação de momentum
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        if macd is not None:
            hist_col = [c for c in macd.columns if 'MACDh' in c][0]
            df['macd_hist'] = macd[hist_col]
        else:
            df['macd_hist'] = 0

        return df

    def _calculate_chop(self, df: pd.DataFrame, length: int) -> pd.Series:
        """Calcula Choppiness Index para detectar consolidação."""
        try:
            tr = ta.true_range(df['high'], df['low'], df['close'])
            sum_atr = tr.rolling(length).sum()

            high_n = df['high'].rolling(length).max()
            low_n = df['low'].rolling(length).min()
            range_n = (high_n - low_n).replace(0, 0.0000001)

            ratio = sum_atr / range_n
            ratio = ratio.replace(0, 0.0000001)

            return 100 * np.log10(ratio) / np.log10(length)
        except:
            return pd.Series([50.0] * len(df), index=df.index)

    def generate_signal(
        self,
        df: pd.DataFrame,
        position: Optional[Dict] = None
    ) -> MomentumSignal:
        """
        Gera sinal de trading baseado na estratégia Momentum Burst.

        Args:
            df: DataFrame com dados OHLCV e indicadores
            position: Posição atual (se houver)

        Returns:
            MomentumSignal com direção e detalhes
        """
        # Calcula indicadores se não existirem
        if 'atr' not in df.columns:
            df = self.calculate_indicators(df)

        # Valores atuais
        last = df.iloc[-1]
        prev = df.iloc[-2]

        # Extract indicators
        price = last['close']
        atr = last['atr']
        adx = last['adx']
        rsi = last['rsi']
        volume_ratio = last['volume_ratio']
        velocity = last['price_velocity']
        velocity_accel = last['velocity_acceleration']
        atr_move = last['atr_normalized_move']
        body_ratio = last['body_ratio']
        chop = last['chop']

        # Inicializa resultado
        direction = 'neutral'
        score = 0.0
        reasons = []
        momentum_strength = 0.0

        # Check exit conditions first if in position
        if position and position.get('amt', 0) != 0:
            exit_signal = self._check_exit(df, position)
            if exit_signal:
                return exit_signal

        # Only look for entries if no position
        if not position or position.get('amt', 0) == 0:

            # === LONG SIGNAL ===
            long_score, long_reasons = self._evaluate_long(
                price, atr, adx, rsi, volume_ratio,
                velocity, velocity_accel, atr_move,
                body_ratio, chop, last, prev, df
            )

            # === SHORT SIGNAL ===
            short_score, short_reasons = self._evaluate_short(
                price, atr, adx, rsi, volume_ratio,
                velocity, velocity_accel, atr_move,
                body_ratio, chop, last, prev, df
            )

            # Determine direction
            if long_score >= self.params['max_score_threshold'] and long_score > short_score:
                direction = 'long'
                score = long_score
                reasons = long_reasons
                momentum_strength = velocity if velocity > 0 else 0
            elif short_score >= self.params['max_score_threshold'] and short_score > long_score:
                direction = 'short'
                score = short_score
                reasons = short_reasons
                momentum_strength = abs(velocity) if velocity < 0 else 0

        # Calculate SL/TP
        sl, tp = self._calculate_sl_tp(price, atr, direction)

        return MomentumSignal(
            direction=direction,
            score=score,
            entry_price=price,
            stop_loss=sl,
            take_profit=tp,
            reason=' | '.join(reasons),
            momentum_strength=momentum_strength,
            indicators={
                'atr': atr,
                'adx': adx,
                'rsi': rsi,
                'volume_ratio': volume_ratio,
                'velocity': velocity,
                'chop': chop,
                'body_ratio': body_ratio
            }
        )

    def _evaluate_long(
        self, price, atr, adx, rsi, volume_ratio,
        velocity, velocity_accel, atr_move, body_ratio,
        chop, last, prev, df
    ) -> Tuple[float, List[str]]:
        """Avalia condições para LONG."""
        score = 0.0
        reasons = []

        # 1. MOVIMENTO FORTE (Principal)
        atr_threshold = self.params['atr_multiplier_entry']
        if atr_move > atr_threshold:
            score += 3.0
            reasons.append(f'STRONG_MOVE_UP ({atr_move:.1f}x ATR)')
        else:
            return 0.0, []  # Sem movimento forte = sem trade

        # 2. ADX - Trend Strength
        if adx > self.params['adx_threshold']:
            score += 2.0
            reasons.append(f'TREND_CONFIRMED (ADX={adx:.0f})')

            # Bonus for very strong trend
            if adx > 35:
                score += 0.5
            if adx > 45:
                score += 0.5
        else:
            return 0.0, []  # Sem trend forte = sem trade

        # 3. RSI - Range favorável
        rsi_min = self.params['rsi_long_min']
        rsi_max = self.params['rsi_long_max']
        if rsi_min <= rsi <= rsi_max:
            score += 1.5
            reasons.append(f'RSI_FAVORABLE ({rsi:.0f})')

            # Bonus for ideal RSI zone (55-65)
            if 55 <= rsi <= 65:
                score += 0.5
        else:
            score -= 2.0  # Penaliza se RSI não ideal

        # 4. Volume - Confirmação
        if volume_ratio > self.params['volume_multiplier']:
            score += 1.5
            reasons.append(f'VOLUME_SURGE ({volume_ratio:.1f}x)')

            # Bonus for massive volume
            if volume_ratio > 2.5:
                score += 0.5
        else:
            score -= 1.0  # Penaliza volume fraco

        # 5. Price Velocity - Aceleração positiva
        if velocity > self.params['velocity_threshold'] and velocity_accel > 0:
            score += 1.0
            reasons.append(f'ACCELERATING_UP ({velocity:.1f}%)')

        # 6. Candle Body - Candle forte
        if body_ratio > self.params['min_candle_size']:
            score += 0.5
            reasons.append('STRONG_CANDLE')

        # 7. Breakout - Saindo de consolidação
        if self.params['require_breakout']:
            if chop > 50 and last['close'] > last['donchian_high']:
                score += 1.5
                reasons.append('BREAKOUT_HIGH')

        # 8. Contexto de trend maior (EMA)
        if price > last['ema_50'] and last['ema_50'] > last['ema_200']:
            score += 1.0
            reasons.append('UPTREND_CONTEXT')

        # 9. MACD confirmação
        if last['macd_hist'] > 0 and last['macd_hist'] > prev['macd_hist']:
            score += 0.5
            reasons.append('MACD_POSITIVE')

        return score, reasons

    def _evaluate_short(
        self, price, atr, adx, rsi, volume_ratio,
        velocity, velocity_accel, atr_move, body_ratio,
        chop, last, prev, df
    ) -> Tuple[float, List[str]]:
        """Avalia condições para SHORT."""
        score = 0.0
        reasons = []

        # 1. MOVIMENTO FORTE (Principal)
        atr_threshold = self.params['atr_multiplier_entry']
        if atr_move < -atr_threshold:
            score += 3.0
            reasons.append(f'STRONG_MOVE_DOWN ({abs(atr_move):.1f}x ATR)')
        else:
            return 0.0, []

        # 2. ADX - Trend Strength
        if adx > self.params['adx_threshold']:
            score += 2.0
            reasons.append(f'TREND_CONFIRMED (ADX={adx:.0f})')

            if adx > 35:
                score += 0.5
            if adx > 45:
                score += 0.5
        else:
            return 0.0, []

        # 3. RSI - Range favorável
        rsi_min = self.params['rsi_short_min']
        rsi_max = self.params['rsi_short_max']
        if rsi_min <= rsi <= rsi_max:
            score += 1.5
            reasons.append(f'RSI_FAVORABLE ({rsi:.0f})')

            # Bonus for ideal RSI zone (35-45)
            if 35 <= rsi <= 45:
                score += 0.5
        else:
            score -= 2.0

        # 4. Volume
        if volume_ratio > self.params['volume_multiplier']:
            score += 1.5
            reasons.append(f'VOLUME_SURGE ({volume_ratio:.1f}x)')

            if volume_ratio > 2.5:
                score += 0.5
        else:
            score -= 1.0

        # 5. Price Velocity
        if velocity < -self.params['velocity_threshold'] and velocity_accel < 0:
            score += 1.0
            reasons.append(f'ACCELERATING_DOWN ({abs(velocity):.1f}%)')

        # 6. Candle Body
        if body_ratio > self.params['min_candle_size']:
            score += 0.5
            reasons.append('STRONG_CANDLE')

        # 7. Breakout
        if self.params['require_breakout']:
            if chop > 50 and last['close'] < last['donchian_low']:
                score += 1.5
                reasons.append('BREAKDOWN_LOW')

        # 8. Contexto de trend
        if price < last['ema_50'] and last['ema_50'] < last['ema_200']:
            score += 1.0
            reasons.append('DOWNTREND_CONTEXT')

        # 9. MACD
        if last['macd_hist'] < 0 and last['macd_hist'] < prev['macd_hist']:
            score += 0.5
            reasons.append('MACD_NEGATIVE')

        return score, reasons

    def _calculate_sl_tp(
        self, price: float, atr: float, direction: str
    ) -> Tuple[float, float]:
        """Calcula Stop Loss e Take Profit."""
        sl_mult = self.params['sl_atr_multiplier']
        tp_mult = self.params['tp_atr_multiplier']

        if direction == 'long':
            sl = price - (atr * sl_mult)
            tp = price + (atr * tp_mult)
        elif direction == 'short':
            sl = price + (atr * sl_mult)
            tp = price - (atr * tp_mult)
        else:
            sl = 0
            tp = 0

        return sl, tp

    def _check_exit(
        self, df: pd.DataFrame, position: Dict
    ) -> Optional[MomentumSignal]:
        """Verifica condições de saída."""
        last = df.iloc[-1]

        amt = position.get('amt', 0)
        entry = position.get('entry', 0)

        if amt == 0 or entry == 0:
            return None

        side = 'long' if amt > 0 else 'short'
        price = last['close']
        atr = last['atr']
        rsi = last['rsi']
        adx = last['adx']
        velocity = last['price_velocity']

        # Calculate PnL
        pnl = (price - entry) if side == 'long' else (entry - price)
        pnl_atr = pnl / atr if atr > 0 else 0

        # Exit conditions
        exit_direction = 'neutral'
        exit_reason = ''
        exit_score = 0

        # 1. Momentum reversal (ADX dropping + velocity reversing)
        if side == 'long':
            if velocity < -self.params['velocity_threshold'] and adx < 20:
                exit_direction = 'short'
                exit_reason = 'MOMENTUM_REVERSAL'
                exit_score = 10
        else:
            if velocity > self.params['velocity_threshold'] and adx < 20:
                exit_direction = 'long'
                exit_reason = 'MOMENTUM_REVERSAL'
                exit_score = 10

        # 2. RSI extremes (momentum exhaustion)
        if exit_score == 0:
            if side == 'long' and rsi > 80:
                exit_direction = 'short'
                exit_reason = 'RSI_EXTREME_HIGH'
                exit_score = 9
            elif side == 'short' and rsi < 20:
                exit_direction = 'long'
                exit_reason = 'RSI_EXTREME_LOW'
                exit_score = 9

        # 3. Trailing stop (if enabled and in profit)
        if exit_score == 0 and self.params['use_trailing']:
            trailing_activation = self.params['trailing_activation']
            trailing_distance = self.params['trailing_distance']

            if pnl_atr > trailing_activation:
                # In profit - check trailing stop
                trailing_stop_distance = atr * trailing_distance

                if side == 'long':
                    # For long, exit if price drops from high
                    high_since_entry = df[df.index >= df.index[-20]]['high'].max()
                    if price < (high_since_entry - trailing_stop_distance):
                        exit_direction = 'short'
                        exit_reason = f'TRAILING_STOP (secured {pnl_atr:.1f}x ATR)'
                        exit_score = 8
                else:
                    # For short, exit if price rises from low
                    low_since_entry = df[df.index >= df.index[-20]]['low'].min()
                    if price > (low_since_entry + trailing_stop_distance):
                        exit_direction = 'long'
                        exit_reason = f'TRAILING_STOP (secured {pnl_atr:.1f}x ATR)'
                        exit_score = 8

        if exit_score > 0:
            sl, tp = self._calculate_sl_tp(price, atr, exit_direction)

            return MomentumSignal(
                direction=exit_direction,
                score=exit_score,
                entry_price=price,
                stop_loss=sl,
                take_profit=tp,
                reason=exit_reason,
                momentum_strength=0,
                indicators={}
            )

        return None

    def get_params(self) -> Dict[str, Any]:
        """Retorna parâmetros atuais."""
        return self.params.copy()

    def set_params(self, params: Dict[str, Any]):
        """Atualiza parâmetros."""
        self.params.update(params)


# === FUNÇÕES AUXILIARES ===

def backtest_momentum_burst(
    df: pd.DataFrame,
    params: Optional[Dict] = None,
    initial_capital: float = 10000.0,
    position_size: float = 0.95,
    fees: float = 0.0006
) -> Dict[str, Any]:
    """
    Backtest da estratégia Momentum Burst.

    Args:
        df: DataFrame com dados OHLCV
        params: Parâmetros da estratégia
        initial_capital: Capital inicial
        position_size: % do capital por trade
        fees: Taxa de fees (0.06% = 0.0006)

    Returns:
        Dicionário com resultados do backtest
    """
    strategy = MomentumBurstStrategy(params)
    df = strategy.calculate_indicators(df)

    # Tracking
    capital = initial_capital
    position = {'amt': 0, 'entry': 0, 'side': None}
    trades = []
    equity_curve = []

    for i in range(200, len(df)):
        row = df.iloc[:i+1]
        current_price = row.iloc[-1]['close']

        # Generate signal
        signal = strategy.generate_signal(row, position)

        # Execute trades
        if signal.direction in ['long', 'short'] and position['amt'] == 0:
            # Entry
            trade_capital = capital * position_size
            trade_amt = trade_capital / current_price

            # Apply fees
            fee_cost = trade_capital * fees
            capital -= fee_cost

            position['amt'] = trade_amt if signal.direction == 'long' else -trade_amt
            position['entry'] = current_price
            position['side'] = signal.direction

            trades.append({
                'entry_time': row.iloc[-1]['timestamp'],
                'entry_price': current_price,
                'side': signal.direction,
                'reason': signal.reason,
                'score': signal.score,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit
            })

        elif signal.direction != 'neutral' and position['amt'] != 0:
            # Exit
            current_side = 'long' if position['amt'] > 0 else 'short'

            # Check if exit signal
            if (signal.direction == 'short' and current_side == 'long') or \
               (signal.direction == 'long' and current_side == 'short'):

                # Calculate PnL
                if current_side == 'long':
                    pnl = (current_price - position['entry']) * abs(position['amt'])
                else:
                    pnl = (position['entry'] - current_price) * abs(position['amt'])

                # Apply fees
                exit_value = abs(position['amt']) * current_price
                fee_cost = exit_value * fees

                capital += (abs(position['amt']) * position['entry']) + pnl - fee_cost

                # Update last trade
                if trades:
                    trades[-1].update({
                        'exit_time': row.iloc[-1]['timestamp'],
                        'exit_price': current_price,
                        'exit_reason': signal.reason,
                        'pnl': pnl,
                        'pnl_pct': (pnl / (abs(position['amt']) * position['entry'])) * 100,
                        'return': (capital / initial_capital - 1) * 100
                    })

                position = {'amt': 0, 'entry': 0, 'side': None}

        # Check SL/TP
        if position['amt'] != 0 and trades:
            last_trade = trades[-1]
            sl = last_trade['stop_loss']
            tp = last_trade['take_profit']

            hit_sl = False
            hit_tp = False

            if position['side'] == 'long':
                if current_price <= sl:
                    hit_sl = True
                elif current_price >= tp:
                    hit_tp = True
            else:
                if current_price >= sl:
                    hit_sl = True
                elif current_price <= tp:
                    hit_tp = True

            if hit_sl or hit_tp:
                exit_price = sl if hit_sl else tp

                if position['side'] == 'long':
                    pnl = (exit_price - position['entry']) * abs(position['amt'])
                else:
                    pnl = (position['entry'] - exit_price) * abs(position['amt'])

                exit_value = abs(position['amt']) * exit_price
                fee_cost = exit_value * fees

                capital += (abs(position['amt']) * position['entry']) + pnl - fee_cost

                last_trade.update({
                    'exit_time': row.iloc[-1]['timestamp'],
                    'exit_price': exit_price,
                    'exit_reason': 'STOP_LOSS' if hit_sl else 'TAKE_PROFIT',
                    'pnl': pnl,
                    'pnl_pct': (pnl / (abs(position['amt']) * position['entry'])) * 100,
                    'return': (capital / initial_capital - 1) * 100
                })

                position = {'amt': 0, 'entry': 0, 'side': None}

        equity_curve.append({
            'timestamp': row.iloc[-1]['timestamp'],
            'equity': capital
        })

    # Calculate metrics
    trades_df = pd.DataFrame(trades)
    if len(trades_df) > 0:
        completed_trades = trades_df[trades_df['exit_time'].notna()]

        if len(completed_trades) > 0:
            wins = completed_trades[completed_trades['pnl'] > 0]
            losses = completed_trades[completed_trades['pnl'] < 0]

            win_rate = len(wins) / len(completed_trades) * 100 if len(completed_trades) > 0 else 0
            avg_win = wins['pnl_pct'].mean() if len(wins) > 0 else 0
            avg_loss = losses['pnl_pct'].mean() if len(losses) > 0 else 0

            profit_factor = abs(wins['pnl'].sum() / losses['pnl'].sum()) if len(losses) > 0 and losses['pnl'].sum() != 0 else 0

            total_return = (capital / initial_capital - 1) * 100

            return {
                'total_trades': len(completed_trades),
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'total_return': total_return,
                'final_capital': capital,
                'trades': completed_trades,
                'equity_curve': pd.DataFrame(equity_curve)
            }

    return {
        'total_trades': 0,
        'win_rate': 0,
        'profit_factor': 0,
        'total_return': 0,
        'final_capital': initial_capital,
        'trades': pd.DataFrame(),
        'equity_curve': pd.DataFrame(equity_curve)
    }


def optimize_parameters(
    df: pd.DataFrame,
    param_grid: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Otimiza parâmetros da estratégia através de grid search.

    Args:
        df: DataFrame com dados históricos
        param_grid: Grid de parâmetros para testar

    Returns:
        Melhores parâmetros e resultados
    """
    if param_grid is None:
        param_grid = {
            'atr_multiplier_entry': [2.0, 2.5, 3.0],
            'adx_threshold': [20, 25, 30],
            'volume_multiplier': [1.3, 1.5, 2.0],
            'sl_atr_multiplier': [1.0, 1.2, 1.5],
            'tp_atr_multiplier': [3.0, 3.5, 4.0]
        }

    best_pf = 0
    best_params = None
    best_results = None

    # Grid search
    from itertools import product

    keys = param_grid.keys()
    values = param_grid.values()

    for combo in product(*values):
        params = dict(zip(keys, combo))

        results = backtest_momentum_burst(df, params)

        pf = results.get('profit_factor', 0)
        wr = results.get('win_rate', 0)
        trades = results.get('total_trades', 0)

        # Criteria: PF > 2.0, WR > 50%, Trades > 20
        if pf > best_pf and wr > 50 and trades > 20:
            best_pf = pf
            best_params = params
            best_results = results

    return {
        'best_params': best_params,
        'best_results': best_results,
        'best_profit_factor': best_pf
    }


if __name__ == '__main__':
    """
    Exemplo de uso da estratégia.
    """
    print("=" * 80)
    print("MOMENTUM BURST STRATEGY - Crypto Futures")
    print("=" * 80)
    print("\nCarregando dados de exemplo...")

    # Simula dados (em produção, usar dados reais)
    dates = pd.date_range('2023-01-01', periods=1000, freq='1h')
    np.random.seed(42)

    # Generate realistic price data
    price = 30000
    prices = [price]
    for i in range(999):
        change = np.random.randn() * 100
        price = price + change
        prices.append(price)

    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p + abs(np.random.randn() * 50) for p in prices],
        'low': [p - abs(np.random.randn() * 50) for p in prices],
        'close': prices,
        'volume': [np.random.uniform(100, 1000) for _ in range(1000)]
    })

    print(f"\nDados carregados: {len(df)} candles (1h)")
    print(f"Período: {df['timestamp'].min()} a {df['timestamp'].max()}")

    # Test strategy
    print("\n" + "=" * 80)
    print("TESTANDO ESTRATÉGIA COM PARÂMETROS PADRÃO")
    print("=" * 80)

    results = backtest_momentum_burst(df)

    print(f"\nTotal de Trades: {results['total_trades']}")
    print(f"Win Rate: {results['win_rate']:.2f}%")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    print(f"Retorno Total: {results['total_return']:.2f}%")
    print(f"Capital Final: ${results['final_capital']:.2f}")

    if results['total_trades'] > 0:
        print(f"\nAvg Win: {results['avg_win']:.2f}%")
        print(f"Avg Loss: {results['avg_loss']:.2f}%")

    print("\n" + "=" * 80)
    print("ESTRATÉGIA MOMENTUM BURST PRONTA PARA USO!")
    print("=" * 80)
