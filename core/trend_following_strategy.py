"""
TREND FOLLOWING STRATEGY - Otimizada para Crypto 1h

Estrategia baseada em pesquisa de melhores praticas para crypto:
- EMA cross como base de entrada
- ADX forte para confirmar tendencia (>25 padrao, >30 ideal para crypto)
- Higher timeframe filter para confirmar direcao
- Trailing stop para maximizar ganhos em tendencias fortes

PESQUISA:
- EMAs: 12/26 oferece balanco entre reatividade e reducao de ruido
- ADX: 25-30 threshold (crypto volatil precisa >30 para trends fortes)
- Trailing Stop: Superior a TP fixo em trending markets (maximiza lucros)
- R:R: Minimo 2:1, idealmente 3:1 para compensar win rate ~40%

Referencias:
- https://altfins.com/knowledge-base/ema-12-50-crossovers/
- https://www.mindmathmoney.com/articles/adx-indicator-trading-strategy
- https://www.altrady.com/crypto-trading/technical-analysis/risk-management-trend-following-strategies
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TrendSignal:
    """Sinal de trend following."""
    direction: str  # 'long', 'short', 'none'
    strength: float  # 0-10
    entry_price: float
    stop_loss: float
    take_profit: float
    trailing_stop_pct: float  # Percentual do trailing stop
    reason: str
    confidence: str  # 'high', 'medium', 'low'


class TrendFollowingStrategy:
    """
    Estrategia TREND FOLLOWING otimizada para crypto 1h.

    CARACTERISTICAS:
    - Win Rate esperado: ~35-45% (normal para trend following)
    - R:R esperado: >2.5:1 (compensa baixo win rate)
    - Trailing stop para maximizar ganhos em trends fortes
    - Filtros multiplos para reduzir sinais falsos

    PARAMETROS OTIMIZADOS (baseados em pesquisa):
    - EMA Fast: 12 (reativo mas nao muito ruidoso)
    - EMA Slow: 26 (filtra ruido, capta trends medios)
    - ADX Threshold: 30 (ideal para crypto volatil)
    - SL: 2.5 ATR (protecao adequada)
    - TP: 6.0 ATR (R:R > 2:1)
    - Trailing: 15-25% (para crypto volatil)
    """

    def __init__(self, params: Dict = None):
        self.params = params or {}

        # EMA PERIODS - Pesquisa mostra 12/26 como balanco ideal
        self.ema_fast = self.params.get('ema_fast', 12)
        self.ema_slow = self.params.get('ema_slow', 26)
        self.ema_trend = self.params.get('ema_trend', 200)  # Filtro de trend geral

        # ADX THRESHOLDS - Crypto precisa >30 para trends fortes
        self.adx_min = self.params.get('adx_min', 30)  # Minimo para operar
        self.adx_strong = self.params.get('adx_strong', 40)  # Trend muito forte
        self.adx_peak = self.params.get('adx_peak', 50)  # Pico (cuidado com reversao)

        # ATR MULTIPLIERS - R:R > 2:1
        self.atr_period = self.params.get('atr_period', 14)
        self.sl_atr_mult = self.params.get('sl_atr_mult', 2.5)  # SL conservador
        self.tp_atr_mult = self.params.get('tp_atr_mult', 6.0)  # TP agressivo (R:R 2.4:1)

        # TRAILING STOP - Percentual do preco (nao ATR)
        # Pesquisa recomenda 15-25% para crypto
        self.trailing_stop_pct = self.params.get('trailing_stop_pct', 0.20)  # 20%
        self.trailing_activation_rr = self.params.get('trailing_activation_rr', 1.5)  # Ativar apos 1.5R

        # FILTROS ADICIONAIS
        self.use_htf_filter = self.params.get('use_htf_filter', True)
        self.use_volume_filter = self.params.get('use_volume_filter', True)
        self.min_volume_ratio = self.params.get('min_volume_ratio', 1.0)

        # RSI para evitar extremos (opcional)
        self.use_rsi_filter = self.params.get('use_rsi_filter', True)
        self.rsi_period = self.params.get('rsi_period', 14)
        self.rsi_extreme_low = self.params.get('rsi_extreme_low', 20)
        self.rsi_extreme_high = self.params.get('rsi_extreme_high', 80)

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcular indicadores necessarios."""
        df = df.copy()

        # EMAs
        df['ema_fast'] = df['close'].ewm(span=self.ema_fast, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.ema_slow, adjust=False).mean()
        df['ema_200'] = df['close'].ewm(span=self.ema_trend, adjust=False).mean()

        # ATR (Wilder's smoothing)
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        alpha = 1 / self.atr_period
        df['atr'] = tr.ewm(alpha=alpha, adjust=False).mean()

        # ADX (Wilder's smoothing)
        up = df['high'].diff()
        down = -df['low'].diff()
        plus_dm = up.where((up > down) & (up > 0), 0)
        minus_dm = down.where((down > up) & (down > 0), 0)

        plus_dm_smooth = plus_dm.ewm(alpha=alpha, adjust=False).mean()
        minus_dm_smooth = minus_dm.ewm(alpha=alpha, adjust=False).mean()

        plus_di = 100 * plus_dm_smooth / (df['atr'] + 1e-10)
        minus_di = 100 * minus_dm_smooth / (df['atr'] + 1e-10)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        df['adx'] = dx.ewm(alpha=alpha, adjust=False).mean()
        df['di_plus'] = plus_di
        df['di_minus'] = minus_di

        # RSI (opcional)
        if self.use_rsi_filter:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = (-delta.where(delta < 0, 0))
            alpha_rsi = 1 / self.rsi_period
            avg_gain = gain.ewm(alpha=alpha_rsi, adjust=False).mean()
            avg_loss = loss.ewm(alpha=alpha_rsi, adjust=False).mean()
            rs = avg_gain / (avg_loss + 1e-10)
            df['rsi'] = 100 - (100 / (1 + rs))

        # Volume ratio
        if self.use_volume_filter:
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-10)

        return df

    def generate_signal(
        self,
        df: pd.DataFrame,
        df_htf: Optional[pd.DataFrame] = None,
        position: Optional[Dict] = None
    ) -> TrendSignal:
        """
        Gerar sinal de trend following.

        REGRAS DE ENTRADA:

        LONG:
        1. EMA fast cruza acima EMA slow (ou fast > slow recentemente)
        2. ADX > threshold (confirma trend forte)
        3. Preco acima EMA 200 (trend geral bullish)
        4. [OPCIONAL] HTF trend bullish
        5. [OPCIONAL] Volume acima da media
        6. [OPCIONAL] RSI nao em extremo overbought

        SHORT:
        1. EMA fast cruza abaixo EMA slow (ou fast < slow recentemente)
        2. ADX > threshold (confirma trend forte)
        3. Preco abaixo EMA 200 (trend geral bearish)
        4. [OPCIONAL] HTF trend bearish
        5. [OPCIONAL] Volume acima da media
        6. [OPCIONAL] RSI nao em extremo oversold

        Args:
            df: DataFrame com OHLCV (timeframe 1h)
            df_htf: DataFrame com timeframe superior (4h recomendado)
            position: Posicao atual (para trailing stop)

        Returns:
            TrendSignal com direcao, entry, SL, TP e trailing stop
        """
        if len(df) < max(self.ema_trend, 50):
            return TrendSignal('none', 0, 0, 0, 0, 0, 'Dados insuficientes', 'low')

        # Calcular indicadores se nao existirem
        if 'ema_fast' not in df.columns:
            df = self.calculate_indicators(df)

        # Valores atuais
        price = df['close'].values[-1]
        ema_f = df['ema_fast'].values[-1]
        ema_s = df['ema_slow'].values[-1]
        ema_200 = df['ema_200'].values[-1]
        atr = df['atr'].values[-1]
        adx = df['adx'].values[-1]
        di_plus = df['di_plus'].values[-1]
        di_minus = df['di_minus'].values[-1]

        # Valores anteriores (para detectar cross)
        ema_f_prev = df['ema_fast'].values[-2]
        ema_s_prev = df['ema_slow'].values[-2]
        adx_prev = df['adx'].values[-2]

        # Validar dados
        if np.isnan(atr) or atr == 0 or np.isnan(adx):
            return TrendSignal('none', 0, price, 0, 0, 0, 'Indicadores invalidos', 'low')

        # Verificar se estamos em posicao (para trailing stop)
        if position and position.get('amt', 0) != 0:
            return self._check_trailing_exit(df, position, price, atr)

        # === FILTROS GLOBAIS ===

        # 1. ADX minimo
        if adx < self.adx_min:
            return TrendSignal('none', 0, price, 0, 0, 0, f'ADX fraco ({adx:.1f} < {self.adx_min})', 'low')

        # 2. ADX muito alto (possivel reversao)
        if adx > self.adx_peak:
            return TrendSignal('none', 0, price, 0, 0, 0, f'ADX pico ({adx:.1f} > {self.adx_peak})', 'low')

        # 3. Volume filter
        if self.use_volume_filter:
            volume_ratio = df['volume_ratio'].values[-1]
            if volume_ratio < self.min_volume_ratio:
                return TrendSignal('none', 0, price, 0, 0, 0, f'Volume baixo ({volume_ratio:.2f}x)', 'low')

        # 4. Higher Timeframe Filter
        htf_trend = 0  # 0=neutral, 1=bullish, -1=bearish
        if self.use_htf_filter and df_htf is not None:
            htf_trend = self._calculate_htf_trend(df_htf)

        # === DETECTAR SINAIS ===

        direction = 'none'
        strength = 0
        confidence = 'low'
        reasons = []

        # EMA Cross Detection
        # Cross recente = ultimas 3 velas
        cross_up_recent = False
        cross_down_recent = False

        for i in range(1, min(4, len(df))):
            if df['ema_fast'].values[-i] > df['ema_slow'].values[-i] and \
               df['ema_fast'].values[-i-1] <= df['ema_slow'].values[-i-1]:
                cross_up_recent = True
                break

        for i in range(1, min(4, len(df))):
            if df['ema_fast'].values[-i] < df['ema_slow'].values[-i] and \
               df['ema_fast'].values[-i-1] >= df['ema_slow'].values[-i-1]:
                cross_down_recent = True
                break

        # === LONG SIGNAL ===
        if ema_f > ema_s and (cross_up_recent or ema_f > ema_s * 1.002):  # 0.2% acima
            # Verificar filtros

            # Filtro EMA 200
            if price < ema_200:
                return TrendSignal('none', 0, price, 0, 0, 0, 'Preco abaixo EMA 200 (long)', 'low')

            # Filtro DI+ > DI-
            if di_plus <= di_minus:
                return TrendSignal('none', 0, price, 0, 0, 0, 'DI+ nao dominante', 'low')

            # Filtro RSI extremo
            if self.use_rsi_filter:
                rsi = df['rsi'].values[-1]
                if rsi > self.rsi_extreme_high:
                    return TrendSignal('none', 0, price, 0, 0, 0, f'RSI overbought ({rsi:.1f})', 'low')

            # Filtro HTF
            if self.use_htf_filter and htf_trend == -1:
                return TrendSignal('none', 0, price, 0, 0, 0, 'HTF bearish (long)', 'medium')

            # Sinal LONG valido!
            direction = 'long'

            # Calcular strength (0-10)
            strength = 5.0  # Base

            # Bonus por cross recente
            if cross_up_recent:
                strength += 1.5
                reasons.append('EMA cross up')
            else:
                reasons.append('EMA bullish')

            # Bonus por ADX forte
            if adx > self.adx_strong:
                strength += 2.0
                reasons.append(f'ADX forte ({adx:.0f})')
            else:
                strength += 1.0
                reasons.append(f'ADX ok ({adx:.0f})')

            # Bonus por HTF alinhado
            if htf_trend == 1:
                strength += 1.0
                reasons.append('HTF aligned')
                confidence = 'high'
            elif htf_trend == 0:
                confidence = 'medium'

            # Bonus por DI spread
            di_spread = di_plus - di_minus
            if di_spread > 20:
                strength += 0.5

            strength = min(10, strength)

        # === SHORT SIGNAL ===
        elif ema_f < ema_s and (cross_down_recent or ema_f < ema_s * 0.998):  # 0.2% abaixo
            # Verificar filtros

            # Filtro EMA 200
            if price > ema_200:
                return TrendSignal('none', 0, price, 0, 0, 0, 'Preco acima EMA 200 (short)', 'low')

            # Filtro DI- > DI+
            if di_minus <= di_plus:
                return TrendSignal('none', 0, price, 0, 0, 0, 'DI- nao dominante', 'low')

            # Filtro RSI extremo
            if self.use_rsi_filter:
                rsi = df['rsi'].values[-1]
                if rsi < self.rsi_extreme_low:
                    return TrendSignal('none', 0, price, 0, 0, 0, f'RSI oversold ({rsi:.1f})', 'low')

            # Filtro HTF
            if self.use_htf_filter and htf_trend == 1:
                return TrendSignal('none', 0, price, 0, 0, 0, 'HTF bullish (short)', 'medium')

            # Sinal SHORT valido!
            direction = 'short'

            # Calcular strength
            strength = 5.0

            if cross_down_recent:
                strength += 1.5
                reasons.append('EMA cross down')
            else:
                reasons.append('EMA bearish')

            if adx > self.adx_strong:
                strength += 2.0
                reasons.append(f'ADX forte ({adx:.0f})')
            else:
                strength += 1.0
                reasons.append(f'ADX ok ({adx:.0f})')

            if htf_trend == -1:
                strength += 1.0
                reasons.append('HTF aligned')
                confidence = 'high'
            elif htf_trend == 0:
                confidence = 'medium'

            di_spread = di_minus - di_plus
            if di_spread > 20:
                strength += 0.5

            strength = min(10, strength)

        # Sem sinal
        if direction == 'none':
            return TrendSignal('none', 0, price, 0, 0, 0, 'Sem setup valido', 'low')

        # === CALCULAR SL/TP ===

        if direction == 'long':
            sl = price - (atr * self.sl_atr_mult)
            tp = price + (atr * self.tp_atr_mult)
        else:  # short
            sl = price + (atr * self.sl_atr_mult)
            tp = price - (atr * self.tp_atr_mult)

        # Trailing stop percentual
        trailing_pct = self.trailing_stop_pct

        # Ajustar trailing dinamicamente baseado em ADX
        if adx > self.adx_strong:
            # Trend muito forte = trailing mais largo (deixar correr)
            trailing_pct = min(0.25, self.trailing_stop_pct * 1.25)

        reason = ' | '.join(reasons)

        return TrendSignal(
            direction=direction,
            strength=strength,
            entry_price=price,
            stop_loss=sl,
            take_profit=tp,
            trailing_stop_pct=trailing_pct,
            reason=reason,
            confidence=confidence
        )

    def _calculate_htf_trend(self, df_htf: pd.DataFrame) -> int:
        """
        Calcular trend do higher timeframe.
        Returns: 1 (bullish), -1 (bearish), 0 (neutral)
        """
        if len(df_htf) < 50:
            return 0

        # Calcular EMAs no HTF
        ema_21 = df_htf['close'].ewm(span=21, adjust=False).mean()
        ema_50 = df_htf['close'].ewm(span=50, adjust=False).mean()

        if ema_21.values[-1] > ema_50.values[-1]:
            return 1  # Bullish
        elif ema_21.values[-1] < ema_50.values[-1]:
            return -1  # Bearish
        return 0

    def _check_trailing_exit(
        self,
        df: pd.DataFrame,
        position: Dict,
        current_price: float,
        atr: float
    ) -> TrendSignal:
        """
        Verificar trailing stop para posicao aberta.

        Trailing stop so ativa apos lucro de X ATR (trailing_activation_rr).
        Usa percentual do preco ao inves de ATR.
        """
        entry = position.get('entry', 0)
        amt = position.get('amt', 0)

        if entry == 0 or amt == 0:
            return TrendSignal('none', 0, current_price, 0, 0, 0, 'Sem posicao', 'low')

        side = 'long' if amt > 0 else 'short'

        # Calcular lucro atual
        if side == 'long':
            pnl_per_unit = current_price - entry
        else:
            pnl_per_unit = entry - current_price

        profit_r = pnl_per_unit / atr  # Lucro em multiplos de R (ATR)

        # Trailing stop ainda nao ativado
        if profit_r < self.trailing_activation_rr:
            return TrendSignal('none', 0, current_price, 0, 0, 0,
                             f'Trailing nao ativo ({profit_r:.1f}R < {self.trailing_activation_rr}R)',
                             'low')

        # Calcular preco maximo/minimo desde entrada
        if side == 'long':
            # Pegar maxima desde entrada
            highest = df['high'].max()  # Simplificado - idealmente guardar no position
            trailing_price = highest * (1 - self.trailing_stop_pct)

            if current_price <= trailing_price:
                # Trailing stop atingido!
                return TrendSignal(
                    direction='short',  # Fechar long
                    strength=10,
                    entry_price=current_price,
                    stop_loss=0,
                    take_profit=0,
                    trailing_stop_pct=0,
                    reason=f'TRAILING STOP HIT (preco caiu {self.trailing_stop_pct*100:.0f}% do pico)',
                    confidence='high'
                )
        else:
            # Pegar minima desde entrada
            lowest = df['low'].min()
            trailing_price = lowest * (1 + self.trailing_stop_pct)

            if current_price >= trailing_price:
                # Trailing stop atingido!
                return TrendSignal(
                    direction='long',  # Fechar short
                    strength=10,
                    entry_price=current_price,
                    stop_loss=0,
                    take_profit=0,
                    trailing_stop_pct=0,
                    reason=f'TRAILING STOP HIT (preco subiu {self.trailing_stop_pct*100:.0f}% do fundo)',
                    confidence='high'
                )

        return TrendSignal('none', 0, current_price, 0, 0, 0,
                         f'Em trailing ({profit_r:.1f}R lucro)',
                         'medium')


def backtest_trend_following(
    df: pd.DataFrame,
    df_htf: Optional[pd.DataFrame] = None,
    params: Dict = None,
    initial_capital: float = 10000,
    position_size_pct: float = 0.95
) -> Dict:
    """
    Backtest simplificado da estrategia trend following.

    Args:
        df: DataFrame com OHLCV 1h
        df_htf: DataFrame com HTF (4h recomendado)
        params: Parametros da estrategia
        initial_capital: Capital inicial
        position_size_pct: % do capital por trade

    Returns:
        Dict com metricas de performance
    """
    strategy = TrendFollowingStrategy(params)

    # Preparar dados
    df = strategy.calculate_indicators(df)

    # Estado
    capital = initial_capital
    position = None
    trades = []
    equity_curve = [initial_capital]

    for i in range(len(df)):
        if i < max(strategy.ema_trend, 50):
            equity_curve.append(capital)
            continue

        # Slice ate candle atual
        df_slice = df.iloc[:i+1]
        current_price = df_slice['close'].values[-1]

        # Gerar sinal
        signal = strategy.generate_signal(df_slice, df_htf, position)

        # Verificar saida
        if position is not None:
            # SL/TP
            if position['side'] == 'long':
                if current_price <= position['sl']:
                    # Stop Loss
                    pnl = (current_price - position['entry']) * position['size']
                    capital += pnl
                    trades.append({
                        'entry': position['entry'],
                        'exit': current_price,
                        'side': 'long',
                        'pnl': pnl,
                        'result': 'SL'
                    })
                    position = None
                elif current_price >= position['tp']:
                    # Take Profit
                    pnl = (current_price - position['entry']) * position['size']
                    capital += pnl
                    trades.append({
                        'entry': position['entry'],
                        'exit': current_price,
                        'side': 'long',
                        'pnl': pnl,
                        'result': 'TP'
                    })
                    position = None
            else:  # short
                if current_price >= position['sl']:
                    pnl = (position['entry'] - current_price) * position['size']
                    capital += pnl
                    trades.append({
                        'entry': position['entry'],
                        'exit': current_price,
                        'side': 'short',
                        'pnl': pnl,
                        'result': 'SL'
                    })
                    position = None
                elif current_price <= position['tp']:
                    pnl = (position['entry'] - current_price) * position['size']
                    capital += pnl
                    trades.append({
                        'entry': position['entry'],
                        'exit': current_price,
                        'side': 'short',
                        'pnl': pnl,
                        'result': 'TP'
                    })
                    position = None

            # Sinal de saida (trailing)
            if position is not None and signal.direction != 'none':
                if (position['side'] == 'long' and signal.direction == 'short') or \
                   (position['side'] == 'short' and signal.direction == 'long'):
                    if position['side'] == 'long':
                        pnl = (current_price - position['entry']) * position['size']
                    else:
                        pnl = (position['entry'] - current_price) * position['size']
                    capital += pnl
                    trades.append({
                        'entry': position['entry'],
                        'exit': current_price,
                        'side': position['side'],
                        'pnl': pnl,
                        'result': 'TRAILING'
                    })
                    position = None

        # Abrir nova posicao
        if position is None and signal.direction in ['long', 'short']:
            if signal.strength >= 6:  # Minimo de confianca
                size = (capital * position_size_pct) / current_price
                position = {
                    'side': signal.direction,
                    'entry': signal.entry_price,
                    'sl': signal.stop_loss,
                    'tp': signal.take_profit,
                    'size': size,
                    'amt': size if signal.direction == 'long' else -size
                }

        equity_curve.append(capital)

    # Calcular metricas
    if len(trades) == 0:
        return {
            'total_return_pct': 0,
            'total_trades': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'max_drawdown_pct': 0
        }

    wins = [t['pnl'] for t in trades if t['pnl'] > 0]
    losses = [t['pnl'] for t in trades if t['pnl'] < 0]

    total_return_pct = ((capital - initial_capital) / initial_capital) * 100
    win_rate = len(wins) / len(trades) if trades else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 0
    profit_factor = sum(wins) / abs(sum(losses)) if losses else 0

    # Max drawdown
    peak = initial_capital
    max_dd = 0
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak
        if dd > max_dd:
            max_dd = dd

    return {
        'total_return_pct': total_return_pct,
        'total_trades': len(trades),
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'max_drawdown_pct': max_dd * 100,
        'trades': trades,
        'equity_curve': equity_curve
    }


if __name__ == '__main__':
    """
    EXEMPLO DE USO
    """
    print("=" * 80)
    print("TREND FOLLOWING STRATEGY - Otimizada para Crypto 1h")
    print("=" * 80)
    print()
    print("PARAMETROS OTIMIZADOS (baseados em pesquisa):")
    print()
    print("EMA Cross:")
    print("  - Fast: 12 (balanco entre reatividade e reducao de ruido)")
    print("  - Slow: 26 (filtra ruido, capta trends medios)")
    print("  Ref: https://altfins.com/knowledge-base/ema-12-50-crossovers/")
    print()
    print("ADX Threshold:")
    print("  - Minimo: 30 (ideal para crypto volatil)")
    print("  - Forte: 40 (trend muito forte)")
    print("  - Pico: 50 (cuidado com reversao)")
    print("  Ref: https://www.mindmathmoney.com/articles/adx-indicator-trading-strategy")
    print()
    print("Risk Management:")
    print("  - SL: 2.5 ATR (protecao adequada)")
    print("  - TP: 6.0 ATR (R:R 2.4:1)")
    print("  - Trailing: 20% (maximiza lucros em trends fortes)")
    print("  Ref: https://www.altrady.com/crypto-trading/technical-analysis/risk-management-trend-following-strategies")
    print()
    print("Performance Esperada:")
    print("  - Win Rate: ~35-45% (normal para trend following)")
    print("  - R:R: >2.5:1 (compensa baixo win rate)")
    print("  - Melhor em: Mercados trending (nao lateral)")
    print()
    print("=" * 80)
    print()
    print("Para usar:")
    print()
    print("from core.trend_following_strategy import TrendFollowingStrategy")
    print()
    print("strategy = TrendFollowingStrategy()")
    print("signal = strategy.generate_signal(df, df_htf)")
    print()
    print("if signal.direction != 'none':")
    print("    print(f'{signal.direction.upper()} @ {signal.entry_price}')")
    print("    print(f'SL: {signal.stop_loss} | TP: {signal.take_profit}')")
    print("    print(f'Trailing: {signal.trailing_stop_pct*100:.0f}%')")
    print("    print(f'Reason: {signal.reason}')")
    print()
