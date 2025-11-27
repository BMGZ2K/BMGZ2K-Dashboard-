"""
STOCHASTIC OPTIMIZED STRATEGY - Crypto Futures (1H Timeframe)
==============================================================

Estrategia conservadora baseada em Stochastic Oscillator otimizado para crypto futures.
Target: Win rate > 50% com risk/reward favoravel.

PARAMETROS OTIMIZADOS (baseado em pesquisa de mercado):
- Stochastic K: 13 (otimizado para 1h crypto)
- Stochastic D: 5 (smooth maior para reduzir falsos sinais)
- Smooth K: 3 (padrao TradingView)
- Oversold: 25 (ajustado para alta volatilidade de crypto)
- Overbought: 75 (ajustado para alta volatilidade de crypto)
- EMA Fast: 9 (confirmacao rapida)
- EMA Slow: 21 (tendencia de medio prazo)
- ADX Threshold: 25 (filtro de trend strength)
- SL ATR Multiplier: 2.5 (conservador para crypto)
- TP ATR Multiplier: 4.0 (ratio 1.6:1)

REFERENCIAS:
- TradingView Stochastic Guide
- LuxAlgo Research on Stochastic Settings
- Crypto Trading Best Practices 2024-2025
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class StochasticSignal:
    """Sinal gerado pela estrategia Stochastic."""
    direction: str  # 'long', 'short', 'none'
    strength: float  # 0-10 (quanto maior, mais forte)
    entry_price: float
    stop_loss: float
    take_profit: float
    reason: str
    stoch_k: float
    stoch_d: float
    adx: float
    ema_alignment: bool
    timestamp: datetime


class StochasticOptimizedStrategy:
    """
    Estrategia Stochastic Otimizada para Crypto Futures.

    LOGICA DE ENTRADA:
    ------------------
    LONG:
    1. Stochastic K < 25 (oversold extremo)
    2. Stochastic K cruza acima de D (confirmacao de reversao)
    3. EMA 9 > EMA 21 OU ADX > 25 (confirmacao de trend/forca)
    4. Preco acima de EMA 21 (opcional para forca adicional)

    SHORT:
    1. Stochastic K > 75 (overbought extremo)
    2. Stochastic K cruza abaixo de D (confirmacao de reversao)
    3. EMA 9 < EMA 21 OU ADX > 25 (confirmacao de trend/forca)
    4. Preco abaixo de EMA 21 (opcional para forca adicional)

    CALCULO DE STRENGTH:
    --------------------
    Base strength: 5.0
    + Extremo oversold/overbought: +0 a +2.0
    + EMA cross confirmacao: +2.0
    + ADX forte (>30): +0 a +2.0
    + Volume acima da media: +1.0
    Max strength: 10.0

    STOP LOSS / TAKE PROFIT:
    ------------------------
    SL: Entry +/- (ATR * 2.5)
    TP: Entry +/- (ATR * 4.0)
    Risk/Reward: 1:1.6
    """

    def __init__(self, params: Optional[Dict] = None):
        """
        Inicializar estrategia com parametros.

        Args:
            params: Dicionario de parametros customizados (opcional)
        """
        # Parametros default otimizados
        default_params = {
            # Stochastic params
            'stoch_k_period': 13,
            'stoch_d_period': 5,
            'stoch_smooth_k': 3,
            'stoch_oversold': 25,
            'stoch_overbought': 75,

            # EMA params
            'ema_fast': 9,
            'ema_slow': 21,

            # ADX params
            'adx_period': 14,
            'adx_threshold': 25,
            'adx_strong': 30,

            # ATR params
            'atr_period': 14,
            'sl_atr_mult': 2.5,
            'tp_atr_mult': 4.0,

            # Signal strength params
            'min_signal_strength': 5.0,
            'base_strength': 5.0,

            # Volume filter
            'volume_sma_period': 20,
            'volume_mult_threshold': 1.2,

            # Data validation
            'min_data_points': 100,
        }

        # Merge com parametros customizados
        self.params = {**default_params, **(params or {})}

    def calculate_stochastic(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Calcular Stochastic Oscillator (%K e %D).

        Formula:
        %K = SMA(100 * (Close - Lowest Low) / (Highest High - Lowest Low), smooth_k)
        %D = SMA(%K, d_period)

        Conforme TradingView/Binance padrao.
        """
        k_period = self.params['stoch_k_period']
        d_period = self.params['stoch_d_period']
        smooth_k = self.params['stoch_smooth_k']

        # Calcular raw stochastic
        lowest_low = low.rolling(k_period).min()
        highest_high = high.rolling(k_period).max()

        raw_k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)

        # Smoothed %K (SMA of raw_k)
        k = raw_k.rolling(smooth_k).mean()

        # %D (SMA of %K)
        d = k.rolling(d_period).mean()

        return k, d

    def calculate_ema(self, close: pd.Series, period: int) -> pd.Series:
        """Calcular Exponential Moving Average."""
        return close.ewm(span=period, adjust=False).mean()

    def calculate_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series
    ) -> pd.Series:
        """
        Calcular Average True Range usando Wilder's smoothing.
        """
        atr_period = self.params['atr_period']

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Wilder's smoothing (alpha = 1/period)
        alpha = 1 / atr_period
        return tr.ewm(alpha=alpha, adjust=False).mean()

    def calculate_adx(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series
    ) -> pd.Series:
        """
        Calcular Average Directional Index (ADX).
        Mede a forca da tendencia (0-100).
        """
        adx_period = self.params['adx_period']

        # Calcular +DM e -DM
        up = high.diff()
        down = -low.diff()

        plus_dm = up.where((up > down) & (up > 0), 0)
        minus_dm = down.where((down > up) & (down > 0), 0)

        # ATR
        atr = self.calculate_atr(high, low, close)

        # Wilder's smoothing
        alpha = 1 / adx_period

        plus_dm_smooth = plus_dm.ewm(alpha=alpha, adjust=False).mean()
        minus_dm_smooth = minus_dm.ewm(alpha=alpha, adjust=False).mean()

        # +DI e -DI
        plus_di = 100 * plus_dm_smooth / (atr + 1e-10)
        minus_di = 100 * minus_dm_smooth / (atr + 1e-10)

        # DX
        di_sum = plus_di + minus_di
        di_diff = abs(plus_di - minus_di)
        dx = 100 * di_diff / (di_sum + 1e-10)

        # ADX (Wilder's smoothed DX)
        adx = dx.ewm(alpha=alpha, adjust=False).mean()

        return adx

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preparar dados com todos os indicadores.

        Args:
            df: DataFrame com colunas ['open', 'high', 'low', 'close', 'volume']

        Returns:
            DataFrame com indicadores adicionados
        """
        df = df.copy()

        # Validar dados minimos
        if len(df) < self.params['min_data_points']:
            raise ValueError(
                f"Dados insuficientes. Necessario minimo {self.params['min_data_points']} candles, "
                f"recebido {len(df)}"
            )

        # Calcular Stochastic
        df['stoch_k'], df['stoch_d'] = self.calculate_stochastic(
            df['high'], df['low'], df['close']
        )

        # Calcular EMAs
        df['ema_fast'] = self.calculate_ema(df['close'], self.params['ema_fast'])
        df['ema_slow'] = self.calculate_ema(df['close'], self.params['ema_slow'])

        # Calcular ATR
        df['atr'] = self.calculate_atr(df['high'], df['low'], df['close'])

        # Calcular ADX
        df['adx'] = self.calculate_adx(df['high'], df['low'], df['close'])

        # Calcular Volume SMA
        df['volume_sma'] = df['volume'].rolling(
            self.params['volume_sma_period']
        ).mean()

        return df

    def generate_signal(
        self,
        df: pd.DataFrame,
        precomputed: bool = False
    ) -> StochasticSignal:
        """
        Gerar sinal de trading.

        Args:
            df: DataFrame com dados OHLCV
            precomputed: Se True, assume que indicadores ja foram calculados

        Returns:
            StochasticSignal com direcao e detalhes
        """
        # Preparar dados se necessario
        if not precomputed or 'stoch_k' not in df.columns:
            df = self.prepare_data(df)

        # Extrair valores atuais e anteriores
        current = df.iloc[-1]
        previous = df.iloc[-2]

        price = current['close']
        stoch_k = current['stoch_k']
        stoch_d = current['stoch_d']
        stoch_k_prev = previous['stoch_k']
        stoch_d_prev = previous['stoch_d']
        ema_fast = current['ema_fast']
        ema_slow = current['ema_slow']
        ema_fast_prev = previous['ema_fast']
        ema_slow_prev = previous['ema_slow']
        atr = current['atr']
        adx = current['adx']
        volume = current['volume']
        volume_sma = current['volume_sma']

        # Validar indicadores
        if np.isnan(stoch_k) or np.isnan(atr) or atr == 0:
            return StochasticSignal(
                direction='none',
                strength=0,
                entry_price=price,
                stop_loss=0,
                take_profit=0,
                reason='Indicadores invalidos',
                stoch_k=stoch_k if not np.isnan(stoch_k) else 0,
                stoch_d=stoch_d if not np.isnan(stoch_d) else 0,
                adx=adx if not np.isnan(adx) else 0,
                ema_alignment=False,
                timestamp=datetime.now()
            )

        # Detectar crossovers
        stoch_cross_up = stoch_k > stoch_d and stoch_k_prev <= stoch_d_prev
        stoch_cross_down = stoch_k < stoch_d and stoch_k_prev >= stoch_d_prev

        ema_cross_up = ema_fast > ema_slow and ema_fast_prev <= ema_slow_prev
        ema_cross_down = ema_fast < ema_slow and ema_fast_prev >= ema_slow_prev

        # Estado de EMA
        ema_bullish = ema_fast > ema_slow
        ema_bearish = ema_fast < ema_slow

        # Filtros de trend
        trend_strong = adx > self.params['adx_threshold']
        trend_very_strong = adx > self.params['adx_strong']

        # Volume filter
        volume_high = volume > volume_sma * self.params['volume_mult_threshold']

        # Inicializar sinal
        direction = 'none'
        strength = 0.0
        reasons = []

        # === LOGICA DE ENTRADA LONG ===
        if stoch_k < self.params['stoch_oversold'] and stoch_cross_up:
            # Confirmacao de trend
            trend_confirmed = (ema_bullish or trend_strong)

            if trend_confirmed:
                direction = 'long'
                strength = self.params['base_strength']
                reasons.append(f'Stoch oversold cross up (K={stoch_k:.1f})')

                # Bonus: Quao extremo esta o oversold
                oversold_extreme = max(0, self.params['stoch_oversold'] - stoch_k)
                oversold_bonus = min(2.0, oversold_extreme / 10)
                strength += oversold_bonus

                # Bonus: EMA cross confirmacao
                if ema_cross_up:
                    strength += 2.0
                    reasons.append('EMA bullish cross')
                elif ema_bullish:
                    strength += 1.0
                    reasons.append('EMA bullish aligned')

                # Bonus: ADX forte
                if trend_very_strong:
                    adx_bonus = min(2.0, (adx - self.params['adx_strong']) / 15)
                    strength += adx_bonus
                    reasons.append(f'Strong trend (ADX={adx:.1f})')
                elif trend_strong:
                    strength += 0.5

                # Bonus: Volume alto
                if volume_high:
                    strength += 1.0
                    reasons.append('High volume')

                # Bonus: Preco acima de EMA slow (confirmacao adicional)
                if price > ema_slow:
                    strength += 0.5
                    reasons.append('Price > EMA21')

                # Cap no maximo
                strength = min(10.0, strength)

        # === LOGICA DE ENTRADA SHORT ===
        elif stoch_k > self.params['stoch_overbought'] and stoch_cross_down:
            # Confirmacao de trend
            trend_confirmed = (ema_bearish or trend_strong)

            if trend_confirmed:
                direction = 'short'
                strength = self.params['base_strength']
                reasons.append(f'Stoch overbought cross down (K={stoch_k:.1f})')

                # Bonus: Quao extremo esta o overbought
                overbought_extreme = max(0, stoch_k - self.params['stoch_overbought'])
                overbought_bonus = min(2.0, overbought_extreme / 10)
                strength += overbought_bonus

                # Bonus: EMA cross confirmacao
                if ema_cross_down:
                    strength += 2.0
                    reasons.append('EMA bearish cross')
                elif ema_bearish:
                    strength += 1.0
                    reasons.append('EMA bearish aligned')

                # Bonus: ADX forte
                if trend_very_strong:
                    adx_bonus = min(2.0, (adx - self.params['adx_strong']) / 15)
                    strength += adx_bonus
                    reasons.append(f'Strong trend (ADX={adx:.1f})')
                elif trend_strong:
                    strength += 0.5

                # Bonus: Volume alto
                if volume_high:
                    strength += 1.0
                    reasons.append('High volume')

                # Bonus: Preco abaixo de EMA slow
                if price < ema_slow:
                    strength += 0.5
                    reasons.append('Price < EMA21')

                # Cap no maximo
                strength = min(10.0, strength)

        # Filtrar sinais fracos
        if strength < self.params['min_signal_strength']:
            direction = 'none'
            strength = 0
            reasons = ['Signal too weak']

        # Calcular SL/TP
        if direction == 'long':
            stop_loss = price - (atr * self.params['sl_atr_mult'])
            take_profit = price + (atr * self.params['tp_atr_mult'])
        elif direction == 'short':
            stop_loss = price + (atr * self.params['sl_atr_mult'])
            take_profit = price - (atr * self.params['tp_atr_mult'])
        else:
            stop_loss = 0
            take_profit = 0

        return StochasticSignal(
            direction=direction,
            strength=strength,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=' | '.join(reasons) if reasons else 'No signal',
            stoch_k=stoch_k,
            stoch_d=stoch_d,
            adx=adx,
            ema_alignment=(ema_bullish if direction == 'long' else ema_bearish),
            timestamp=datetime.now()
        )

    def check_exit_signal(
        self,
        position: Dict,
        current_price: float,
        current_high: float,
        current_low: float,
        stoch_k: float,
        stoch_d: float,
        atr: float,
        adx: float
    ) -> Optional[str]:
        """
        Verificar sinais de saida para posicoes abertas.

        Args:
            position: Dict com 'side', 'entry', 'sl', 'tp', 'amt'
            current_price: Preco atual
            current_high: High da vela atual
            current_low: Low da vela atual
            stoch_k: Valor atual de %K
            stoch_d: Valor atual de %D
            atr: ATR atual
            adx: ADX atual

        Returns:
            Razao da saida ou None se deve manter posicao
        """
        side = position['side']
        entry = position['entry']
        sl = position.get('sl', 0)
        tp = position.get('tp', 0)

        # Calcular PnL
        if side == 'long':
            pnl_pct = (current_price - entry) / entry if entry > 0 else 0
        else:
            pnl_pct = (entry - current_price) / entry if entry > 0 else 0

        # === CHECK STOP LOSS ===
        if sl > 0:
            if side == 'long' and current_low <= sl:
                return 'STOP_LOSS'
            elif side == 'short' and current_high >= sl:
                return 'STOP_LOSS'

        # === CHECK TAKE PROFIT ===
        if tp > 0:
            if side == 'long' and current_high >= tp:
                return 'TAKE_PROFIT'
            elif side == 'short' and current_low <= tp:
                return 'TAKE_PROFIT'

        # === REVERSAL SIGNALS ===
        # Long position: Sair se stochastic ficar overbought
        if side == 'long':
            if stoch_k > self.params['stoch_overbought'] and stoch_k < stoch_d:
                # Cross down em zona overbought
                if pnl_pct > 0.01:  # Apenas se estiver lucrativo (>1%)
                    return 'STOCH_REVERSAL'

        # Short position: Sair se stochastic ficar oversold
        elif side == 'short':
            if stoch_k < self.params['stoch_oversold'] and stoch_k > stoch_d:
                # Cross up em zona oversold
                if pnl_pct > 0.01:  # Apenas se estiver lucrativo (>1%)
                    return 'STOCH_REVERSAL'

        # === TRAILING STOP (opcional, baseado em ADX) ===
        # Se tendencia muito forte, dar espaco para rodar
        if adx > 40 and pnl_pct > 0.03:  # >3% de lucro em trend forte
            # Ajustar SL dinamicamente para breakeven + 1%
            new_sl_distance = atr * 1.0  # Trailing stop mais apertado
            if side == 'long':
                trailing_sl = current_price - new_sl_distance
                if current_low <= trailing_sl:
                    return 'TRAILING_STOP'
            else:
                trailing_sl = current_price + new_sl_distance
                if current_high >= trailing_sl:
                    return 'TRAILING_STOP'

        return None

    def get_params(self) -> Dict:
        """Retornar parametros atuais da estrategia."""
        return self.params.copy()

    def update_params(self, new_params: Dict) -> None:
        """Atualizar parametros da estrategia."""
        self.params.update(new_params)


# ============================================================================
# FUNCOES DE BACKTEST
# ============================================================================

def backtest_strategy(
    df: pd.DataFrame,
    strategy: StochasticOptimizedStrategy,
    initial_capital: float = 10000.0,
    position_size_pct: float = 0.95,
    fee_rate: float = 0.0004
) -> Dict:
    """
    Backtester simples para a estrategia Stochastic.

    Args:
        df: DataFrame com dados OHLCV
        strategy: Instancia da estrategia
        initial_capital: Capital inicial
        position_size_pct: % do capital a usar por trade (0-1)
        fee_rate: Taxa de transacao (0.04% = 0.0004)

    Returns:
        Dict com resultados do backtest
    """
    # Preparar dados
    df = strategy.prepare_data(df)

    # Estado
    capital = initial_capital
    position = None
    trades = []
    equity_curve = [initial_capital]

    for i in range(len(df)):
        if i < 2:  # Precisa de pelo menos 2 candles
            equity_curve.append(capital)
            continue

        current_df = df.iloc[:i+1]
        current = current_df.iloc[-1]

        # Se em posicao, verificar saida
        if position:
            exit_reason = strategy.check_exit_signal(
                position=position,
                current_price=current['close'],
                current_high=current['high'],
                current_low=current['low'],
                stoch_k=current['stoch_k'],
                stoch_d=current['stoch_d'],
                atr=current['atr'],
                adx=current['adx']
            )

            if exit_reason:
                # Fechar posicao
                exit_price = current['close']

                if position['side'] == 'long':
                    pnl = (exit_price - position['entry']) * position['size']
                else:
                    pnl = (position['entry'] - exit_price) * position['size']

                # Descontar fees
                fee = exit_price * position['size'] * fee_rate
                pnl -= fee
                pnl -= position['entry_fee']

                capital += pnl

                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': current.name,
                    'side': position['side'],
                    'entry_price': position['entry'],
                    'exit_price': exit_price,
                    'size': position['size'],
                    'pnl': pnl,
                    'pnl_pct': (pnl / (position['entry'] * position['size'])) * 100,
                    'exit_reason': exit_reason
                })

                position = None

        # Se sem posicao, verificar entrada
        else:
            signal = strategy.generate_signal(current_df, precomputed=True)

            if signal.direction in ['long', 'short'] and signal.strength >= strategy.params['min_signal_strength']:
                # Abrir posicao
                entry_price = signal.entry_price
                position_value = capital * position_size_pct
                size = position_value / entry_price
                entry_fee = entry_price * size * fee_rate

                position = {
                    'side': signal.direction,
                    'entry': entry_price,
                    'entry_time': current.name,
                    'size': size,
                    'sl': signal.stop_loss,
                    'tp': signal.take_profit,
                    'entry_fee': entry_fee,
                    'strength': signal.strength
                }

        # Atualizar equity curve
        if position:
            # Mark-to-market
            if position['side'] == 'long':
                unrealized_pnl = (current['close'] - position['entry']) * position['size']
            else:
                unrealized_pnl = (position['entry'] - current['close']) * position['size']
            equity = capital + unrealized_pnl - position['entry_fee']
        else:
            equity = capital

        equity_curve.append(equity)

    # Calcular metricas
    if len(trades) == 0:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'total_return_pct': 0,
            'final_capital': initial_capital,
            'max_drawdown_pct': 0,
            'profit_factor': 0,
            'trades': [],
            'equity_curve': equity_curve
        }

    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]

    win_rate = len(wins) / len(trades) * 100 if trades else 0
    total_return_pct = (capital - initial_capital) / initial_capital * 100

    total_profit = sum(t['pnl'] for t in wins) if wins else 0
    total_loss = abs(sum(t['pnl'] for t in losses)) if losses else 0
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

    # Max drawdown
    peak = initial_capital
    max_dd = 0
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd:
            max_dd = dd

    avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
    avg_loss = np.mean([t['pnl'] for t in losses]) if losses else 0

    return {
        'total_trades': len(trades),
        'winning_trades': len(wins),
        'losing_trades': len(losses),
        'win_rate': win_rate,
        'total_return_pct': total_return_pct,
        'final_capital': capital,
        'max_drawdown_pct': max_dd,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'trades': trades,
        'equity_curve': equity_curve
    }


def print_backtest_results(results: Dict) -> None:
    """Imprimir resultados do backtest de forma formatada."""
    print("\n" + "="*60)
    print("STOCHASTIC OPTIMIZED STRATEGY - BACKTEST RESULTS")
    print("="*60)
    print(f"\nTotal Trades: {results['total_trades']}")
    print(f"Winning Trades: {results['winning_trades']}")
    print(f"Losing Trades: {results['losing_trades']}")
    print(f"Win Rate: {results['win_rate']:.2f}%")
    print(f"\nTotal Return: {results['total_return_pct']:.2f}%")
    print(f"Final Capital: ${results['final_capital']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown_pct']:.2f}%")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    print(f"\nAvg Win: ${results['avg_win']:.2f}")
    print(f"Avg Loss: ${results['avg_loss']:.2f}")
    print("="*60 + "\n")


# ============================================================================
# EXEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    """
    Exemplo de uso da estrategia com dados de exemplo.
    Para uso real, substitua com dados da Binance API.
    """

    # Criar dados de exemplo (substituir com dados reais)
    print("Gerando dados de exemplo...")
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='1h')
    np.random.seed(42)

    # Simular precos com trend e volatilidade
    price = 50000
    prices = []
    for _ in range(len(dates)):
        change = np.random.normal(0, 500)
        price += change
        prices.append(price)

    df = pd.DataFrame({
        'open': prices,
        'high': [p + np.random.uniform(100, 500) for p in prices],
        'low': [p - np.random.uniform(100, 500) for p in prices],
        'close': [p + np.random.normal(0, 200) for p in prices],
        'volume': [np.random.uniform(1000, 5000) for _ in prices]
    }, index=dates)

    # Criar estrategia
    print("Inicializando estrategia Stochastic Otimizada...")
    strategy = StochasticOptimizedStrategy()

    # Mostrar parametros
    print("\nParametros da Estrategia:")
    for key, value in strategy.get_params().items():
        print(f"  {key}: {value}")

    # Rodar backtest
    print("\nExecutando backtest...")
    results = backtest_strategy(df, strategy, initial_capital=10000.0)

    # Mostrar resultados
    print_backtest_results(results)

    # Gerar sinal no ultimo candle
    print("Gerando sinal no ultimo candle...")
    signal = strategy.generate_signal(df)
    print(f"\nDirecao: {signal.direction}")
    print(f"Strength: {signal.strength:.2f}/10")
    print(f"Entry: ${signal.entry_price:.2f}")
    print(f"Stop Loss: ${signal.stop_loss:.2f}")
    print(f"Take Profit: ${signal.take_profit:.2f}")
    print(f"Razao: {signal.reason}")
    print(f"Stoch K: {signal.stoch_k:.2f}")
    print(f"Stoch D: {signal.stoch_d:.2f}")
    print(f"ADX: {signal.adx:.2f}")
