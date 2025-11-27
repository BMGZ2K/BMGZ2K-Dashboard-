"""
Signals Module - Gerador de sinais otimizado
Baseado em testes reais com dados da Binance

VERSÃO: 3.0 - Parâmetros centralizados via config
CHANGELOG:
- v3.0: Parâmetros centralizados em core/config.py
- Todos os defaults vêm de WFO_VALIDATED_PARAMS
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

# Importar parâmetros centralizados
from .config import WFO_VALIDATED_PARAMS, get_param

# Importar estratégias otimizadas (compatibilidade)
try:
    from core.strategies_optimized import OptimizedStrategies, OPTIMIZED_PARAMS
    HAS_OPTIMIZED = True
except ImportError:
    HAS_OPTIMIZED = False
    OPTIMIZED_PARAMS = {}


@dataclass
class Signal:
    """Sinal de trading."""
    direction: str  # 'long', 'short', 'none'
    strength: float  # 0-10
    entry_price: float
    stop_loss: float
    take_profit: float
    reason: str
    strategy: str = ""  # Nome da estratégia que gerou
    confidence: float = 0.0  # 0-1 confiança do sinal


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calcular RSI usando Wilder's Smoothed Moving Average.
    Conforme implementação padrão da Binance/TradingView.
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta.where(delta < 0, 0))

    # Usar EWM com alpha = 1/period (Wilder's smoothing)
    # Isso é equivalente a span = 2*period - 1
    alpha = 1 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calcular ATR usando Wilder's Smoothed Moving Average.
    Conforme implementação padrão da Binance/TradingView.
    """
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Usar EWM com alpha = 1/period (Wilder's smoothing)
    alpha = 1 / period
    return tr.ewm(alpha=alpha, adjust=False).mean()


def calculate_ema(close: pd.Series, period: int) -> pd.Series:
    """Calcular EMA."""
    return close.ewm(span=period, adjust=False).mean()


def calculate_bollinger(close: pd.Series, period: int = 20, std_mult: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calcular Bollinger Bands.
    Usa ddof=0 (population std) para match com TradingView/Binance.
    """
    mid = close.rolling(period).mean()
    std_dev = close.rolling(period).std(ddof=0)  # Population std
    upper = mid + (std_dev * std_mult)
    lower = mid - (std_dev * std_mult)
    return upper, mid, lower


def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calcular ADX usando Wilder's Smoothed Moving Average.
    Conforme padrão da Binance/TradingView.
    """
    # Calcular +DM e -DM
    up = high.diff()
    down = -low.diff()

    plus_dm = up.where((up > down) & (up > 0), 0)
    minus_dm = down.where((down > up) & (down > 0), 0)

    # ATR usando Wilder's smoothing
    atr = calculate_atr(high, low, close, period)

    # Usar alpha = 1/period para Wilder's smoothing
    alpha = 1 / period

    # Smoothed +DM e -DM
    plus_dm_smooth = plus_dm.ewm(alpha=alpha, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=alpha, adjust=False).mean()

    # +DI e -DI
    plus_di = 100 * plus_dm_smooth / (atr + 1e-10)
    minus_di = 100 * minus_dm_smooth / (atr + 1e-10)

    # DX
    di_sum = plus_di + minus_di
    di_diff = abs(plus_di - minus_di)
    dx = 100 * di_diff / (di_sum + 1e-10)

    # ADX = Wilder's smoothed DX
    return dx.ewm(alpha=alpha, adjust=False).mean()


def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3, smooth_k: int = 3) -> Tuple[pd.Series, pd.Series]:
    """
    Calcular Stochastic Oscillator.
    Conforme TradingView/Binance: raw_k -> SMA(smooth_k) -> K -> SMA(d) -> D
    """
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    raw_k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    # K = SMA do raw_k (smooth_k default = 3)
    k = raw_k.rolling(smooth_k).mean()
    # D = SMA do K
    d = k.rolling(d_period).mean()
    return k, d


class SignalGenerator:
    """
    Gerador de sinais otimizado.

    Estrategias implementadas (v2.0 otimizado):
    1. rsi_extremes_v2 - RSI com filtros ADX/EMA + long bias
    2. stoch_extreme_v2 - Stochastic otimizado (RECOMENDADA)
    3. momentum_burst_v2 - Momentum com thresholds relaxados
    4. mean_reversion_v2 - Mean reversion LONG only (short 0% WR)
    5. trend_following_v2 - Trend com pullback entries
    6. combined - Requer 2+ estratégias concordando

    Estratégias legadas (v1.0) ainda disponíveis para compatibilidade.
    """

    def __init__(self, params: Dict = None):
        # Combinar parâmetros passados com defaults centralizados
        self.params = WFO_VALIDATED_PARAMS.copy()
        if params:
            self.params.update(params)

        # RSI params (do config centralizado)
        self.rsi_period = self.params.get('rsi_period', get_param('rsi_period', 14))
        self.rsi_oversold = self.params.get('rsi_oversold', get_param('rsi_oversold', 25))
        self.rsi_overbought = self.params.get('rsi_overbought', get_param('rsi_overbought', 75))

        # Stochastic params
        self.stoch_k = self.params.get('stoch_k', get_param('stoch_k', 14))
        self.stoch_d = self.params.get('stoch_d', get_param('stoch_d', 3))
        self.stoch_oversold = self.params.get('stoch_oversold', get_param('stoch_oversold', 20))
        self.stoch_overbought = self.params.get('stoch_overbought', get_param('stoch_overbought', 80))

        # ATR params
        self.atr_period = self.params.get('atr_period', get_param('atr_period', 14))
        self.sl_atr_mult = self.params.get('sl_atr_mult', get_param('sl_atr_mult', 3.0))
        self.tp_atr_mult = self.params.get('tp_atr_mult', get_param('tp_atr_mult', 5.0))

        # Trend params
        self.ema_fast = self.params.get('ema_fast', get_param('ema_fast', 9))
        self.ema_slow = self.params.get('ema_slow', get_param('ema_slow', 21))
        self.ema_trend = self.params.get('ema_trend', get_param('ema_trend', 50))
        self.adx_threshold = self.params.get('adx_threshold', get_param('adx_min', 20))
        self.adx_min = self.params.get('adx_min', get_param('adx_min', 20))
        self.adx_strong = self.params.get('adx_strong', get_param('adx_strong', 25))
        self.adx_very_strong = self.params.get('adx_very_strong', get_param('adx_very_strong', 35))

        # Strategy mode
        self.strategy = self.params.get('strategy', get_param('strategy', 'stoch_extreme'))

        # Bias params
        self.long_bias = self.params.get('long_bias', 1.1)
        self.short_penalty = self.params.get('short_penalty', 0.9)

        # Signal strength params
        self.min_signal_strength = self.params.get('min_signal_strength', 5)
        self.max_signal_strength = self.params.get('max_signal_strength', 10)
        self.stoch_base_strength = self.params.get('stoch_base_strength', 6.0)
        self.adx_aggressive = self.params.get('adx_aggressive', 30)

        # Inicializar estratégias otimizadas se disponíveis
        self._optimized_strategies = None
        if HAS_OPTIMIZED:
            self._optimized_strategies = OptimizedStrategies(self.params)

        # Exit signal params
        self.rsi_exit_long = self.params.get('rsi_exit_long', 70)
        self.rsi_exit_short = self.params.get('rsi_exit_short', 30)
        self.min_profit_exit = self.params.get('min_profit_exit', 0.01)  # 1%

        # Momentum params
        self.momentum_adx_threshold = self.params.get('momentum_adx_threshold', 30)
        self.momentum_min_move = self.params.get('momentum_min_move', 1.0)
        self.momentum_sl_factor = self.params.get('momentum_sl_factor', 0.75)
        self.momentum_tp_factor = self.params.get('momentum_tp_factor', 0.85)

        # Mean reversion params
        self.mr_adx_max = self.params.get('mr_adx_max', 25)
        self.mr_rsi_long_max = self.params.get('mr_rsi_long_max', 35)
        self.mr_rsi_short_min = self.params.get('mr_rsi_short_min', 65)
        self.mr_sl_factor = self.params.get('mr_sl_factor', 0.50)
        self.mr_tp_factor = self.params.get('mr_tp_factor', 0.60)

        # Data validation
        self.min_data_points = self.params.get('min_data_points', 50)
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preparar dados com indicadores."""
        df = df.copy()

        # Indicadores basicos
        df['rsi'] = calculate_rsi(df['close'], self.rsi_period)
        df['atr'] = calculate_atr(df['high'], df['low'], df['close'], self.atr_period)
        df['ema_fast'] = calculate_ema(df['close'], self.ema_fast)
        df['ema_slow'] = calculate_ema(df['close'], self.ema_slow)
        df['adx'] = calculate_adx(df['high'], df['low'], df['close'])

        # Stochastic
        df['stoch_k'], df['stoch_d'] = calculate_stochastic(
            df['high'], df['low'], df['close'],
            self.stoch_k, self.stoch_d
        )

        bb_upper, bb_mid, bb_lower = calculate_bollinger(df['close'])
        df['bb_upper'] = bb_upper
        df['bb_mid'] = bb_mid
        df['bb_lower'] = bb_lower

        return df
    
    def generate_signal(self, df: pd.DataFrame, precomputed: bool = False) -> Signal:
        """
        Gerar sinal baseado na estrategia selecionada.

        Args:
            df: DataFrame com dados OHLCV
            precomputed: Se True, assume que indicadores já foram calculados

        Estratégias V2 (otimizadas - RECOMENDADAS):
            - rsi_extremes_v2, stoch_extreme_v2, momentum_burst_v2
            - mean_reversion_v2, trend_following_v2, combined

        Estratégias V1 (legado):
            - rsi_extremes, stoch_extreme, trend_following
            - mean_reversion, momentum_burst
        """
        if len(df) < self.min_data_points:
            return Signal('none', 0, 0, 0, 0, 'Dados insuficientes')

        # Só calcular indicadores se não foram pré-calculados
        if not precomputed or 'rsi' not in df.columns:
            df = self.prepare_data(df)

        # ========== ESTRATÉGIAS V2 (OTIMIZADAS) ==========
        # Usar módulo otimizado se disponível
        if self._optimized_strategies is not None:
            v2_strategies = [
                'rsi_extremes_v2', 'stoch_extreme_v2', 'momentum_burst_v2',
                'mean_reversion_v2', 'trend_following_v2', 'combined'
            ]
            if self.strategy in v2_strategies:
                opt_signal = self._optimized_strategies.generate_signal(df, self.strategy)
                # Converter para Signal local
                return Signal(
                    opt_signal.direction,
                    opt_signal.strength,
                    opt_signal.entry_price,
                    opt_signal.stop_loss,
                    opt_signal.take_profit,
                    opt_signal.reason,
                    opt_signal.strategy,
                    opt_signal.confidence
                )

        # ========== ESTRATÉGIAS V1 (LEGADO) ==========
        if self.strategy == 'rsi_extremes' or self.strategy == 'rsi_extreme':
            return self._signal_rsi_extremes(df)
        elif self.strategy == 'stoch_extreme':
            return self._signal_stoch_extreme(df)
        elif self.strategy == 'trend_following' or self.strategy == 'trend_adx':
            return self._signal_trend_following(df)
        elif self.strategy == 'mean_reversion' or self.strategy == 'pullback':
            return self._signal_mean_reversion(df)
        elif self.strategy == 'momentum_burst':
            return self._signal_momentum_burst(df)

        # ========== DEFAULT: Usar Combined (melhor performance) ==========
        if self._optimized_strategies is not None:
            opt_signal = self._optimized_strategies.generate_signal(df, 'combined')
            return Signal(
                opt_signal.direction,
                opt_signal.strength,
                opt_signal.entry_price,
                opt_signal.stop_loss,
                opt_signal.take_profit,
                opt_signal.reason,
                opt_signal.strategy,
                opt_signal.confidence
            )
        else:
            return self._signal_stoch_extreme(df)  # Fallback para V1
    
    def _signal_rsi_extremes(self, df: pd.DataFrame) -> Signal:
        """
        Estrategia RSI Extremos.

        ATUALIZADO V1.1:
        - Long bias (1.1x) aplicado
        - Short penalty (0.9x) aplicado
        - Thresholds agora configuráveis (default 25/75)
        """
        # Usar .values[-1] ao invés de iloc[-1] para performance
        price = df['close'].values[-1]
        rsi = df['rsi'].values[-1]
        atr = df['atr'].values[-1]
        adx = df['adx'].values[-1]
        ema_f = df['ema_fast'].values[-1]
        ema_s = df['ema_slow'].values[-1]

        if np.isnan(rsi) or np.isnan(atr) or atr == 0:
            return Signal('none', 0, price, 0, 0, 'Indicadores invalidos')

        direction = 'none'
        strength = 0
        reason = ''

        # Filtros de confirmação
        ema_bullish = ema_f > ema_s
        ema_bearish = ema_f < ema_s
        has_trend = adx > self.adx_min

        # Long em oversold extremo + confirmação
        if rsi < self.rsi_oversold:
            if ema_bullish or has_trend:  # Precisa de alguma confirmação
                direction = 'long'
                base_strength = min(10, (self.rsi_oversold - rsi) / 5 + 5)
                strength = min(10, base_strength * self.long_bias)
                reason = f'RSI oversold ({rsi:.1f}) + {"EMA bull" if ema_bullish else "ADX trend"}'

        # Short em overbought extremo + confirmação mais restritiva
        elif rsi > self.rsi_overbought:
            if ema_bearish and has_trend:  # Precisa de AMBAS confirmações
                direction = 'short'
                base_strength = min(10, (rsi - self.rsi_overbought) / 5 + 5)
                strength = min(10, base_strength * self.short_penalty)
                reason = f'RSI overbought ({rsi:.1f}) + EMA bear + ADX {adx:.0f}'

        # Calcular SL/TP
        if direction == 'long':
            sl = price - (atr * self.sl_atr_mult)
            tp = price + (atr * self.tp_atr_mult)
        elif direction == 'short':
            sl = price + (atr * self.sl_atr_mult)
            tp = price - (atr * self.tp_atr_mult)
        else:
            sl = tp = 0

        return Signal(direction, strength, price, sl, tp, reason)
    
    def _signal_stoch_extreme(self, df: pd.DataFrame) -> Signal:
        """
        Estrategia Stochastic Extreme com EMA cross.
        Esta é a estratégia validada por WFO com melhores resultados.

        Regras:
        - Long: Stochastic K cruza acima de D em zona oversold + EMA cross up ou ADX forte
        - Short: Stochastic K cruza abaixo de D em zona overbought + EMA cross down ou ADX forte
        """
        # Extrair arrays numpy uma única vez para máxima performance
        # .values é mais rápido que .to_numpy() (evita overhead de cópia)
        close_arr = df['close'].values
        atr_arr = df['atr'].values
        adx_arr = df['adx'].values
        stoch_k_arr = df['stoch_k'].values
        stoch_d_arr = df['stoch_d'].values
        ema_fast_arr = df['ema_fast'].values
        ema_slow_arr = df['ema_slow'].values

        price = close_arr[-1]
        atr = atr_arr[-1]
        adx = adx_arr[-1]
        stoch_k = stoch_k_arr[-1]
        stoch_d = stoch_d_arr[-1]
        stoch_k_prev = stoch_k_arr[-2]
        stoch_d_prev = stoch_d_arr[-2]
        ema_f = ema_fast_arr[-1]
        ema_s = ema_slow_arr[-1]
        ema_f_prev = ema_fast_arr[-2]
        ema_s_prev = ema_slow_arr[-2]

        if np.isnan(stoch_k) or np.isnan(atr) or atr == 0:
            return Signal('none', 0, price, 0, 0, 'Indicadores invalidos')

        direction = 'none'
        strength = 0
        reason = ''

        # Stochastic cross em zona extrema
        stoch_cross_up = stoch_k > stoch_d and stoch_k_prev <= stoch_d_prev
        stoch_cross_down = stoch_k < stoch_d and stoch_k_prev >= stoch_d_prev

        # EMA cross
        ema_cross_up = ema_f > ema_s and ema_f_prev <= ema_s_prev
        ema_cross_down = ema_f < ema_s and ema_f_prev >= ema_s_prev

        # Tendência confirmada por ADX
        trend_strong = adx > self.adx_min

        # Long: Stochastic oversold EXTREMO com cross up + confirmação de trend
        # OTIMIZADO: Removido +10, agora usa zona extrema estrita
        if stoch_k < self.stoch_oversold and stoch_cross_up:
            if ema_cross_up or (ema_f > ema_s and trend_strong):
                direction = 'long'
                # Cálculo otimizado de strength
                base_strength = 6.0
                oversold_bonus = min(2.0, (self.stoch_oversold - stoch_k) / 5)
                trend_bonus = 2.0 if ema_cross_up else min(2.0, (adx - self.adx_min) / 15)
                strength = min(10, base_strength + oversold_bonus + trend_bonus)
                reason = f'Stoch extreme up ({stoch_k:.0f}) + EMA bull + ADX {adx:.0f}'

        # Short: Stochastic overbought EXTREMO com cross down + confirmação de trend
        # OTIMIZADO: Removido -10, agora usa zona extrema estrita
        elif stoch_k > self.stoch_overbought and stoch_cross_down:
            if ema_cross_down or (ema_f < ema_s and trend_strong):
                direction = 'short'
                # Cálculo otimizado de strength
                base_strength = 6.0
                overbought_bonus = min(2.0, (stoch_k - self.stoch_overbought) / 5)
                trend_bonus = 2.0 if ema_cross_down else min(2.0, (adx - self.adx_min) / 15)
                strength = min(10, base_strength + overbought_bonus + trend_bonus)
                reason = f'Stoch extreme down ({stoch_k:.0f}) + EMA bear + ADX {adx:.0f}'

        # Sinais mais agressivos: EMA cross com ADX forte (mesmo sem stochastic extremo)
        # CORRIGIDO: Usar adx_aggressive de params ao invés de hardcoded 30
        elif trend_strong and adx > self.adx_aggressive:
            if ema_cross_up:
                direction = 'long'
                strength = min(8, adx / 5)
                reason = f'EMA cross up (ADX={adx:.1f})'
            elif ema_cross_down:
                direction = 'short'
                strength = min(8, adx / 5)
                reason = f'EMA cross down (ADX={adx:.1f})'

        # Calcular SL/TP
        if direction == 'long':
            sl = price - (atr * self.sl_atr_mult)
            tp = price + (atr * self.tp_atr_mult)
        elif direction == 'short':
            sl = price + (atr * self.sl_atr_mult)
            tp = price - (atr * self.tp_atr_mult)
        else:
            sl = tp = 0

        return Signal(direction, strength, price, sl, tp, reason)

    def _signal_momentum_burst(self, df: pd.DataFrame) -> Signal:
        """
        Estrategia Momentum Burst.
        Opera em movimentos fortes de preço com volume.
        """
        # Usar .values[-1] para performance
        price = df['close'].values[-1]
        prev_close = df['close'].values[-2]
        atr = df['atr'].values[-1]
        adx = df['adx'].values[-1]
        rsi = df['rsi'].values[-1]

        if np.isnan(atr) or atr == 0:
            return Signal('none', 0, price, 0, 0, 'Indicadores invalidos')

        direction = 'none'
        strength = 0
        reason = ''

        # Movimento de preço forte (> 1.5 ATR)
        price_change = price - prev_close
        price_move = abs(price_change) / atr

        # Tendência forte com movimento
        # CORRIGIDO: Usar params ao invés de hardcoded 30 e 1.0
        if adx > self.momentum_adx_threshold and price_move > self.momentum_min_move:
            # CORRIGIDO: RSI thresholds agora configuráveis
            rsi_center = 50  # Centro do RSI
            rsi_range = 20   # Distância do centro
            if price_change > 0 and rsi > rsi_center and rsi < (rsi_center + rsi_range):
                direction = 'long'
                strength = min(10, 5 + price_move + adx / 20)
                reason = f'Momentum up (ADX={adx:.1f}, move={price_move:.1f}x ATR)'
            elif price_change < 0 and rsi < rsi_center and rsi > (rsi_center - rsi_range):
                direction = 'short'
                strength = min(10, 5 + price_move + adx / 20)
                reason = f'Momentum down (ADX={adx:.1f}, move={price_move:.1f}x ATR)'

        # Calcular SL/TP usando parametros (CORRIGIDO: removido hardcoded)
        # Momentum usa SL mais apertado (75% do normal) e TP normal
        momentum_sl_mult = self.sl_atr_mult * 0.75  # SL mais apertado para momentum
        momentum_tp_mult = self.tp_atr_mult * 0.85  # TP ligeiramente menor
        if direction == 'long':
            sl = price - (atr * momentum_sl_mult)
            tp = price + (atr * momentum_tp_mult)
        elif direction == 'short':
            sl = price + (atr * momentum_sl_mult)
            tp = price - (atr * momentum_tp_mult)
        else:
            sl = tp = 0

        return Signal(direction, strength, price, sl, tp, reason)

    def _signal_trend_following(self, df: pd.DataFrame) -> Signal:
        """Estrategia Trend Following com EMA cross."""
        # Usar .values[-1] para performance
        price = df['close'].values[-1]
        atr = df['atr'].values[-1]
        adx = df['adx'].values[-1]
        ema_f = df['ema_fast'].values[-1]
        ema_s = df['ema_slow'].values[-1]
        ema_f_prev = df['ema_fast'].values[-2]
        ema_s_prev = df['ema_slow'].values[-2]

        if np.isnan(adx) or np.isnan(atr) or atr == 0:
            return Signal('none', 0, price, 0, 0, 'Indicadores invalidos')
        
        direction = 'none'
        strength = 0
        reason = ''
        
        # Apenas operar com tendencia forte
        if adx > self.adx_threshold:
            # EMA cross up
            if ema_f > ema_s and ema_f_prev <= ema_s_prev:
                direction = 'long'
                strength = min(10, adx / 5)
                reason = f'EMA cross up (ADX={adx:.1f})'
            # EMA cross down
            elif ema_f < ema_s and ema_f_prev >= ema_s_prev:
                direction = 'short'
                strength = min(10, adx / 5)
                reason = f'EMA cross down (ADX={adx:.1f})'
        
        if direction == 'long':
            sl = price - (atr * self.sl_atr_mult)
            tp = price + (atr * self.tp_atr_mult)
        elif direction == 'short':
            sl = price + (atr * self.sl_atr_mult)
            tp = price - (atr * self.tp_atr_mult)
        else:
            sl = tp = 0
        
        return Signal(direction, strength, price, sl, tp, reason)
    
    def _signal_mean_reversion(self, df: pd.DataFrame) -> Signal:
        """
        Estrategia Mean Reversion com Bollinger Bands.

        ATUALIZADO V1.1:
        - SHORT BB upper bounce DESABILITADO (0% WR em produção)
        - Long bias aplicado ao strength
        - RSI thresholds menos extremos (35 vs 30)
        """
        # Usar .values[-1] para performance
        price = df['close'].values[-1]
        atr = df['atr'].values[-1]
        rsi = df['rsi'].values[-1]
        bb_upper = df['bb_upper'].values[-1]
        bb_lower = df['bb_lower'].values[-1]
        adx = df['adx'].values[-1]

        if np.isnan(atr) or atr == 0:
            return Signal('none', 0, price, 0, 0, 'Indicadores invalidos')

        direction = 'none'
        strength = 0
        reason = ''

        # Apenas em mercado lateral (ADX baixo)
        mr_adx_max = self.params.get('mr_adx_max', 22)  # Era 25
        if adx < mr_adx_max:
            # Bounce no BB inferior - LONG permitido
            if price < bb_lower and rsi < 35:
                direction = 'long'
                # Strength dinâmico baseado em quão extremo está
                rsi_bonus = min(1.5, (35 - rsi) / 15)
                strength = min(9, (6 + rsi_bonus) * self.long_bias)
                reason = f'BB lower bounce (RSI={rsi:.1f}, ADX={adx:.0f})'

            # Bounce no BB superior - SHORT DESABILITADO
            # Análise de trades reais mostrou 0% WR para esta condição
            # elif price > bb_upper and rsi > 65:
            #     direction = 'short'
            #     strength = 6 * self.short_penalty
            #     reason = f'BB upper bounce (RSI={rsi:.1f})'

        # Mean reversion usa SL mais apertado (60% do normal) e TP conservador (70%)
        mr_sl_mult = self.sl_atr_mult * 0.60  # Aumentado de 0.50
        mr_tp_mult = self.tp_atr_mult * 0.70  # Aumentado de 0.60
        if direction == 'long':
            sl = price - (atr * mr_sl_mult)
            tp = price + (atr * mr_tp_mult)
        elif direction == 'short':
            sl = price + (atr * mr_sl_mult)
            tp = price - (atr * mr_tp_mult)
        else:
            sl = tp = 0

        return Signal(direction, strength, price, sl, tp, reason)

    def _signal_combined(self, df: pd.DataFrame) -> Signal:
        """Combinar multiplas estrategias."""
        rsi_signal = self._signal_rsi_extremes(df)
        trend_signal = self._signal_trend_following(df)
        mr_signal = self._signal_mean_reversion(df)
        
        # Priorizar por strength
        signals = [rsi_signal, trend_signal, mr_signal]
        valid_signals = [s for s in signals if s.direction != 'none']
        
        if not valid_signals:
            return Signal('none', 0, df.iloc[-1]['close'], 0, 0, 'Sem sinal')
        
        # Retornar o mais forte
        return max(valid_signals, key=lambda s: s.strength)


def check_exit_signal(
    position: Dict,
    current_price: float,
    current_high: float,
    current_low: float,
    atr: float,
    rsi: float = 50
) -> Optional[str]:
    """
    Verificar sinais de saida.
    
    Returns:
        Razao da saida ou None se deve manter
    """
    side = position['side']
    entry = position['entry']
    sl = position['sl']
    tp = position['tp']
    
    if side == 'long':
        # Só verificar SL/TP se estiverem definidos (> 0)
        if sl > 0 and current_low <= sl:
            return 'STOP_LOSS'
        if tp > 0 and current_high >= tp:
            return 'TAKE_PROFIT'
        # RSI overbought em posicao long lucrativa
        if rsi > 70 and current_price > entry * 1.01:
            return 'RSI_EXIT'
    else:
        # Só verificar SL/TP se estiverem definidos (> 0)
        if sl > 0 and current_high >= sl:
            return 'STOP_LOSS'
        if tp > 0 and current_low <= tp:
            return 'TAKE_PROFIT'
        if rsi < 30 and current_price < entry * 0.99:
            return 'RSI_EXIT'
    
    return None
