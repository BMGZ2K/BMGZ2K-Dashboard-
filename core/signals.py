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
import threading
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

# Importar Config centralizado
from .config import Config, get_param

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


def calculate_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calcular MACD (Moving Average Convergence Divergence).
    Conforme padrão TradingView/Binance.

    Returns:
        macd_line, signal_line, histogram
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_volume_profile(volume: pd.Series, period: int = 20) -> Tuple[pd.Series, pd.Series]:
    """
    Calcular perfil de volume.

    Returns:
        volume_ma: Media movel do volume
        volume_ratio: Volume atual / media (>1 = acima da media)
    """
    volume_ma = volume.rolling(period).mean()
    volume_ratio = volume / (volume_ma + 1e-10)
    return volume_ma, volume_ratio


def calculate_rsi_divergence(close: pd.Series, rsi: pd.Series, lookback: int = 14) -> Tuple[bool, bool]:
    """
    Detectar divergências RSI.

    Returns:
        bullish_divergence: Preço novo low + RSI higher low
        bearish_divergence: Preço novo high + RSI lower high
    """
    bullish_div = False
    bearish_div = False

    if len(close) < lookback + 2:
        return bullish_div, bearish_div

    # Últimos N candles
    price_recent = close.iloc[-lookback:]
    rsi_recent = rsi.iloc[-lookback:]

    # Encontrar minimos e maximos locais
    price_min_idx = price_recent.idxmin()
    price_max_idx = price_recent.idxmax()

    # Verificar divergencia bullish
    # Preço fez novo low, mas RSI fez higher low
    if price_min_idx == price_recent.index[-1] or price_min_idx == price_recent.index[-2]:
        # Preço está nos mínimos recentes
        prev_low_idx = price_recent.iloc[:-3].idxmin() if len(price_recent) > 3 else None
        if prev_low_idx is not None:
            if close.loc[price_min_idx] < close.loc[prev_low_idx]:
                # Preço fez novo low
                if rsi.loc[price_min_idx] > rsi.loc[prev_low_idx]:
                    # RSI fez higher low = divergência bullish
                    bullish_div = True

    # Verificar divergencia bearish
    # Preço fez novo high, mas RSI fez lower high
    if price_max_idx == price_recent.index[-1] or price_max_idx == price_recent.index[-2]:
        prev_high_idx = price_recent.iloc[:-3].idxmax() if len(price_recent) > 3 else None
        if prev_high_idx is not None:
            if close.loc[price_max_idx] > close.loc[prev_high_idx]:
                # Preço fez novo high
                if rsi.loc[price_max_idx] < rsi.loc[prev_high_idx]:
                    # RSI fez lower high = divergência bearish
                    bearish_div = True

    return bullish_div, bearish_div


def calculate_momentum(close: pd.Series, period: int = 10) -> pd.Series:
    """
    Calcular Rate of Change (Momentum).

    Returns:
        ROC em percentual
    """
    return ((close - close.shift(period)) / close.shift(period)) * 100


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
        # Combinar parâmetros passados com defaults do Config centralizado
        self.params = Config.get_strategy_params()
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

        # Bias params (do config centralizado - padrão neutro 1.0 se não definido)
        # NOTA: long_bias > 1.0 favorece LONGs, short_penalty < 1.0 penaliza SHORTs
        # Estes parâmetros devem ser validados via WFO, não hardcoded
        self.long_bias = self.params.get('long_bias', get_param('long_bias', 1.0))
        self.short_penalty = self.params.get('short_penalty', get_param('short_penalty', 1.0))

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

        # Confluence scoring params
        self.use_confluence = self.params.get('use_confluence', True)
        self.min_confluence_for_bonus = self.params.get('min_confluence_for_bonus', 2)

        # Multi-timeframe params (B2)
        self.use_htf_filter = self.params.get('use_htf_filter', True)
        self.htf_ema_fast = self.params.get('htf_ema_fast', 9)
        self.htf_ema_slow = self.params.get('htf_ema_slow', 21)
        self.htf_threshold = self.params.get('htf_threshold', 0.002)  # 0.2%

        # Cache para HTF trend (atualizado externamente) - thread-safe
        self._htf_trend_cache: Dict[str, str] = {}
        self._htf_cache_lock = threading.Lock()
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preparar dados com indicadores."""
        df = df.copy()

        # Garantir que timestamp é o índice (necessário para WFO multi-symbol)
        if 'timestamp' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            df = df.sort_index()

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

    def calculate_confluence_score(self, df: pd.DataFrame) -> Dict:
        """
        MELHORIA C2: Calcular score de confluência de indicadores.

        Analisa múltiplos indicadores e retorna quantos concordam em cada direção.
        Isso permite aumentar a confiança do sinal quando múltiplos indicadores concordam.

        Returns:
            Dict com:
            - long_confluence: número de indicadores apontando LONG
            - short_confluence: número de indicadores apontando SHORT
            - long_strength: força agregada dos sinais LONG
            - short_strength: força agregada dos sinais SHORT
            - signals: lista de (indicador, direção, força)
        """
        signals = []

        # Extrair valores
        rsi = df['rsi'].values[-1]
        stoch_k = df['stoch_k'].values[-1]
        ema_f = df['ema_fast'].values[-1]
        ema_s = df['ema_slow'].values[-1]
        price = df['close'].values[-1]
        bb_lower = df['bb_lower'].values[-1]
        bb_upper = df['bb_upper'].values[-1]

        # RSI signal
        if not np.isnan(rsi):
            if rsi < 30:
                signals.append(('rsi', 'long', min(1.5, (30 - rsi) / 10)))
            elif rsi > 70:
                signals.append(('rsi', 'short', min(1.5, (rsi - 70) / 10)))

        # Stochastic signal
        if not np.isnan(stoch_k):
            if stoch_k < 20:
                signals.append(('stoch', 'long', min(1.5, (20 - stoch_k) / 10)))
            elif stoch_k > 80:
                signals.append(('stoch', 'short', min(1.5, (stoch_k - 80) / 10)))

        # EMA signal (trend)
        if not np.isnan(ema_f) and not np.isnan(ema_s) and ema_s != 0:
            ema_diff_pct = (ema_f - ema_s) / ema_s * 100
            if ema_f > ema_s:
                signals.append(('ema', 'long', min(1.5, abs(ema_diff_pct) / 2)))
            else:
                signals.append(('ema', 'short', min(1.5, abs(ema_diff_pct) / 2)))

        # Bollinger Bands signal
        if not np.isnan(bb_lower) and not np.isnan(bb_upper):
            if price < bb_lower:
                signals.append(('bb', 'long', 1.0))
            elif price > bb_upper:
                signals.append(('bb', 'short', 1.0))

        # Contar confluência
        long_count = sum(1 for s in signals if s[1] == 'long')
        short_count = sum(1 for s in signals if s[1] == 'short')
        long_strength = sum(s[2] for s in signals if s[1] == 'long')
        short_strength = sum(s[2] for s in signals if s[1] == 'short')

        return {
            'long_confluence': long_count,
            'short_confluence': short_count,
            'long_strength': long_strength,
            'short_strength': short_strength,
            'signals': signals
        }

    def check_htf_trend(self, df_htf: pd.DataFrame) -> str:
        """
        MELHORIA B2: Verificar tendência no timeframe superior (HTF).

        Args:
            df_htf: DataFrame com dados do timeframe superior (ex: 4h)

        Returns:
            'bullish': EMA fast > EMA slow * (1 + threshold)
            'bearish': EMA fast < EMA slow * (1 - threshold)
            'neutral': Sem tendência clara
        """
        if df_htf is None or len(df_htf) < 30:
            return 'neutral'

        try:
            ema_fast = df_htf['close'].ewm(span=self.htf_ema_fast, adjust=False).mean().values[-1]
            ema_slow = df_htf['close'].ewm(span=self.htf_ema_slow, adjust=False).mean().values[-1]

            if np.isnan(ema_fast) or np.isnan(ema_slow) or ema_slow == 0:
                return 'neutral'

            # Verificar tendência com threshold
            if ema_fast > ema_slow * (1 + self.htf_threshold):
                return 'bullish'
            elif ema_fast < ema_slow * (1 - self.htf_threshold):
                return 'bearish'
            return 'neutral'

        except Exception:
            return 'neutral'

    def update_htf_trend(self, symbol: str, df_htf: pd.DataFrame) -> str:
        """
        Atualiza cache de tendência HTF para um símbolo.

        Args:
            symbol: Símbolo (ex: 'BTC/USDT')
            df_htf: DataFrame do timeframe superior

        Returns:
            Tendência detectada ('bullish', 'bearish', 'neutral')
        """
        trend = self.check_htf_trend(df_htf)
        self._htf_trend_cache[symbol] = trend
        return trend

    def get_htf_trend(self, symbol: str) -> str:
        """Obtém tendência HTF do cache."""
        return self._htf_trend_cache.get(symbol, 'neutral')

    def detect_market_regime(self, df: pd.DataFrame) -> str:
        """
        FASE 3.4: Detectar regime de mercado baseado em ADX.

        Args:
            df: DataFrame com indicadores

        Returns:
            'trending' se ADX > 25
            'ranging' se ADX < 18
            'mixed' caso contrário
        """
        adx = df['adx'].values[-1]

        if np.isnan(adx):
            return 'mixed'

        if adx > 25:
            return 'trending'
        elif adx < 18:
            return 'ranging'
        else:
            return 'mixed'

    def validate_regime_for_signal(self, signal: 'Signal', df: pd.DataFrame) -> 'Signal':
        """
        FASE 3.4: Validar se sinal é apropriado para regime atual.

        Penaliza sinais que não combinam com o regime de mercado:
        - Mean reversion em mercado trending → penalizar
        - Momentum/trend em mercado ranging → penalizar

        Args:
            signal: Sinal gerado
            df: DataFrame com indicadores

        Returns:
            Signal ajustado
        """
        if signal.direction == 'none':
            return signal

        regime = self.detect_market_regime(df)
        strategy_type = signal.strategy.lower() if signal.strategy else ''

        # Identificar tipo de estratégia
        is_mean_reversion = 'reversion' in strategy_type or 'mr' in strategy_type or 'pullback' in strategy_type
        is_momentum = 'momentum' in strategy_type or 'burst' in strategy_type
        is_trend = 'trend' in strategy_type or 'ema' in strategy_type

        # Validar combinação regime/estratégia
        penalty = 1.0

        if is_mean_reversion and regime == 'trending':
            # Mean reversion em tendência forte → não funciona bem
            penalty = 0.3
        elif (is_momentum or is_trend) and regime == 'ranging':
            # Momentum/trend em mercado lateral → whipsaws
            penalty = 0.3
        elif regime == 'mixed':
            # Regime misto → sem penalidade (ocorre 70% do tempo)
            penalty = 1.0

        if penalty < 1.0:
            return Signal(
                signal.direction,
                signal.strength * penalty,
                signal.entry_price,
                signal.stop_loss,
                signal.take_profit,
                f"{signal.reason} [regime={regime}]",
                signal.strategy,
                signal.confidence * penalty
            )

        return signal

    def is_signal_aligned_with_htf(self, direction: str, symbol: str = None, htf_trend: str = None) -> bool:
        """
        MELHORIA B2: Verifica se o sinal está alinhado com a tendência HTF.

        Regras:
        - LONG permitido se HTF = bullish ou neutral
        - SHORT permitido se HTF = bearish ou neutral

        Args:
            direction: 'long' ou 'short'
            symbol: Símbolo para buscar no cache
            htf_trend: Tendência HTF já calculada (opcional)

        Returns:
            True se alinhado, False se contra a tendência
        """
        if not self.use_htf_filter:
            return True

        # Usar tendência passada ou buscar do cache
        trend = htf_trend or self.get_htf_trend(symbol or '')

        if direction == 'long':
            # LONG só se HTF bullish ou neutral
            return trend in ('bullish', 'neutral')
        elif direction == 'short':
            # SHORT só se HTF bearish ou neutral
            return trend in ('bearish', 'neutral')

        return True

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
        signal = None
        if self.strategy == 'rsi_extremes' or self.strategy == 'rsi_extreme':
            signal = self._signal_rsi_extremes(df)
        elif self.strategy == 'stoch_extreme':
            signal = self._signal_stoch_extreme(df)
        elif self.strategy == 'trend_following' or self.strategy == 'trend_adx':
            signal = self._signal_trend_following(df)
        elif self.strategy == 'mean_reversion' or self.strategy == 'pullback':
            signal = self._signal_mean_reversion(df)
        elif self.strategy == 'momentum_burst':
            signal = self._signal_momentum_burst(df)

        # ========== DEFAULT: Usar Combined (melhor performance) ==========
        if signal is None:
            if self._optimized_strategies is not None:
                opt_signal = self._optimized_strategies.generate_signal(df, 'combined')
                signal = Signal(
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
                signal = self._signal_stoch_extreme(df)  # Fallback para V1

        # ========== FASE 4.1: FILTRO HTF OBRIGATÓRIO (MULTI-TIMEFRAME) ==========
        # REJEITA sinais que vão contra a tendência do timeframe superior (4h)
        # Mudança de "penalizar 50%" para "rejeitar completamente"
        if signal.direction != 'none' and self.use_htf_filter:
            # Verificar alinhamento com HTF (usa cache interno)
            if not self.is_signal_aligned_with_htf(signal.direction):
                # FASE 4.1: Rejeitar completamente sinais contra HTF
                return Signal('none', 0, signal.entry_price, 0, 0, f'{signal.reason} [HTF REJECTED]')

        # ========== FASE 3.4: VALIDAÇÃO DE REGIME DE MERCADO ==========
        if signal.direction != 'none':
            signal = self.validate_regime_for_signal(signal, df)

        # ========== FASE 4.2: CONFLUENCE SCORING COM MULTIPLICADORES ==========
        # Mudança de bônus aditivo (+0.5) para multiplicadores (1.8x, 1.4x, 0.7x)
        if self.use_confluence and signal.direction != 'none':
            confluence = self.calculate_confluence_score(df)

            if signal.direction == 'long':
                conf_count = confluence['long_confluence']
            else:  # short
                conf_count = confluence['short_confluence']

            # FASE 4.2: Usar multiplicadores em vez de bônus aditivos
            if conf_count >= 3:  # 3+ indicadores alinhados
                strength_multiplier = 1.8  # 80% boost
                conf_label = 'STRONG'
            elif conf_count >= 2:  # 2 indicadores alinhados
                strength_multiplier = 1.4  # 40% boost
                conf_label = 'GOOD'
            elif conf_count == 1:  # Apenas 1 indicador
                strength_multiplier = 1.0  # Sem mudança
                conf_label = 'OK'
            else:  # 0 indicadores alinhados
                strength_multiplier = 0.7  # 30% penalidade
                conf_label = 'WEAK'

            new_strength = min(self.max_signal_strength, signal.strength * strength_multiplier)

            # Confiança baseada na confluência (0-1)
            new_confidence = min(1.0, conf_count / 4)

            # Criar signal com valores atualizados
            signal = Signal(
                signal.direction,
                new_strength,
                signal.entry_price,
                signal.stop_loss,
                signal.take_profit,
                f"{signal.reason} [conf={conf_label}]",
                signal.strategy,
                new_confidence
            )

        return signal
    
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

        # Validar que temos dados suficientes para acessar índices anteriores
        if len(stoch_k_arr) < 2 or len(ema_fast_arr) < 2:
            price = close_arr[-1] if len(close_arr) > 0 else 0
            return Signal('none', 0, price, 0, 0, 'Insufficient data for stochastic')

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

        # Long: Stochastic oversold EXTREMO com cross up
        # CORRIGIDO: Para reversões, não exigir EMA bullish (seria contraditório)
        # Condições: (1) Stoch oversold + cross up + ADX confirma tendência existe
        #            (2) OU EMA cross up (entrada em pullback)
        if stoch_k < self.stoch_oversold and stoch_cross_up:
            # ADX forte indica que havia tendência - bom para reversão
            if trend_strong or ema_cross_up:
                direction = 'long'
                # Cálculo otimizado de strength (usando params)
                oversold_bonus = min(2.0, (self.stoch_oversold - stoch_k) / 5)
                # Bonus maior se EMA também confirma, menor se só ADX
                trend_bonus = 2.0 if ema_cross_up else min(1.5, (adx - self.adx_min) / 20)
                # Se EMA bearish mas ADX forte, reduzir strength (mais arriscado)
                ema_penalty = 0 if ema_f >= ema_s else 0.5
                strength = min(self.max_signal_strength, self.stoch_base_strength + oversold_bonus + trend_bonus - ema_penalty)
                reason = f'Stoch extreme up ({stoch_k:.0f}) ADX {adx:.0f}'

        # Short: Stochastic overbought EXTREMO com cross down
        # CORRIGIDO: Mesma lógica - não exigir EMA bearish para reversão
        elif stoch_k > self.stoch_overbought and stoch_cross_down:
            if trend_strong or ema_cross_down:
                direction = 'short'
                # Cálculo otimizado de strength (usando params)
                overbought_bonus = min(2.0, (stoch_k - self.stoch_overbought) / 5)
                trend_bonus = 2.0 if ema_cross_down else min(1.5, (adx - self.adx_min) / 20)
                # Se EMA bullish mas ADX forte, reduzir strength (mais arriscado)
                ema_penalty = 0 if ema_f <= ema_s else 0.5
                strength = min(self.max_signal_strength, self.stoch_base_strength + overbought_bonus + trend_bonus - ema_penalty)
                reason = f'Stoch extreme down ({stoch_k:.0f}) ADX {adx:.0f}'

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

        MELHORIA B1: Filtros mais estritos para SHORT para evitar short squeezes.
        """
        # Usar .values[-1] para performance
        price = df['close'].values[-1]
        prev_close = df['close'].values[-2]
        atr = df['atr'].values[-1]
        adx = df['adx'].values[-1]
        rsi = df['rsi'].values[-1]
        ema_f = df['ema_fast'].values[-1]
        ema_s = df['ema_slow'].values[-1]

        if np.isnan(atr) or atr == 0:
            return Signal('none', 0, price, 0, 0, 'Indicadores invalidos')

        direction = 'none'
        strength = 0
        reason = ''

        # Movimento de preço forte (> 1.5 ATR)
        price_change = price - prev_close
        price_move = abs(price_change) / atr

        # Confirmações de tendência
        ema_bullish = ema_f > ema_s
        ema_bearish = ema_f < ema_s

        # Tendência forte com movimento
        if adx > self.momentum_adx_threshold and price_move > self.momentum_min_move:
            # LONG: RSI 45-65 + movimento up + EMA bullish (opcional)
            if price_change > 0 and rsi > 45 and rsi < 65:
                direction = 'long'
                strength = min(10, 5 + price_move + adx / 20)
                # Bonus se EMA confirma
                if ema_bullish:
                    strength = min(10, strength + 0.5)
                reason = f'Momentum up (ADX={adx:.1f}, move={price_move:.1f}x ATR)'

            # MELHORIA B1: SHORT com filtros mais estritos
            # RSI 35-55 (evita short squeezes) + movimento down + EMA bearish OBRIGATÓRIO
            elif price_change < 0 and rsi > 35 and rsi < 55:
                # OBRIGATÓRIO: EMA deve confirmar tendência bearish
                if ema_bearish:
                    direction = 'short'
                    strength = min(10, 5 + price_move + adx / 20)
                    reason = f'Momentum down (ADX={adx:.1f}, move={price_move:.1f}x ATR, EMA bear)'
                # Se EMA não confirma, reduzir muito a força ou não entrar
                elif adx > 35:  # Só permitir SHORT sem EMA se ADX muito forte
                    direction = 'short'
                    strength = min(7, 4 + price_move)  # Força reduzida
                    reason = f'Momentum down (ADX={adx:.1f}, move={price_move:.1f}x ATR, weak)'

        # Calcular SL/TP usando parametros
        # Momentum usa SL mais apertado (75% do normal) e TP normal
        momentum_sl_mult = self.sl_atr_mult * 0.75
        momentum_tp_mult = self.tp_atr_mult * 0.85
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

        ATUALIZADO V1.3 (Melhoria B3):
        - SHORT BB upper bounce com confirmações ainda mais estritas
        - Requer: candle anterior tocou BB upper + candle atual fechou ABAIXO
        - RSI > 70 (extremo)
        - ADX < 20 (confirma mercado lateral)
        - Candle bearish (price < prev_close)
        """
        # Usar .values[-1] para performance
        price = df['close'].values[-1]
        prev_close = df['close'].values[-2]
        prev_high = df['high'].values[-2]
        atr = df['atr'].values[-1]
        rsi = df['rsi'].values[-1]
        bb_upper = df['bb_upper'].values[-1]
        bb_upper_prev = df['bb_upper'].values[-2]
        bb_lower = df['bb_lower'].values[-1]
        adx = df['adx'].values[-1]

        if np.isnan(atr) or atr == 0:
            return Signal('none', 0, price, 0, 0, 'Indicadores invalidos')

        direction = 'none'
        strength = 0
        reason = ''

        # Verificar se shorts de MR estão habilitados
        mr_short_enabled = self.params.get('mr_short_enabled', True)

        # MELHORIA B3: Apenas em mercado lateral (ADX < mr_adx_max)
        mr_adx_max = self.params.get('mr_adx_max', 22)
        if adx < mr_adx_max:
            # Bounce no BB inferior - LONG permitido
            if price < bb_lower and rsi < 35:
                direction = 'long'
                # Strength dinâmico baseado em quão extremo está
                rsi_bonus = min(1.5, (35 - rsi) / 15)
                strength = min(9, (6 + rsi_bonus) * self.long_bias)
                reason = f'BB lower bounce (RSI={rsi:.1f}, ADX={adx:.0f})'

            # MELHORIA B3: Bounce no BB superior - SHORT com confirmações extras
            # Condições:
            # 1. Candle anterior tocou/excedeu BB upper (prev_high > bb_upper_prev)
            # 2. Candle atual fechou ABAIXO do BB upper (reversão confirmada)
            # 3. RSI > 70 (extremo)
            # 4. Candle bearish (price < prev_close)
            elif mr_short_enabled and rsi > 70:
                # Confirmar que candle anterior tocou BB upper
                prev_touched_upper = prev_high > bb_upper_prev or prev_close > bb_upper_prev
                # Confirmar que candle atual fechou ABAIXO do BB upper (reversão)
                current_below_upper = price < bb_upper
                # Confirmar candle bearish
                bearish_candle = price < prev_close

                if prev_touched_upper and current_below_upper and bearish_candle:
                    direction = 'short'
                    # Strength dinâmico baseado em quão extremo está
                    rsi_bonus = min(1.5, (rsi - 70) / 15)
                    strength = min(9, (6 + rsi_bonus) * self.short_penalty)
                    reason = f'BB upper reversal (RSI={rsi:.1f}, ADX={adx:.0f}, confirmed)'

        # Mean reversion usa SL mais apertado (60% do normal) e TP conservador (70%)
        mr_sl_mult = self.sl_atr_mult * 0.60
        mr_tp_mult = self.tp_atr_mult * 0.70
        if direction == 'long':
            sl = price - (atr * mr_sl_mult)
            tp = price + (atr * mr_tp_mult)
        elif direction == 'short':
            sl = price + (atr * mr_sl_mult)
            tp = price - (atr * mr_tp_mult)
        else:
            sl = tp = 0

        return Signal(direction, strength, price, sl, tp, reason)




def check_exit_signal(
    position: Dict,
    current_price: float,
    current_high: float,
    current_low: float,
    atr: float,
    rsi: float = 50,
    rsi_exit_long: float = 70,
    rsi_exit_short: float = 30
) -> Optional[str]:
    """
    Verificar sinais de saida.

    Returns:
        Razao da saida ou None se deve manter
    """
    # CORRIGIDO (Bug #14): Validar keys com .get() para evitar KeyError
    side = position.get('side')
    entry = position.get('entry', 0)
    sl = position.get('sl', 0)
    tp = position.get('tp', 0)

    if not side or entry <= 0:
        return None
    
    if side == 'long':
        # Só verificar SL/TP se estiverem definidos (> 0)
        if sl > 0 and current_low <= sl:
            return 'STOP_LOSS'
        if tp > 0 and current_high >= tp:
            return 'TAKE_PROFIT'
        # RSI overbought em posicao long lucrativa
        if rsi > rsi_exit_long and current_price > entry * 1.01:
            return 'RSI_EXIT'
    else:
        # Só verificar SL/TP se estiverem definidos (> 0)
        if sl > 0 and current_high >= sl:
            return 'STOP_LOSS'
        if tp > 0 and current_low <= tp:
            return 'TAKE_PROFIT'
        if rsi < rsi_exit_short and current_price < entry * 0.99:
            return 'RSI_EXIT'
    
    return None
