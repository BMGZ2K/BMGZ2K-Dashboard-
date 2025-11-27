"""
RSI OPTIMIZED STRATEGY - Crypto Futures (1H Timeframe)

Estrategia baseada em pesquisa de mercado e melhores praticas para crypto:
- RSI com parametros otimizados para alta volatilidade
- Confirmacoes multiplas (ADX, Volume, EMA)
- SL/TP dinamicos baseados em ATR
- Filtros para evitar falsos sinais

Referencias:
- https://www.mc2.fi/blog/best-rsi-settings-for-1-hour-chart-crypto
- https://coindar.org/en/article/article/optimised-stochrsi-rsi-settings-for-crypto-day-trading-expert-parameters-1130
- https://www.altrady.com/blog/crypto-trading-strategies/combine-multiple-indicators
- https://coinstats.app/news/top-5-atr-stoploss-settings
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from .signals import (
    calculate_rsi,
    calculate_atr,
    calculate_ema,
    calculate_adx,
    Signal
)


@dataclass
class RSIOptimizedParams:
    """
    Parametros otimizados para RSI em crypto futures.

    PARAMETROS RSI:
    - rsi_period: 9 (otimo para 1h crypto, mais responsivo que 14)
    - rsi_oversold: 20 (extremo para crypto, evita falsos sinais)
    - rsi_overbought: 80 (extremo para crypto, evita falsos sinais)

    CONFIRMACOES:
    - adx_min: 20 (tendencia presente)
    - adx_strong: 30 (tendencia forte)
    - volume_min_ratio: 1.2 (volume acima da media)
    - ema_fast: 9 (rapido)
    - ema_slow: 21 (lento)

    SL/TP (baseado em ATR):
    - atr_period: 14 (padrao)
    - sl_atr_mult: 2.5 (otimo para crypto volatilidade)
    - tp_atr_mult: 5.0 (ratio 2:1 risk/reward)

    FILTROS:
    - min_signal_strength: 6.0 (minimo para entrar)
    - require_volume_confirmation: True
    - require_trend_confirmation: True
    """

    # RSI Settings (otimizado para crypto 1h)
    rsi_period: int = 9
    rsi_oversold: float = 20
    rsi_overbought: float = 80

    # RSI Zonas extremas (ainda mais restritivas)
    rsi_extreme_oversold: float = 15
    rsi_extreme_overbought: float = 85

    # ADX Settings
    adx_min: float = 20
    adx_strong: float = 30
    adx_very_strong: float = 40

    # Volume Settings
    volume_min_ratio: float = 1.2
    volume_strong_ratio: float = 2.0

    # EMA Settings
    ema_fast: int = 9
    ema_slow: int = 21
    ema_trend: int = 50  # Filtro de tendencia

    # ATR Settings (otimizado para crypto)
    atr_period: int = 14
    sl_atr_mult: float = 2.5
    tp_atr_mult: float = 5.0

    # Dynamic SL/TP (ajustes por volatilidade)
    sl_atr_mult_low_vol: float = 2.0
    sl_atr_mult_high_vol: float = 3.5
    tp_atr_mult_low_vol: float = 4.0
    tp_atr_mult_high_vol: float = 6.0

    # Signal Strength
    min_signal_strength: float = 6.0

    # Filters
    require_volume_confirmation: bool = True
    require_trend_confirmation: bool = True
    require_ema_alignment: bool = True

    # Exit Settings
    trailing_stop_activation: float = 2.0  # Ativar trailing apos 2x ATR lucro
    trailing_stop_distance: float = 1.5  # Distance do trailing stop
    rsi_exit_long: float = 75  # Sair de long quando RSI > 75
    rsi_exit_short: float = 25  # Sair de short quando RSI < 25


class RSIOptimizedStrategy:
    """
    Estrategia RSI Otimizada para Crypto Futures.

    LOGICA DE ENTRADA (LONG):
    1. RSI < 20 (oversold extremo)
    2. RSI comecando a subir (divergencia positiva)
    3. ADX > 20 (tendencia presente)
    4. Volume > 1.2x media (confirmacao)
    5. Preco acima EMA50 OU EMA9 > EMA21 (tendencia alta)

    LOGICA DE ENTRADA (SHORT):
    1. RSI > 80 (overbought extremo)
    2. RSI comecando a cair (divergencia negativa)
    3. ADX > 20 (tendencia presente)
    4. Volume > 1.2x media (confirmacao)
    5. Preco abaixo EMA50 OU EMA9 < EMA21 (tendencia baixa)

    LOGICA DE SAIDA:
    1. Stop Loss: 2.5x ATR (ajustavel por volatilidade)
    2. Take Profit: 5.0x ATR (ratio 2:1)
    3. Trailing Stop: Ativa apos 2x ATR lucro
    4. RSI Exit: Sai quando RSI cruza zona oposta
    """

    def __init__(self, params: Optional[RSIOptimizedParams] = None):
        self.params = params or RSIOptimizedParams()

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preparar dados com todos os indicadores necessarios.
        """
        df = df.copy()

        # RSI
        df['rsi'] = calculate_rsi(df['close'], self.params.rsi_period)
        df['rsi_prev'] = df['rsi'].shift(1)
        df['rsi_slope'] = df['rsi'] - df['rsi_prev']

        # ATR
        df['atr'] = calculate_atr(
            df['high'],
            df['low'],
            df['close'],
            self.params.atr_period
        )

        # EMAs
        df['ema_fast'] = calculate_ema(df['close'], self.params.ema_fast)
        df['ema_slow'] = calculate_ema(df['close'], self.params.ema_slow)
        df['ema_trend'] = calculate_ema(df['close'], self.params.ema_trend)

        # ADX
        df['adx'] = calculate_adx(df['high'], df['low'], df['close'])

        # Volume
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-10)

        # Price momentum
        df['momentum'] = df['close'].pct_change(5) * 100

        return df

    def generate_signal(
        self,
        df: pd.DataFrame,
        precomputed: bool = False
    ) -> Signal:
        """
        Gerar sinal de trading.

        Args:
            df: DataFrame com dados OHLCV
            precomputed: Se True, assume indicadores ja calculados

        Returns:
            Signal object com direcao, strength, SL, TP
        """
        if len(df) < 50:
            return Signal('none', 0, 0, 0, 0, 'Dados insuficientes')

        # Calcular indicadores se necessario
        if not precomputed or 'rsi' not in df.columns:
            df = self.prepare_data(df)

        # Extrair valores atuais
        idx = -1
        price = df['close'].iloc[idx]
        rsi = df['rsi'].iloc[idx]
        rsi_prev = df['rsi'].iloc[idx-1]
        rsi_slope = df['rsi_slope'].iloc[idx]
        atr = df['atr'].iloc[idx]
        adx = df['adx'].iloc[idx]
        ema_fast = df['ema_fast'].iloc[idx]
        ema_slow = df['ema_slow'].iloc[idx]
        ema_trend = df['ema_trend'].iloc[idx]
        volume_ratio = df['volume_ratio'].iloc[idx]
        momentum = df['momentum'].iloc[idx]

        # Validar dados
        if np.isnan(rsi) or np.isnan(atr) or atr == 0:
            return Signal('none', 0, price, 0, 0, 'Indicadores invalidos')

        # ===== SISTEMA DE PONTUACAO =====
        long_score = 0.0
        short_score = 0.0
        long_reasons = []
        short_reasons = []

        # ===== LONG SIGNALS =====

        # 1. RSI Oversold (peso: 3.0)
        if rsi < self.params.rsi_oversold:
            oversold_strength = (self.params.rsi_oversold - rsi) / 5
            long_score += min(3.0, 2.0 + oversold_strength)
            long_reasons.append(f'RSI oversold ({rsi:.1f})')

            # Bonus: RSI EXTREMO
            if rsi < self.params.rsi_extreme_oversold:
                long_score += 1.5
                long_reasons.append('RSI EXTREME oversold')

        # 2. RSI Recovery (divergencia positiva) (peso: 2.0)
        if rsi < self.params.rsi_oversold and rsi_slope > 0:
            recovery_strength = min(2.0, rsi_slope / 2)
            long_score += recovery_strength
            long_reasons.append(f'RSI recovery (+{rsi_slope:.1f})')

        # 3. Trend Confirmation - ADX (peso: 2.0)
        if self.params.require_trend_confirmation:
            if adx > self.params.adx_min:
                trend_strength = min(2.0, (adx - self.params.adx_min) / 10)
                long_score += trend_strength
                long_reasons.append(f'ADX trend ({adx:.1f})')

                # Bonus: ADX muito forte
                if adx > self.params.adx_very_strong:
                    long_score += 1.0
                    long_reasons.append('ADX STRONG')

        # 4. Volume Confirmation (peso: 1.5)
        if self.params.require_volume_confirmation:
            if volume_ratio > self.params.volume_min_ratio:
                vol_strength = min(1.5, (volume_ratio - 1) * 1.5)
                long_score += vol_strength
                long_reasons.append(f'Volume ({volume_ratio:.1f}x)')

                # Bonus: Volume FORTE
                if volume_ratio > self.params.volume_strong_ratio:
                    long_score += 1.0
                    long_reasons.append('Volume STRONG')

        # 5. EMA Alignment (peso: 2.0)
        if self.params.require_ema_alignment:
            # Preco acima EMA trend = bullish
            if price > ema_trend:
                long_score += 1.0
                long_reasons.append('Price > EMA50')

            # EMA fast > EMA slow = crossover bullish
            if ema_fast > ema_slow:
                long_score += 1.0
                long_reasons.append('EMA9 > EMA21')

                # Bonus: Crossover recente
                ema_fast_prev = df['ema_fast'].iloc[idx-1]
                ema_slow_prev = df['ema_slow'].iloc[idx-1]
                if ema_fast_prev <= ema_slow_prev:
                    long_score += 1.0
                    long_reasons.append('EMA CROSS UP')

        # 6. Momentum Positivo (peso: 1.0)
        if momentum > 0 and rsi < self.params.rsi_oversold:
            long_score += min(1.0, abs(momentum) / 2)
            long_reasons.append(f'Momentum +{momentum:.1f}%')

        # ===== SHORT SIGNALS =====

        # 1. RSI Overbought (peso: 3.0)
        if rsi > self.params.rsi_overbought:
            overbought_strength = (rsi - self.params.rsi_overbought) / 5
            short_score += min(3.0, 2.0 + overbought_strength)
            short_reasons.append(f'RSI overbought ({rsi:.1f})')

            # Bonus: RSI EXTREMO
            if rsi > self.params.rsi_extreme_overbought:
                short_score += 1.5
                short_reasons.append('RSI EXTREME overbought')

        # 2. RSI Decline (divergencia negativa) (peso: 2.0)
        if rsi > self.params.rsi_overbought and rsi_slope < 0:
            decline_strength = min(2.0, abs(rsi_slope) / 2)
            short_score += decline_strength
            short_reasons.append(f'RSI decline ({rsi_slope:.1f})')

        # 3. Trend Confirmation - ADX (peso: 2.0)
        if self.params.require_trend_confirmation:
            if adx > self.params.adx_min:
                trend_strength = min(2.0, (adx - self.params.adx_min) / 10)
                short_score += trend_strength
                short_reasons.append(f'ADX trend ({adx:.1f})')

                # Bonus: ADX muito forte
                if adx > self.params.adx_very_strong:
                    short_score += 1.0
                    short_reasons.append('ADX STRONG')

        # 4. Volume Confirmation (peso: 1.5)
        if self.params.require_volume_confirmation:
            if volume_ratio > self.params.volume_min_ratio:
                vol_strength = min(1.5, (volume_ratio - 1) * 1.5)
                short_score += vol_strength
                short_reasons.append(f'Volume ({volume_ratio:.1f}x)')

                # Bonus: Volume FORTE
                if volume_ratio > self.params.volume_strong_ratio:
                    short_score += 1.0
                    short_reasons.append('Volume STRONG')

        # 5. EMA Alignment (peso: 2.0)
        if self.params.require_ema_alignment:
            # Preco abaixo EMA trend = bearish
            if price < ema_trend:
                short_score += 1.0
                short_reasons.append('Price < EMA50')

            # EMA fast < EMA slow = crossover bearish
            if ema_fast < ema_slow:
                short_score += 1.0
                short_reasons.append('EMA9 < EMA21')

                # Bonus: Crossover recente
                ema_fast_prev = df['ema_fast'].iloc[idx-1]
                ema_slow_prev = df['ema_slow'].iloc[idx-1]
                if ema_fast_prev >= ema_slow_prev:
                    short_score += 1.0
                    short_reasons.append('EMA CROSS DOWN')

        # 6. Momentum Negativo (peso: 1.0)
        if momentum < 0 and rsi > self.params.rsi_overbought:
            short_score += min(1.0, abs(momentum) / 2)
            short_reasons.append(f'Momentum {momentum:.1f}%')

        # ===== DECISAO FINAL =====

        direction = 'none'
        strength = 0.0
        reason = 'No signal'

        if long_score >= self.params.min_signal_strength and long_score > short_score:
            direction = 'long'
            strength = min(10.0, long_score)
            reason = ' | '.join(long_reasons)
        elif short_score >= self.params.min_signal_strength and short_score > long_score:
            direction = 'short'
            strength = min(10.0, short_score)
            reason = ' | '.join(short_reasons)

        # ===== CALCULAR SL/TP DINAMICOS =====

        sl, tp = self._calculate_sl_tp(
            direction=direction,
            price=price,
            atr=atr,
            adx=adx,
            volatility_ratio=volume_ratio
        )

        return Signal(
            direction=direction,
            strength=strength,
            entry_price=price,
            stop_loss=sl,
            take_profit=tp,
            reason=reason
        )

    def _calculate_sl_tp(
        self,
        direction: str,
        price: float,
        atr: float,
        adx: float,
        volatility_ratio: float
    ) -> Tuple[float, float]:
        """
        Calcular Stop Loss e Take Profit dinamicos baseados em ATR.

        AJUSTES DINAMICOS:
        - Volatilidade alta (volume_ratio > 2.0): SL/TP mais largos
        - Volatilidade baixa (volume_ratio < 1.0): SL/TP mais apertados
        - ADX forte (> 40): TP mais largo (deixar lucro correr)
        - ADX fraco (< 25): TP mais conservador

        Args:
            direction: 'long', 'short', ou 'none'
            price: Preco atual
            atr: Average True Range
            adx: ADX value
            volatility_ratio: Volume ratio (volatilidade proxy)

        Returns:
            (stop_loss, take_profit)
        """
        if direction == 'none':
            return 0, 0

        # Ajustar multiplicadores por volatilidade
        if volatility_ratio > 2.0:
            # Alta volatilidade: SL/TP mais largos
            sl_mult = self.params.sl_atr_mult_high_vol
            tp_mult = self.params.tp_atr_mult_high_vol
        elif volatility_ratio < 1.0:
            # Baixa volatilidade: SL/TP mais apertados
            sl_mult = self.params.sl_atr_mult_low_vol
            tp_mult = self.params.tp_atr_mult_low_vol
        else:
            # Volatilidade normal
            sl_mult = self.params.sl_atr_mult
            tp_mult = self.params.tp_atr_mult

        # Ajustar TP por forca da tendencia (ADX)
        if adx > self.params.adx_very_strong:
            # Tendencia muito forte: deixar lucro correr
            tp_mult *= 1.3
        elif adx > self.params.adx_strong:
            # Tendencia forte: TP normal aumentado
            tp_mult *= 1.15
        elif adx < self.params.adx_min:
            # Tendencia fraca: TP conservador
            tp_mult *= 0.85

        # Calcular SL/TP
        if direction == 'long':
            sl = price - (atr * sl_mult)
            tp = price + (atr * tp_mult)
        else:  # short
            sl = price + (atr * sl_mult)
            tp = price - (atr * tp_mult)

        return sl, tp

    def check_exit_signal(
        self,
        df: pd.DataFrame,
        position: Dict,
        precomputed: bool = False
    ) -> Optional[str]:
        """
        Verificar sinais de saida para posicao aberta.

        LOGICA DE SAIDA:
        1. Stop Loss atingido
        2. Take Profit atingido
        3. Trailing Stop ativado e atingido
        4. RSI cruza zona oposta (exit antecipado)
        5. Divergencia forte contra a posicao

        Args:
            df: DataFrame com dados
            position: Dict com info da posicao (side, entry, sl, tp, amt)
            precomputed: Se indicadores ja foram calculados

        Returns:
            String com razao da saida ou None
        """
        if not position or position.get('amt', 0) == 0:
            return None

        # Calcular indicadores se necessario
        if not precomputed or 'rsi' not in df.columns:
            df = self.prepare_data(df)

        # Extrair info
        side = position['side']
        entry = position['entry']
        sl = position.get('sl', 0)
        tp = position.get('tp', 0)

        current_price = df['close'].iloc[-1]
        current_high = df['high'].iloc[-1]
        current_low = df['low'].iloc[-1]
        rsi = df['rsi'].iloc[-1]
        rsi_prev = df['rsi'].iloc[-1]
        atr = df['atr'].iloc[-1]

        # 1. Stop Loss
        if side == 'long':
            if sl > 0 and current_low <= sl:
                return 'STOP_LOSS'
        else:
            if sl > 0 and current_high >= sl:
                return 'STOP_LOSS'

        # 2. Take Profit
        if side == 'long':
            if tp > 0 and current_high >= tp:
                return 'TAKE_PROFIT'
        else:
            if tp > 0 and current_low <= tp:
                return 'TAKE_PROFIT'

        # 3. Trailing Stop (ativar apos lucro de 2x ATR)
        if side == 'long':
            pnl = current_price - entry
            if pnl > (atr * self.params.trailing_stop_activation):
                trailing_sl = current_price - (atr * self.params.trailing_stop_distance)
                if current_low <= trailing_sl:
                    return 'TRAILING_STOP'
        else:
            pnl = entry - current_price
            if pnl > (atr * self.params.trailing_stop_activation):
                trailing_sl = current_price + (atr * self.params.trailing_stop_distance)
                if current_high >= trailing_sl:
                    return 'TRAILING_STOP'

        # 4. RSI Exit (quando cruza para zona oposta)
        if side == 'long':
            # Sair de long quando RSI fica overbought
            if rsi > self.params.rsi_exit_long:
                # Confirmar que esta caindo
                if rsi < rsi_prev:
                    return f'RSI_EXIT (RSI={rsi:.1f})'
        else:
            # Sair de short quando RSI fica oversold
            if rsi < self.params.rsi_exit_short:
                # Confirmar que esta subindo
                if rsi > rsi_prev:
                    return f'RSI_EXIT (RSI={rsi:.1f})'

        return None

    def get_params_dict(self) -> Dict:
        """
        Retornar parametros como dicionario (para compatibilidade com sistema).
        """
        return {
            'strategy': 'rsi_optimized',
            'rsi_period': self.params.rsi_period,
            'rsi_oversold': self.params.rsi_oversold,
            'rsi_overbought': self.params.rsi_overbought,
            'adx_min': self.params.adx_min,
            'atr_period': self.params.atr_period,
            'sl_atr_mult': self.params.sl_atr_mult,
            'tp_atr_mult': self.params.tp_atr_mult,
            'ema_fast': self.params.ema_fast,
            'ema_slow': self.params.ema_slow,
            'min_signal_strength': self.params.min_signal_strength,
        }


# ===== PARAMETROS RECOMENDADOS =====

# Configuracao CONSERVADORA (menor risco, menos trades)
CONSERVATIVE_PARAMS = RSIOptimizedParams(
    rsi_period=14,
    rsi_oversold=15,
    rsi_overbought=85,
    adx_min=25,
    volume_min_ratio=1.5,
    sl_atr_mult=3.0,
    tp_atr_mult=6.0,
    min_signal_strength=7.0,
    require_volume_confirmation=True,
    require_trend_confirmation=True,
    require_ema_alignment=True,
)

# Configuracao BALANCEADA (recomendada)
BALANCED_PARAMS = RSIOptimizedParams(
    rsi_period=9,
    rsi_oversold=20,
    rsi_overbought=80,
    adx_min=20,
    volume_min_ratio=1.2,
    sl_atr_mult=2.5,
    tp_atr_mult=5.0,
    min_signal_strength=6.0,
    require_volume_confirmation=True,
    require_trend_confirmation=True,
    require_ema_alignment=True,
)

# Configuracao AGRESSIVA (mais trades, maior risco)
AGGRESSIVE_PARAMS = RSIOptimizedParams(
    rsi_period=7,
    rsi_oversold=25,
    rsi_overbought=75,
    adx_min=15,
    volume_min_ratio=1.0,
    sl_atr_mult=2.0,
    tp_atr_mult=4.0,
    min_signal_strength=5.0,
    require_volume_confirmation=False,
    require_trend_confirmation=True,
    require_ema_alignment=False,
)


# ===== EXEMPLO DE USO =====

if __name__ == '__main__':
    """
    Exemplo de uso da estrategia RSI Optimized.
    """

    # Criar estrategia com parametros balanceados
    strategy = RSIOptimizedStrategy(BALANCED_PARAMS)

    # Supondo que voce tem um DataFrame com dados OHLCV
    # df = fetch_data('BTCUSDT', '1h', ...)

    # Preparar dados (calcular indicadores)
    # df = strategy.prepare_data(df)

    # Gerar sinal
    # signal = strategy.generate_signal(df, precomputed=True)

    # print(f"Direcao: {signal.direction}")
    # print(f"Strength: {signal.strength:.1f}/10")
    # print(f"Entry: {signal.entry_price:.2f}")
    # print(f"SL: {signal.stop_loss:.2f}")
    # print(f"TP: {signal.take_profit:.2f}")
    # print(f"Razao: {signal.reason}")

    print("RSI Optimized Strategy - Parametros:")
    print(f"- RSI Period: {strategy.params.rsi_period}")
    print(f"- RSI Oversold/Overbought: {strategy.params.rsi_oversold}/{strategy.params.rsi_overbought}")
    print(f"- ADX Min: {strategy.params.adx_min}")
    print(f"- SL/TP: {strategy.params.sl_atr_mult}x / {strategy.params.tp_atr_mult}x ATR")
    print(f"- Min Strength: {strategy.params.min_signal_strength}")
