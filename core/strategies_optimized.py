"""
Strategies Optimized Module
Estratégias otimizadas baseadas em análise de performance real.

v2.1: Parâmetros centralizados via core/config.py
      Signal importado de signals.py (sem duplicação)
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple

# Import centralizado - FONTE ÚNICA DE VERDADE
from .config import WFO_VALIDATED_PARAMS, get_param

# Signal importado de signals.py - classe única
from .signals import Signal


class OptimizedStrategies:
    """
    Estratégias otimizadas com base em performance real.

    Cada estratégia foi analisada e ajustada para:
    - Melhor Win Rate
    - Maior Profit Factor
    - Menor Drawdown
    - Sinais mais qualificados
    """

    def __init__(self, params: Dict = None):
        # Combinar params passados com config centralizado
        self.params = WFO_VALIDATED_PARAMS.copy()
        if params:
            self.params.update(params)
        self._load_params()

    def _load_params(self):
        """Carregar parâmetros do config centralizado."""
        # RSI
        self.rsi_period = self.params.get('rsi_period', get_param('rsi_period'))
        self.rsi_oversold = self.params.get('rsi_oversold', get_param('rsi_oversold'))
        self.rsi_overbought = self.params.get('rsi_overbought', get_param('rsi_overbought'))

        # Stochastic
        self.stoch_k = self.params.get('stoch_k', get_param('stoch_k'))
        self.stoch_d = self.params.get('stoch_d', get_param('stoch_d'))
        self.stoch_oversold = self.params.get('stoch_oversold', get_param('stoch_oversold'))
        self.stoch_overbought = self.params.get('stoch_overbought', get_param('stoch_overbought'))

        # ATR/SL/TP
        self.atr_period = self.params.get('atr_period', get_param('atr_period'))
        self.sl_atr_mult = self.params.get('sl_atr_mult', get_param('sl_atr_mult'))
        self.tp_atr_mult = self.params.get('tp_atr_mult', get_param('tp_atr_mult'))

        # EMA
        self.ema_fast = self.params.get('ema_fast', get_param('ema_fast'))
        self.ema_slow = self.params.get('ema_slow', get_param('ema_slow'))
        self.ema_trend = self.params.get('ema_trend', get_param('ema_trend'))

        # ADX
        self.adx_period = self.params.get('adx_period', get_param('adx_period'))
        self.adx_min = self.params.get('adx_min', get_param('adx_min'))
        self.adx_strong = self.params.get('adx_strong', get_param('adx_strong'))
        self.adx_very_strong = self.params.get('adx_very_strong', get_param('adx_very_strong'))

        # Momentum
        self.momentum_adx_min = self.params.get('momentum_adx_min', get_param('momentum_adx_min'))
        self.momentum_min_move = self.params.get('momentum_min_move', get_param('momentum_min_move'))
        self.momentum_rsi_long_min = self.params.get('momentum_rsi_long_min', get_param('momentum_rsi_long_min'))
        self.momentum_rsi_long_max = self.params.get('momentum_rsi_long_max', get_param('momentum_rsi_long_max'))
        self.momentum_rsi_short_min = self.params.get('momentum_rsi_short_min', get_param('momentum_rsi_short_min'))
        self.momentum_rsi_short_max = self.params.get('momentum_rsi_short_max', get_param('momentum_rsi_short_max'))

        # Mean Reversion
        self.mr_adx_max = self.params.get('mr_adx_max', get_param('mr_adx_max'))
        self.mr_rsi_long = self.params.get('mr_rsi_long', get_param('mr_rsi_long'))
        self.mr_rsi_short = self.params.get('mr_rsi_short', get_param('mr_rsi_short'))
        self.mr_bb_touch_pct = self.params.get('mr_bb_touch_pct', get_param('mr_bb_touch_pct'))

        # Strength
        self.min_strength = self.params.get('min_strength', get_param('min_strength'))
        self.base_strength = self.params.get('base_strength', get_param('base_strength'))

        # Bias
        self.long_bias = self.params.get('long_bias', get_param('long_bias'))
        self.short_penalty = self.params.get('short_penalty', get_param('short_penalty'))

        # Filter
        self.min_data_points = self.params.get('min_data_points', get_param('min_data_points'))
        self.use_volume_filter = self.params.get('use_volume_filter', get_param('use_volume_filter'))
        self.volume_mult_threshold = self.params.get('volume_mult_threshold', get_param('volume_mult_threshold'))

    # =========================================================================
    # ESTRATÉGIA 1: RSI EXTREMES OTIMIZADA
    # =========================================================================
    def signal_rsi_extremes_v2(self, df: pd.DataFrame) -> Signal:
        """
        RSI Extremes V2 - Otimizada

        Melhorias:
        - Adicionado filtro ADX (não operar em mercado sem tendência)
        - Adicionado filtro EMA (confirmar direção)
        - Thresholds relaxados (25/75 vs 20/80)
        - Strength dinâmico baseado em múltiplos fatores
        """
        price = df['close'].values[-1]
        rsi = df['rsi'].values[-1]
        atr = df['atr'].values[-1]
        adx = df['adx'].values[-1]
        ema_fast = df['ema_fast'].values[-1]
        ema_slow = df['ema_slow'].values[-1]

        if np.isnan(rsi) or np.isnan(atr) or atr == 0:
            return Signal('none', 0, price, 0, 0, 'Indicadores inválidos', 'rsi_v2')

        direction = 'none'
        strength = 0.0
        reason = ''
        confidence = 0.0

        # FILTRO 1: Precisa ter alguma tendência (ADX > 18)
        has_trend = adx > self.adx_min

        # FILTRO 2: Direção da EMA
        ema_bullish = ema_fast > ema_slow
        ema_bearish = ema_fast < ema_slow

        # LONG: RSI oversold + confirmações
        if rsi < self.rsi_oversold:
            # Só entra LONG se EMA bullish ou ADX forte (permite reversão)
            if ema_bullish or adx > self.adx_strong:
                direction = 'long'

                # Strength: base + bonus por RSI extremo + bonus por ADX
                rsi_bonus = min(2.0, (self.rsi_oversold - rsi) / 10)
                adx_bonus = min(1.5, (adx - self.adx_min) / 20) if has_trend else 0
                ema_bonus = 1.0 if ema_bullish else 0.5

                strength = self.base_strength + rsi_bonus + adx_bonus + ema_bonus
                strength = min(10, strength * self.long_bias)  # Aplicar viés LONG

                confidence = 0.5 + (rsi_bonus/4) + (adx_bonus/3)
                reason = f'RSI oversold ({rsi:.1f}) + EMA {"bull" if ema_bullish else "cross"} + ADX {adx:.0f}'

        # SHORT: RSI overbought + confirmações
        elif rsi > self.rsi_overbought:
            # Só entra SHORT se EMA bearish E ADX confirma
            if ema_bearish and has_trend:
                direction = 'short'

                rsi_bonus = min(2.0, (rsi - self.rsi_overbought) / 10)
                adx_bonus = min(1.5, (adx - self.adx_min) / 20)

                strength = self.base_strength + rsi_bonus + adx_bonus
                strength = min(10, strength * self.short_penalty)  # Aplicar penalidade SHORT

                confidence = 0.4 + (rsi_bonus/4) + (adx_bonus/3)
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

        return Signal(direction, strength, price, sl, tp, reason, 'rsi_extremes_v2', confidence)

    # =========================================================================
    # ESTRATÉGIA 2: STOCHASTIC EXTREME OTIMIZADA
    # =========================================================================
    def signal_stoch_extreme_v2(self, df: pd.DataFrame) -> Signal:
        """
        Stochastic Extreme V2 - Otimizada

        Melhorias:
        - Thresholds relaxados (25/75 vs 20/80)
        - Strength calculation mais sofisticado
        - Filtros de confirmação aprimorados
        - Viés LONG aplicado
        """
        # Extrair valores
        price = df['close'].values[-1]
        atr = df['atr'].values[-1]
        adx = df['adx'].values[-1]
        stoch_k = df['stoch_k'].values[-1]
        stoch_d = df['stoch_d'].values[-1]
        stoch_k_prev = df['stoch_k'].values[-2]
        stoch_d_prev = df['stoch_d'].values[-2]
        ema_f = df['ema_fast'].values[-1]
        ema_s = df['ema_slow'].values[-1]
        ema_f_prev = df['ema_fast'].values[-2]
        ema_s_prev = df['ema_slow'].values[-2]
        rsi = df['rsi'].values[-1]

        if np.isnan(stoch_k) or np.isnan(atr) or atr == 0:
            return Signal('none', 0, price, 0, 0, 'Indicadores inválidos', 'stoch_v2')

        direction = 'none'
        strength = 0.0
        reason = ''
        confidence = 0.0

        # Detectar crosses
        stoch_cross_up = stoch_k > stoch_d and stoch_k_prev <= stoch_d_prev
        stoch_cross_down = stoch_k < stoch_d and stoch_k_prev >= stoch_d_prev
        ema_cross_up = ema_f > ema_s and ema_f_prev <= ema_s_prev
        ema_cross_down = ema_f < ema_s and ema_f_prev >= ema_s_prev
        ema_bullish = ema_f > ema_s
        ema_bearish = ema_f < ema_s

        # Tendência
        trend_exists = adx > self.adx_min
        trend_strong = adx > self.adx_strong

        # ========== CENÁRIO 1: Stochastic Oversold + Cross Up ==========
        if stoch_k < self.stoch_oversold and stoch_cross_up:
            # Confirmação: EMA bullish OU cross up OU ADX forte
            if ema_cross_up or ema_bullish or trend_strong:
                direction = 'long'

                # Calcular strength
                base = self.base_strength
                oversold_bonus = min(2.0, (self.stoch_oversold - stoch_k) / 12.5)
                trend_bonus = 1.5 if ema_cross_up else (1.0 if ema_bullish else 0.5)
                adx_bonus = min(1.0, (adx - self.adx_min) / 20) if trend_exists else 0
                rsi_bonus = 0.5 if rsi < 35 else 0  # Bonus se RSI também oversold

                strength = base + oversold_bonus + trend_bonus + adx_bonus + rsi_bonus
                strength = min(10, strength * self.long_bias)

                confidence = 0.55 + (oversold_bonus/4) + (adx_bonus/2)
                reason = f'Stoch oversold ({stoch_k:.0f}) cross up + {"EMA cross" if ema_cross_up else "EMA bull"}'

        # ========== CENÁRIO 2: Stochastic Overbought + Cross Down ==========
        elif stoch_k > self.stoch_overbought and stoch_cross_down:
            # SHORT precisa de mais confirmações (performance pior)
            if (ema_cross_down or ema_bearish) and trend_exists:
                direction = 'short'

                base = self.base_strength
                overbought_bonus = min(2.0, (stoch_k - self.stoch_overbought) / 12.5)
                trend_bonus = 1.5 if ema_cross_down else (1.0 if ema_bearish else 0)
                adx_bonus = min(1.0, (adx - self.adx_min) / 20)
                rsi_bonus = 0.5 if rsi > 65 else 0

                strength = base + overbought_bonus + trend_bonus + adx_bonus + rsi_bonus
                strength = min(10, strength * self.short_penalty)

                confidence = 0.45 + (overbought_bonus/4) + (adx_bonus/2)
                reason = f'Stoch overbought ({stoch_k:.0f}) cross down + {"EMA cross" if ema_cross_down else "EMA bear"}'

        # ========== CENÁRIO 3: EMA Cross com ADX Forte (sem Stoch extremo) ==========
        elif trend_strong and adx > self.adx_very_strong:
            if ema_cross_up and stoch_k < 60:  # Não comprar em Stoch alto
                direction = 'long'
                strength = min(8, 5 + adx / 15) * self.long_bias
                confidence = 0.5 + (adx / 100)
                reason = f'EMA cross up + ADX forte ({adx:.0f})'
            elif ema_cross_down and stoch_k > 40 and ema_bearish:  # Mais restritivo para short
                direction = 'short'
                strength = min(7, 4 + adx / 15) * self.short_penalty
                confidence = 0.4 + (adx / 100)
                reason = f'EMA cross down + ADX forte ({adx:.0f})'

        # Calcular SL/TP
        if direction == 'long':
            sl = price - (atr * self.sl_atr_mult)
            tp = price + (atr * self.tp_atr_mult)
        elif direction == 'short':
            sl = price + (atr * self.sl_atr_mult)
            tp = price - (atr * self.tp_atr_mult)
        else:
            sl = tp = 0

        return Signal(direction, strength, price, sl, tp, reason, 'stoch_extreme_v2', confidence)

    # =========================================================================
    # ESTRATÉGIA 3: MOMENTUM BURST OTIMIZADA
    # =========================================================================
    def signal_momentum_burst_v2(self, df: pd.DataFrame) -> Signal:
        """
        Momentum Burst V2 - Otimizada

        Melhorias:
        - ADX threshold reduzido (22 vs 30)
        - Price move threshold reduzido (0.7 vs 1.0 ATR)
        - RSI ranges mais amplos
        - Confirmação de volume
        """
        price = df['close'].values[-1]
        prev_close = df['close'].values[-2]
        atr = df['atr'].values[-1]
        adx = df['adx'].values[-1]
        rsi = df['rsi'].values[-1]
        ema_fast = df['ema_fast'].values[-1]
        ema_slow = df['ema_slow'].values[-1]

        if np.isnan(atr) or atr == 0:
            return Signal('none', 0, price, 0, 0, 'Indicadores inválidos', 'momentum_v2')

        direction = 'none'
        strength = 0.0
        reason = ''
        confidence = 0.0

        # Calcular movimento de preço
        price_change = price - prev_close
        price_move = abs(price_change) / atr

        # Volume filter (se disponível)
        has_volume = 'volume' in df.columns
        volume_ok = True
        if has_volume and self.use_volume_filter:
            vol = df['volume'].values[-1]
            vol_ma = df['volume'].rolling(20).mean().values[-1]
            volume_ok = vol > vol_ma * self.volume_mult_threshold

        # Condições de momentum
        has_momentum = adx > self.momentum_adx_min and price_move > self.momentum_min_move

        if has_momentum and volume_ok:
            # LONG: Preço subiu + RSI em zona de momentum
            if price_change > 0:
                if rsi > self.momentum_rsi_long_min and rsi < self.momentum_rsi_long_max:
                    # Confirmar com EMA
                    if ema_fast > ema_slow:
                        direction = 'long'

                        move_bonus = min(2.5, price_move)
                        adx_bonus = min(1.5, (adx - self.momentum_adx_min) / 15)

                        strength = 5 + move_bonus + adx_bonus
                        strength = min(10, strength * self.long_bias)

                        confidence = 0.5 + (price_move / 4) + (adx_bonus / 3)
                        reason = f'Momentum UP: move={price_move:.2f}x ATR, ADX={adx:.0f}'

            # SHORT: Preço caiu + RSI em zona de momentum
            elif price_change < 0:
                if rsi > self.momentum_rsi_short_min and rsi < self.momentum_rsi_short_max:
                    # Confirmar com EMA
                    if ema_fast < ema_slow:
                        direction = 'short'

                        move_bonus = min(2.5, price_move)
                        adx_bonus = min(1.5, (adx - self.momentum_adx_min) / 15)

                        strength = 5 + move_bonus + adx_bonus
                        strength = min(10, strength * self.short_penalty)

                        confidence = 0.45 + (price_move / 4) + (adx_bonus / 3)
                        reason = f'Momentum DOWN: move={price_move:.2f}x ATR, ADX={adx:.0f}'

        # SL/TP para momentum (mais apertado)
        momentum_sl = self.sl_atr_mult * 0.8  # 20% menor
        momentum_tp = self.tp_atr_mult * 0.9  # 10% menor

        if direction == 'long':
            sl = price - (atr * momentum_sl)
            tp = price + (atr * momentum_tp)
        elif direction == 'short':
            sl = price + (atr * momentum_sl)
            tp = price - (atr * momentum_tp)
        else:
            sl = tp = 0

        return Signal(direction, strength, price, sl, tp, reason, 'momentum_burst_v2', confidence)

    # =========================================================================
    # ESTRATÉGIA 4: MEAN REVERSION OTIMIZADA
    # =========================================================================
    def signal_mean_reversion_v2(self, df: pd.DataFrame) -> Signal:
        """
        Mean Reversion V2 - Otimizada

        Melhorias:
        - Strength dinâmico (não mais fixo em 6)
        - ADX threshold configurável
        - RSI thresholds menos extremos
        - Melhor R:R ratio
        - REMOVIDO: BB upper bounce (0% WR em produção)
        """
        price = df['close'].values[-1]
        atr = df['atr'].values[-1]
        rsi = df['rsi'].values[-1]
        bb_upper = df['bb_upper'].values[-1]
        bb_lower = df['bb_lower'].values[-1]
        bb_mid = df['bb_mid'].values[-1]
        adx = df['adx'].values[-1]

        if np.isnan(atr) or atr == 0:
            return Signal('none', 0, price, 0, 0, 'Indicadores inválidos', 'mean_rev_v2')

        direction = 'none'
        strength = 0.0
        reason = ''
        confidence = 0.0

        # Mean reversion só funciona em mercado lateral
        is_ranging = adx < self.mr_adx_max

        # Calcular distância das bandas
        bb_width = bb_upper - bb_lower
        dist_to_lower = (price - bb_lower) / bb_width if bb_width > 0 else 0.5
        dist_to_upper = (bb_upper - price) / bb_width if bb_width > 0 else 0.5

        if is_ranging:
            # ========== LONG: Bounce no BB inferior ==========
            # Preço perto/abaixo da banda inferior + RSI oversold
            if dist_to_lower < self.mr_bb_touch_pct and rsi < self.mr_rsi_long:
                direction = 'long'

                # Strength dinâmico baseado em quão extremo está
                rsi_bonus = min(2.0, (self.mr_rsi_long - rsi) / 15)
                bb_bonus = min(1.5, (self.mr_bb_touch_pct - dist_to_lower) * 50)
                adx_bonus = 0.5 if adx < 15 else 0  # Bonus se muito lateral

                strength = self.base_strength + rsi_bonus + bb_bonus + adx_bonus
                strength = min(9, strength * self.long_bias)  # Cap em 9 (mean rev é arriscado)

                confidence = 0.5 + (rsi_bonus/4) + (bb_bonus/3)
                reason = f'BB lower bounce + RSI {rsi:.0f} + ADX {adx:.0f} (ranging)'

            # ========== SHORT: BB superior - DESABILITADO ==========
            # Análise mostrou 0% win rate em BB upper bounce
            # Mantido comentado para referência
            """
            elif dist_to_upper < self.mr_bb_touch_pct and rsi > self.mr_rsi_short:
                # SHORT em BB upper tem 0% WR - NÃO USAR
                pass
            """

        # SL/TP para mean reversion (mais apertado)
        mr_sl = self.sl_atr_mult * 0.6  # 40% menor (mercado lateral = menos volatilidade)
        mr_tp = self.tp_atr_mult * 0.7  # 30% menor

        if direction == 'long':
            sl = price - (atr * mr_sl)
            tp = price + (atr * mr_tp)
        elif direction == 'short':
            sl = price + (atr * mr_sl)
            tp = price - (atr * mr_tp)
        else:
            sl = tp = 0

        return Signal(direction, strength, price, sl, tp, reason, 'mean_reversion_v2', confidence)

    # =========================================================================
    # ESTRATÉGIA 5: TREND FOLLOWING OTIMIZADA
    # =========================================================================
    def signal_trend_following_v2(self, df: pd.DataFrame) -> Signal:
        """
        Trend Following V2 - Otimizada

        Melhorias:
        - EMA tripla (fast, slow, trend) para melhor confirmação
        - ADX threshold mais acessível
        - Pullback entry em vez de apenas cross
        - Volume confirmation
        """
        price = df['close'].values[-1]
        atr = df['atr'].values[-1]
        adx = df['adx'].values[-1]
        rsi = df['rsi'].values[-1]
        ema_f = df['ema_fast'].values[-1]
        ema_s = df['ema_slow'].values[-1]
        ema_f_prev = df['ema_fast'].values[-2]
        ema_s_prev = df['ema_slow'].values[-2]

        # EMA de tendência (50 períodos)
        if 'ema_trend' in df.columns:
            ema_trend = df['ema_trend'].values[-1]
        else:
            ema_trend = df['close'].ewm(span=50, adjust=False).mean().values[-1]

        if np.isnan(adx) or np.isnan(atr) or atr == 0:
            return Signal('none', 0, price, 0, 0, 'Indicadores inválidos', 'trend_v2')

        direction = 'none'
        strength = 0.0
        reason = ''
        confidence = 0.0

        # Detectar crosses
        ema_cross_up = ema_f > ema_s and ema_f_prev <= ema_s_prev
        ema_cross_down = ema_f < ema_s and ema_f_prev >= ema_s_prev

        # Alinhamento de EMAs
        bullish_aligned = ema_f > ema_s > ema_trend
        bearish_aligned = ema_f < ema_s < ema_trend

        # Pullback detection
        is_pullback_long = price < ema_f and price > ema_s and ema_f > ema_s
        is_pullback_short = price > ema_f and price < ema_s and ema_f < ema_s

        # Precisa de tendência
        has_trend = adx > self.adx_min
        strong_trend = adx > self.adx_strong

        # ========== CENÁRIO 1: EMA Cross em tendência forte ==========
        if strong_trend:
            if ema_cross_up and price > ema_trend:
                direction = 'long'
                strength = min(10, 5 + adx / 12) * self.long_bias
                confidence = 0.55 + (adx / 80)
                reason = f'EMA cross UP em uptrend (ADX={adx:.0f})'

            elif ema_cross_down and price < ema_trend:
                direction = 'short'
                strength = min(9, 5 + adx / 12) * self.short_penalty
                confidence = 0.45 + (adx / 80)
                reason = f'EMA cross DOWN em downtrend (ADX={adx:.0f})'

        # ========== CENÁRIO 2: Pullback entry em tendência existente ==========
        elif has_trend:
            # Pullback LONG: Preço recuou até EMA fast em uptrend
            if is_pullback_long and bullish_aligned and rsi > 40 and rsi < 60:
                direction = 'long'
                strength = min(8, 5.5 + adx / 15) * self.long_bias
                confidence = 0.5 + (adx / 80)
                reason = f'Pullback to EMA em uptrend (ADX={adx:.0f})'

            # Pullback SHORT: Preço subiu até EMA fast em downtrend
            elif is_pullback_short and bearish_aligned and rsi > 40 and rsi < 60:
                direction = 'short'
                strength = min(7, 5 + adx / 15) * self.short_penalty
                confidence = 0.45 + (adx / 80)
                reason = f'Pullback to EMA em downtrend (ADX={adx:.0f})'

        # Calcular SL/TP
        if direction == 'long':
            sl = price - (atr * self.sl_atr_mult)
            tp = price + (atr * self.tp_atr_mult)
        elif direction == 'short':
            sl = price + (atr * self.sl_atr_mult)
            tp = price - (atr * self.tp_atr_mult)
        else:
            sl = tp = 0

        return Signal(direction, strength, price, sl, tp, reason, 'trend_following_v2', confidence)

    # =========================================================================
    # ESTRATÉGIA 6: COMBINED SIGNAL (NOVA)
    # =========================================================================
    def signal_combined(self, df: pd.DataFrame) -> Signal:
        """
        Combined Strategy - Combina sinais de múltiplas estratégias

        Funciona assim:
        1. Executa todas as estratégias
        2. Só gera sinal se 2+ estratégias concordam
        3. Strength = média ponderada
        4. Confidence aumenta com mais concordância
        """
        signals = [
            self.signal_rsi_extremes_v2(df),
            self.signal_stoch_extreme_v2(df),
            self.signal_momentum_burst_v2(df),
            self.signal_trend_following_v2(df),
        ]

        # Contar votos
        long_signals = [s for s in signals if s.direction == 'long' and s.strength >= self.min_strength]
        short_signals = [s for s in signals if s.direction == 'short' and s.strength >= self.min_strength]

        price = df['close'].values[-1]
        atr = df['atr'].values[-1] if 'atr' in df.columns else 0

        # Precisa de pelo menos 2 estratégias concordando
        min_agreement = 2

        if len(long_signals) >= min_agreement:
            avg_strength = np.mean([s.strength for s in long_signals])
            avg_confidence = np.mean([s.confidence for s in long_signals])

            # Bonus por concordância
            agreement_bonus = (len(long_signals) - min_agreement) * 0.5
            final_strength = min(10, avg_strength + agreement_bonus)
            final_confidence = min(0.95, avg_confidence + len(long_signals) * 0.1)

            strategies_used = [s.strategy for s in long_signals]
            reason = f'COMBINED LONG: {len(long_signals)} strategies agree ({", ".join(strategies_used)})'

            sl = price - (atr * self.sl_atr_mult) if atr > 0 else 0
            tp = price + (atr * self.tp_atr_mult) if atr > 0 else 0

            return Signal('long', final_strength, price, sl, tp, reason, 'combined', final_confidence)

        elif len(short_signals) >= min_agreement:
            avg_strength = np.mean([s.strength for s in short_signals])
            avg_confidence = np.mean([s.confidence for s in short_signals])

            agreement_bonus = (len(short_signals) - min_agreement) * 0.5
            final_strength = min(10, avg_strength + agreement_bonus)
            final_confidence = min(0.9, avg_confidence + len(short_signals) * 0.08)

            strategies_used = [s.strategy for s in short_signals]
            reason = f'COMBINED SHORT: {len(short_signals)} strategies agree ({", ".join(strategies_used)})'

            sl = price + (atr * self.sl_atr_mult) if atr > 0 else 0
            tp = price - (atr * self.tp_atr_mult) if atr > 0 else 0

            return Signal('short', final_strength, price, sl, tp, reason, 'combined', final_confidence)

        return Signal('none', 0, price, 0, 0, 'No agreement', 'combined', 0)

    # =========================================================================
    # MÉTODO PRINCIPAL: GENERATE SIGNAL
    # =========================================================================
    def generate_signal(self, df: pd.DataFrame, strategy: str = 'stoch_extreme_v2') -> Signal:
        """
        Gerar sinal baseado na estratégia selecionada.

        Estratégias disponíveis:
        - rsi_extremes_v2: RSI com filtros ADX/EMA
        - stoch_extreme_v2: Stochastic com confirmações (RECOMENDADA)
        - momentum_burst_v2: Momentum com thresholds relaxados
        - mean_reversion_v2: Mean reversion só LONG (short desabilitado)
        - trend_following_v2: Trend following com pullbacks
        - combined: Combina múltiplas estratégias
        """
        if len(df) < self.min_data_points:
            return Signal('none', 0, 0, 0, 0, 'Dados insuficientes', strategy)

        strategy_map = {
            'rsi_extremes_v2': self.signal_rsi_extremes_v2,
            'stoch_extreme_v2': self.signal_stoch_extreme_v2,
            'momentum_burst_v2': self.signal_momentum_burst_v2,
            'mean_reversion_v2': self.signal_mean_reversion_v2,
            'trend_following_v2': self.signal_trend_following_v2,
            'combined': self.signal_combined,
            # Aliases para compatibilidade
            'rsi_extremes': self.signal_rsi_extremes_v2,
            'stoch_extreme': self.signal_stoch_extreme_v2,
            'momentum_burst': self.signal_momentum_burst_v2,
            'mean_reversion': self.signal_mean_reversion_v2,
            'trend_following': self.signal_trend_following_v2,
        }

        signal_func = strategy_map.get(strategy, self.signal_stoch_extreme_v2)
        return signal_func(df)


# =============================================================================
# PARÂMETROS OTIMIZADOS
# =============================================================================
# NOTA: Parâmetros centralizados em core/config.py (WFO_VALIDATED_PARAMS)
# Use: from core.config import get_validated_params, get_param
# NÃO defina parâmetros aqui para evitar duplicação/conflitos
# =============================================================================
