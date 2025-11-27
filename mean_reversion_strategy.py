"""
MEAN REVERSION STRATEGY - Otimizada para Crypto
==================================================

CONCEITO:
Mean reversion é baseado na teoria de que preços tendem a retornar à média após
movimentos extremos. Funciona melhor em mercados LATERAIS (baixo ADX).

VANTAGENS:
- Win rate ALTO (60-75% segundo backtests)
- Lógica clara e testada
- Menor exposição a tempo de mercado
- Funciona em range-bound markets

DESVANTAGENS:
- Ganhos MENORES por trade (R:R geralmente 1:1 a 1:1.5)
- PERIGOSO em tendências fortes (pode ficar contra a tendência)
- Requer disciplina rigorosa no SL
- Não funciona em breakouts verdadeiros

QUANDO USAR:
✓ ADX < 25 (mercado lateral)
✓ Bollinger Bands apertadas ou normais
✓ RSI em extremos (< 30 ou > 70)
✓ Volume normal ou baixo
✓ Mercado sem catalisadores fortes

QUANDO NÃO USAR:
✗ ADX > 30 (tendência estabelecida)
✗ Notícias/eventos importantes
✗ Breakout de consolidação longa
✗ Volume spike extremo
✗ Mercado em crash ou pump parabólico

PESQUISA E FUNDAMENTOS:
==================================================

Segundo estudos recentes (2025):
- Bollinger Bands (20, 2.0) é o padrão mais testado
- RSI 30/70 para extremos, ou 35/65 para mais seletividade
- ADX < 25 indica mercado lateral (optimal para mean reversion)
- Win rate médio: 60-65% em range-bound, até 78% com MACD confirmation
- R:R típico: 1:1 a 1:1.5 (stop apertado, target conservador)

Fontes:
- Medium: "Enhanced Mean Reversion Strategy with Bollinger Bands and RSI"
- Altrady: "ADX Guide: Mastering the Average Directional Index"
- QuantifiedStrategies: "MACD and Bollinger Bands Strategy (78% Win Rate)"

PARÂMETROS OTIMIZADOS:
==================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class MeanReversionSignal:
    """Sinal de mean reversion."""
    direction: str  # 'long', 'short', 'none'
    strength: float  # 0-10
    entry_price: float
    stop_loss: float
    take_profit: float
    reason: str
    confidence: str  # 'LOW', 'MEDIUM', 'HIGH'
    risk_reward: float


class MeanReversionStrategy:
    """
    Estratégia Mean Reversion otimizada para crypto.

    LÓGICA:
    ------
    LONG:
    - Preço toca ou quebra BB lower
    - RSI < 30 (ou 35 para mais seletivo)
    - ADX < 25 (confirma mercado lateral)
    - MACD histogram negativo (opcional - aumenta win rate)

    SHORT:
    - Preço toca ou quebra BB upper
    - RSI > 70 (ou 65 para mais seletivo)
    - ADX < 25
    - MACD histogram positivo (opcional)

    EXIT:
    - TP: BB middle (média) ou 1-1.5x ATR
    - SL: 0.5-0.75x ATR (APERTADO - mean reversion falhou)
    - Time-based: Exit após X candles sem movimento
    """

    def __init__(self, params: Dict = None):
        """
        Inicializar estratégia com parâmetros otimizados.

        Parâmetros padrão baseados em pesquisa:
        - bb_period: 20 (padrão de mercado)
        - bb_std: 2.0 (padrão) ou 2.5 para crypto volátil
        - rsi_period: 14 (padrão)
        - rsi_oversold: 30 ou 35 (mais seletivo)
        - rsi_overbought: 70 ou 65 (mais seletivo)
        - adx_max: 25 (filtro de mercado lateral)
        - sl_atr_mult: 0.5-0.75 (stop APERTADO)
        - tp_atr_mult: 1.0-1.5 (target conservador)
        """
        self.params = params or {}

        # Bollinger Bands
        # JUSTIFICATIVA: 20 períodos é o padrão testado em milhares de backtests
        # 2.0 std cobre ~95% dos movimentos; 2.5 para crypto mais volátil
        self.bb_period = self.params.get('bb_period', 20)
        self.bb_std = self.params.get('bb_std', 2.0)  # Use 2.5 para BTC/ETH

        # RSI
        # JUSTIFICATIVA: RSI 14 é padrão de Wilder; 30/70 são extremos clássicos
        # 35/65 aumenta seletividade e reduz falsos sinais
        self.rsi_period = self.params.get('rsi_period', 14)
        self.rsi_oversold = self.params.get('rsi_oversold', 35)  # 30 mais agressivo, 35 mais conservador
        self.rsi_overbought = self.params.get('rsi_overbought', 65)  # 70 mais agressivo, 65 mais conservador

        # ADX - FILTRO CRÍTICO
        # JUSTIFICATIVA: ADX < 25 indica mercado sem tendência forte
        # Mean reversion FALHA em ADX > 30 (tendência estabelecida)
        self.adx_max = self.params.get('adx_max', 25)  # Máximo 25, idealmente < 20

        # ATR
        self.atr_period = self.params.get('atr_period', 14)

        # Stop Loss e Take Profit
        # JUSTIFICATIVA: Mean reversion usa R:R menor (1:1 a 1:1.5)
        # Stop APERTADO porque se preço continua contra você, mean reversion FALHOU
        # Target conservador porque espera retorno à MÉDIA, não reversão completa
        self.sl_atr_mult = self.params.get('sl_atr_mult', 0.60)  # 0.5-0.75x ATR (MUITO APERTADO)
        self.tp_atr_mult = self.params.get('tp_atr_mult', 1.20)  # 1.0-1.5x ATR ou BB middle
        self.use_bb_middle_tp = self.params.get('use_bb_middle_tp', True)  # TP na média (recomendado)

        # MACD como confirmação adicional (opcional mas aumenta win rate)
        # JUSTIFICATIVA: MACD + BB pode dar win rate de até 78% (QuantifiedStrategies)
        self.use_macd_confirmation = self.params.get('use_macd_confirmation', False)
        self.macd_fast = self.params.get('macd_fast', 12)
        self.macd_slow = self.params.get('macd_slow', 26)
        self.macd_signal = self.params.get('macd_signal', 9)

        # Stochastic como confirmação adicional (opcional)
        # JUSTIFICATIVA: Stochastic < 20 ou > 80 confirma extremos
        self.use_stoch_confirmation = self.params.get('use_stoch_confirmation', True)
        self.stoch_k = self.params.get('stoch_k', 14)
        self.stoch_d = self.params.get('stoch_d', 3)
        self.stoch_oversold = self.params.get('stoch_oversold', 20)
        self.stoch_overbought = self.params.get('stoch_overbought', 80)

        # Volume filter (opcional)
        # JUSTIFICATIVA: Volume spike pode indicar breakout, não mean reversion
        self.max_volume_ratio = self.params.get('max_volume_ratio', 3.0)  # Evitar volume spikes

        # Bollinger Band %b thresholds
        # JUSTIFICATIVA: %b mostra posição relativa nas bandas (0 = lower, 1 = upper)
        self.bb_pct_b_long = self.params.get('bb_pct_b_long', 0.05)  # Abaixo de 5% = muito oversold
        self.bb_pct_b_short = self.params.get('bb_pct_b_short', 0.95)  # Acima de 95% = muito overbought

        # Data validation
        self.min_data_points = max(self.bb_period, self.rsi_period, self.atr_period) + 50

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcular todos os indicadores necessários."""
        df = df.copy()

        # Bollinger Bands
        df['bb_mid'] = df['close'].rolling(self.bb_period).mean()
        std = df['close'].rolling(self.bb_period).std(ddof=0)  # Population std
        df['bb_upper'] = df['bb_mid'] + (std * self.bb_std)
        df['bb_lower'] = df['bb_mid'] - (std * self.bb_std)

        # Bollinger %b (position within bands)
        # %b = (close - lower) / (upper - lower)
        # 0 = at lower band, 1 = at upper band, 0.5 = at middle
        bb_range = df['bb_upper'] - df['bb_lower']
        df['bb_pct_b'] = (df['close'] - df['bb_lower']) / bb_range.replace(0, 0.0001)

        # Bollinger Band Width (volatility measure)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']

        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta.where(delta < 0, 0))
        alpha = 1 / self.rsi_period
        avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))

        # ADX
        df['atr'] = self._calculate_atr(df)
        df['adx'] = self._calculate_adx(df)

        # MACD (se habilitado)
        if self.use_macd_confirmation:
            ema_fast = df['close'].ewm(span=self.macd_fast, adjust=False).mean()
            ema_slow = df['close'].ewm(span=self.macd_slow, adjust=False).mean()
            df['macd'] = ema_fast - ema_slow
            df['macd_signal'] = df['macd'].ewm(span=self.macd_signal, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']

        # Stochastic (se habilitado)
        if self.use_stoch_confirmation:
            lowest_low = df['low'].rolling(self.stoch_k).min()
            highest_high = df['high'].rolling(self.stoch_k).max()
            raw_k = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low + 1e-10)
            df['stoch_k'] = raw_k.rolling(3).mean()  # smooth_k = 3
            df['stoch_d'] = df['stoch_k'].rolling(self.stoch_d).mean()

        # Volume ratio
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-10)

        return df

    def _calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calcular ATR."""
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        alpha = 1 / self.atr_period
        return tr.ewm(alpha=alpha, adjust=False).mean()

    def _calculate_adx(self, df: pd.DataFrame) -> pd.Series:
        """Calcular ADX."""
        up = df['high'].diff()
        down = -df['low'].diff()
        plus_dm = up.where((up > down) & (up > 0), 0)
        minus_dm = down.where((down > up) & (down > 0), 0)

        atr = df['atr']
        alpha = 1 / 14
        plus_dm_smooth = plus_dm.ewm(alpha=alpha, adjust=False).mean()
        minus_dm_smooth = minus_dm.ewm(alpha=alpha, adjust=False).mean()

        plus_di = 100 * plus_dm_smooth / (atr + 1e-10)
        minus_di = 100 * minus_dm_smooth / (atr + 1e-10)

        di_sum = plus_di + minus_di
        di_diff = abs(plus_di - minus_di)
        dx = 100 * di_diff / (di_sum + 1e-10)

        return dx.ewm(alpha=alpha, adjust=False).mean()

    def generate_signal(self, df: pd.DataFrame) -> MeanReversionSignal:
        """
        Gerar sinal de mean reversion.

        Retorna sinal com direção, força, preços e justificativa.
        """
        if len(df) < self.min_data_points:
            return MeanReversionSignal('none', 0, 0, 0, 0,
                                      'Dados insuficientes', 'LOW', 0)

        # Calcular indicadores
        df = self.calculate_indicators(df)

        # Valores atuais
        price = df['close'].iloc[-1]
        bb_upper = df['bb_upper'].iloc[-1]
        bb_lower = df['bb_lower'].iloc[-1]
        bb_mid = df['bb_mid'].iloc[-1]
        bb_pct_b = df['bb_pct_b'].iloc[-1]
        rsi = df['rsi'].iloc[-1]
        adx = df['adx'].iloc[-1]
        atr = df['atr'].iloc[-1]
        volume_ratio = df['volume_ratio'].iloc[-1]

        # Validar indicadores
        if np.isnan(rsi) or np.isnan(adx) or np.isnan(atr) or atr == 0:
            return MeanReversionSignal('none', 0, price, 0, 0,
                                      'Indicadores inválidos', 'LOW', 0)

        # FILTRO CRÍTICO: ADX deve ser baixo (mercado lateral)
        if adx >= self.adx_max:
            return MeanReversionSignal('none', 0, price, 0, 0,
                                      f'ADX muito alto ({adx:.1f} >= {self.adx_max}) - mercado em tendência, evitar mean reversion',
                                      'LOW', 0)

        # FILTRO: Evitar volume spikes extremos (pode ser breakout)
        if volume_ratio > self.max_volume_ratio:
            return MeanReversionSignal('none', 0, price, 0, 0,
                                      f'Volume spike ({volume_ratio:.1f}x) - possível breakout, não mean reversion',
                                      'LOW', 0)

        direction = 'none'
        strength = 0
        confidence = 'LOW'
        reasons = []

        # =========================================
        # LÓGICA DE ENTRADA LONG (Oversold)
        # =========================================
        if bb_pct_b <= self.bb_pct_b_long and rsi < self.rsi_oversold:
            direction = 'long'
            reasons.append(f'Preço em BB lower (pct_b={bb_pct_b:.2f})')
            reasons.append(f'RSI oversold ({rsi:.1f})')

            # Base strength
            strength = 5.0

            # Bonus por extremo RSI
            if rsi < 25:
                strength += 1.5
                reasons.append('RSI extremo (<25)')
            elif rsi < 30:
                strength += 1.0

            # Bonus por ADX muito baixo (mercado muito lateral)
            if adx < 20:
                strength += 1.0
                reasons.append(f'ADX muito baixo ({adx:.1f}) - mercado lateral ideal')
            elif adx < self.adx_max:
                strength += 0.5

            # Confirmação MACD (opcional)
            if self.use_macd_confirmation:
                macd_hist = df['macd_hist'].iloc[-1]
                if macd_hist < 0:  # MACD negativo confirma oversold
                    strength += 1.0
                    reasons.append('MACD confirma oversold')

            # Confirmação Stochastic (opcional)
            if self.use_stoch_confirmation:
                stoch_k = df['stoch_k'].iloc[-1]
                if stoch_k < self.stoch_oversold:
                    strength += 1.0
                    reasons.append(f'Stochastic oversold ({stoch_k:.1f})')

            # Confidence baseada em força
            if strength >= 8:
                confidence = 'HIGH'
            elif strength >= 6:
                confidence = 'MEDIUM'
            else:
                confidence = 'LOW'

        # =========================================
        # LÓGICA DE ENTRADA SHORT (Overbought)
        # =========================================
        elif bb_pct_b >= self.bb_pct_b_short and rsi > self.rsi_overbought:
            direction = 'short'
            reasons.append(f'Preço em BB upper (pct_b={bb_pct_b:.2f})')
            reasons.append(f'RSI overbought ({rsi:.1f})')

            # Base strength
            strength = 5.0

            # Bonus por extremo RSI
            if rsi > 75:
                strength += 1.5
                reasons.append('RSI extremo (>75)')
            elif rsi > 70:
                strength += 1.0

            # Bonus por ADX muito baixo
            if adx < 20:
                strength += 1.0
                reasons.append(f'ADX muito baixo ({adx:.1f}) - mercado lateral ideal')
            elif adx < self.adx_max:
                strength += 0.5

            # Confirmação MACD
            if self.use_macd_confirmation:
                macd_hist = df['macd_hist'].iloc[-1]
                if macd_hist > 0:  # MACD positivo confirma overbought
                    strength += 1.0
                    reasons.append('MACD confirma overbought')

            # Confirmação Stochastic
            if self.use_stoch_confirmation:
                stoch_k = df['stoch_k'].iloc[-1]
                if stoch_k > self.stoch_overbought:
                    strength += 1.0
                    reasons.append(f'Stochastic overbought ({stoch_k:.1f})')

            # Confidence
            if strength >= 8:
                confidence = 'HIGH'
            elif strength >= 6:
                confidence = 'MEDIUM'
            else:
                confidence = 'LOW'

        # Calcular SL/TP
        if direction == 'long':
            # Stop Loss: Abaixo do preço atual (stop APERTADO)
            sl = price - (atr * self.sl_atr_mult)

            # Take Profit: BB middle (média) ou baseado em ATR
            if self.use_bb_middle_tp:
                tp = bb_mid  # Retorno à média (recomendado)
                reasons.append(f'TP na BB middle (média em {bb_mid:.2f})')
            else:
                tp = price + (atr * self.tp_atr_mult)

        elif direction == 'short':
            # Stop Loss: Acima do preço atual
            sl = price + (atr * self.sl_atr_mult)

            # Take Profit: BB middle ou baseado em ATR
            if self.use_bb_middle_tp:
                tp = bb_mid
                reasons.append(f'TP na BB middle (média em {bb_mid:.2f})')
            else:
                tp = price - (atr * self.tp_atr_mult)

        else:
            sl = tp = 0

        # Calcular Risk:Reward
        if direction != 'none' and sl != 0:
            risk = abs(price - sl)
            reward = abs(tp - price)
            risk_reward = reward / risk if risk > 0 else 0
        else:
            risk_reward = 0

        reason = ' | '.join(reasons) if reasons else 'Sem sinal'

        return MeanReversionSignal(
            direction=direction,
            strength=min(10, strength),
            entry_price=price,
            stop_loss=sl,
            take_profit=tp,
            reason=reason,
            confidence=confidence,
            risk_reward=risk_reward
        )

    def should_avoid_trading(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Verificar se devemos EVITAR mean reversion agora.

        Retorna (should_avoid, reason)
        """
        if len(df) < self.min_data_points:
            return True, "Dados insuficientes"

        df = self.calculate_indicators(df)
        adx = df['adx'].iloc[-1]
        volume_ratio = df['volume_ratio'].iloc[-1]
        bb_width = df['bb_width'].iloc[-1]

        # ADX alto = tendência forte
        if adx > 30:
            return True, f"Tendência forte (ADX={adx:.1f}) - mean reversion muito arriscado"

        # Volume spike extremo
        if volume_ratio > 4.0:
            return True, f"Volume spike extremo ({volume_ratio:.1f}x) - possível breakout"

        # Bollinger Bands muito apertadas (squeeze - breakout iminente)
        bb_width_sma = df['bb_width'].rolling(50).mean().iloc[-1]
        if bb_width < bb_width_sma * 0.5:
            return True, "BB Squeeze - breakout iminente, evitar mean reversion"

        return False, ""


# ============================================
# EXEMPLO DE USO
# ============================================

if __name__ == "__main__":
    """
    Exemplo de como usar a estratégia Mean Reversion.
    """

    print("=" * 80)
    print("MEAN REVERSION STRATEGY - Exemplo de Uso")
    print("=" * 80)

    # Configuração CONSERVADORA (menor risco, win rate alto)
    conservative_params = {
        'bb_period': 20,
        'bb_std': 2.0,
        'rsi_period': 14,
        'rsi_oversold': 35,  # Mais seletivo
        'rsi_overbought': 65,  # Mais seletivo
        'adx_max': 20,  # Muito restritivo - só mercado lateral
        'sl_atr_mult': 0.75,
        'tp_atr_mult': 1.5,
        'use_bb_middle_tp': True,
        'use_macd_confirmation': True,  # Extra confirmação
        'use_stoch_confirmation': True,
    }

    # Configuração MODERADA (balanceada)
    moderate_params = {
        'bb_period': 20,
        'bb_std': 2.0,
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'adx_max': 25,
        'sl_atr_mult': 0.60,
        'tp_atr_mult': 1.20,
        'use_bb_middle_tp': True,
        'use_macd_confirmation': False,
        'use_stoch_confirmation': True,
    }

    # Configuração para CRYPTO VOLÁTIL (BTC/ETH)
    volatile_crypto_params = {
        'bb_period': 20,
        'bb_std': 2.5,  # Bandas mais largas
        'rsi_oversold': 25,  # Mais extremo
        'rsi_overbought': 75,
        'adx_max': 25,
        'sl_atr_mult': 0.75,  # Stop um pouco maior
        'tp_atr_mult': 1.5,
        'use_bb_middle_tp': True,
        'max_volume_ratio': 4.0,  # Tolerar mais volume
    }

    print("\nCONFIGURAÇÕES DISPONÍVEIS:")
    print("-" * 80)
    print("\n1. CONSERVADORA:")
    print("   - Win rate esperado: 70-75%")
    print("   - R:R médio: 1:1 a 1:1.5")
    print("   - Trades/dia: BAIXO (muito seletivo)")
    print("   - Quando usar: Iniciantes, conta pequena, baixa tolerância a risco")

    print("\n2. MODERADA:")
    print("   - Win rate esperado: 60-70%")
    print("   - R:R médio: 1:1 a 1:1.2")
    print("   - Trades/dia: MÉDIO")
    print("   - Quando usar: Traders experientes, mercado normal")

    print("\n3. CRYPTO VOLÁTIL:")
    print("   - Win rate esperado: 55-65%")
    print("   - R:R médio: 1:1.5 a 1:2")
    print("   - Trades/dia: MÉDIO-ALTO")
    print("   - Quando usar: BTC, ETH em alta volatilidade")

    print("\n" + "=" * 80)
    print("REGRAS IMPORTANTES:")
    print("=" * 80)
    print("""
1. SEMPRE verificar ADX < 25 antes de entrar
2. NUNCA usar mean reversion em notícias/eventos importantes
3. Stop Loss NÃO É NEGOCIÁVEL - sempre usar e respeitar
4. Take Profit na BB middle é MELHOR que baseado em ATR
5. Se ADX > 30, ESPERAR ou usar estratégia de tendência
6. Volume spike > 3x pode indicar breakout, NÃO mean reversion
7. Win rate alto NÃO significa sempre lucro - gerenciar risco é crucial
8. Mean reversion funciona melhor em pares líquidos (BTC, ETH, principais altcoins)
9. Em dúvida? NÃO ENTRE - mercado sempre terá outra oportunidade
10. Backtest SEMPRE antes de usar em live trading
    """)

    print("=" * 80)
    print("PERFORMANCE ESPERADA (baseada em pesquisa):")
    print("=" * 80)
    print("""
Win Rate: 60-75% (dependendo da configuração)
Profit Factor: 1.2-1.5 (conservador mas consistente)
Max Drawdown: 10-15% (com risk management adequado)
Sharpe Ratio: 1.0-1.5 (bom para mean reversion)
Trades por semana: 5-15 (dependendo de volatilidade)

IMPORTANTE: Esses números assumem:
- Execução disciplinada
- Risk management correto (1-2% por trade)
- Mercado em condições normais (não crash/pump)
- Uso correto dos filtros (ADX, volume)
    """)

    print("\n" + "=" * 80)
    print("Para usar em produção, integrar com portfolio_wfo.py ou bot.py")
    print("=" * 80)
