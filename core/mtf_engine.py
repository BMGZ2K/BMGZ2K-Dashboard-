"""
Multi-Timeframe Scalping Engine
Processa sinais em multiplos timeframes com validacao hierarquica
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import time

logger = logging.getLogger(__name__)


@dataclass
class TimeframeSignal:
    """Sinal de um timeframe especifico"""
    symbol: str
    timeframe: str
    direction: str  # 'long', 'short', 'neutral'
    strength: float
    indicators: Dict[str, Any]
    timestamp: datetime
    is_valid: bool = True


@dataclass
class MTFSignal:
    """Sinal validado por multiplos timeframes"""
    symbol: str
    entry_timeframe: str
    direction: str
    base_strength: float
    mtf_bonus: float
    final_strength: float
    aligned_timeframes: List[str]
    entry_price: float
    stop_loss: float
    take_profit: float
    timeframe_signals: Dict[str, TimeframeSignal] = field(default_factory=dict)
    confidence: float = 0.0


class MTFCache:
    """Cache de dados e indicadores por timeframe"""

    def __init__(self, ttl_config: Dict[str, int]):
        self.ttl_config = ttl_config
        self._cache: Dict[str, Dict] = defaultdict(dict)
        self._timestamps: Dict[str, Dict[str, datetime]] = defaultdict(dict)

    def get(self, symbol: str, timeframe: str, key: str) -> Optional[Any]:
        """Recupera valor do cache se ainda valido"""
        cache_key = f"{symbol}:{timeframe}"
        if cache_key not in self._cache or key not in self._cache[cache_key]:
            return None

        timestamp = self._timestamps[cache_key].get(key)
        if timestamp is None:
            return None

        ttl = self.ttl_config.get(timeframe, 60)
        if (datetime.now() - timestamp).total_seconds() > ttl:
            del self._cache[cache_key][key]
            del self._timestamps[cache_key][key]
            return None

        return self._cache[cache_key][key]

    def set(self, symbol: str, timeframe: str, key: str, value: Any):
        """Armazena valor no cache"""
        cache_key = f"{symbol}:{timeframe}"
        self._cache[cache_key][key] = value
        self._timestamps[cache_key][key] = datetime.now()

    def clear_symbol(self, symbol: str):
        """Limpa cache de um simbolo"""
        keys_to_remove = [k for k in self._cache.keys() if k.startswith(f"{symbol}:")]
        for key in keys_to_remove:
            del self._cache[key]
            if key in self._timestamps:
                del self._timestamps[key]

    def cleanup(self):
        """Remove entradas expiradas"""
        now = datetime.now()
        for cache_key in list(self._cache.keys()):
            parts = cache_key.split(":")
            if len(parts) < 2:
                continue
            timeframe = parts[1]
            ttl = self.ttl_config.get(timeframe, 60)

            for key in list(self._cache[cache_key].keys()):
                timestamp = self._timestamps[cache_key].get(key)
                if timestamp and (now - timestamp).total_seconds() > ttl:
                    del self._cache[cache_key][key]
                    del self._timestamps[cache_key][key]


class MultiTimeframeEngine:
    """
    Engine para processamento multi-timeframe hierarquico

    Hierarquia:
    - HTF (4h): Direcao geral do mercado (filtro)
    - MTF (1h): Confirmacao de movimento
    - LTF (15m): Timing de entrada
    - SCALP (5m): Execucao de alta frequencia
    """

    def __init__(self, config: Dict, data_fetcher, signal_generator):
        self.config = config
        self.data_fetcher = data_fetcher
        self.signal_generator = signal_generator

        # Configuracao de timeframes
        tf_config = config.get('timeframes', {})
        self.htf = tf_config.get('htf', '4h')
        self.mtf = tf_config.get('mtf', '1h')
        self.ltf = tf_config.get('ltf', '15m')
        self.scalp = tf_config.get('scalp', '5m')
        self.active_timeframes = tf_config.get('active_timeframes', ['5m', '15m', '1h'])

        # Validacao MTF
        self.mtf_validation = tf_config.get('mtf_validation', {
            '5m_requires': ['15m', '1h'],
            '15m_requires': ['1h'],
            '1h_requires': []
        })

        # Cache
        cache_ttl = {
            '5m': 60,
            '15m': 180,
            '1h': 600,
            '4h': 1800
        }
        self.cache = MTFCache(cache_ttl)

        # Estado
        self._htf_directions: Dict[str, str] = {}
        self._last_htf_update: Dict[str, datetime] = {}

        # Metricas
        self.stats = {
            'signals_generated': 0,
            'signals_validated': 0,
            'signals_rejected': 0,
            'by_timeframe': defaultdict(int)
        }

        logger.info(f"MTF Engine inicializado com timeframes: {self.active_timeframes}")

    async def get_htf_direction(self, symbol: str) -> str:
        """
        Obtem direcao do HTF (4h) como filtro principal
        Atualiza a cada 30 minutos
        """
        now = datetime.now()
        last_update = self._last_htf_update.get(symbol)

        if last_update and (now - last_update).total_seconds() < 1800:
            return self._htf_directions.get(symbol, 'neutral')

        try:
            # Busca dados HTF
            df = await self.data_fetcher.fetch_ohlcv(symbol, self.htf, limit=100)
            if df is None or len(df) < 50:
                return 'neutral'

            # Calcula EMAs para direcao
            ema_fast = df['close'].ewm(span=9, adjust=False).mean()
            ema_slow = df['close'].ewm(span=21, adjust=False).mean()

            current_price = df['close'].iloc[-1]
            ema_fast_val = ema_fast.iloc[-1]
            ema_slow_val = ema_slow.iloc[-1]

            # Determina direcao
            if ema_fast_val > ema_slow_val and current_price > ema_fast_val:
                direction = 'bullish'
            elif ema_fast_val < ema_slow_val and current_price < ema_fast_val:
                direction = 'bearish'
            else:
                direction = 'neutral'

            self._htf_directions[symbol] = direction
            self._last_htf_update[symbol] = now

            return direction

        except Exception as e:
            logger.error(f"Erro ao obter HTF direction para {symbol}: {e}")
            return 'neutral'

    async def process_timeframe(
        self,
        symbol: str,
        timeframe: str,
        df=None
    ) -> Optional[TimeframeSignal]:
        """
        Processa sinais para um timeframe especifico
        """
        try:
            # Verifica cache
            cached = self.cache.get(symbol, timeframe, 'signal')
            if cached is not None:
                return cached

            # Busca dados se nao fornecido
            if df is None:
                df = await self.data_fetcher.fetch_ohlcv(symbol, timeframe, limit=100)

            if df is None or len(df) < 50:
                return None

            # Gera sinal usando o gerador existente
            signal_result = self.signal_generator.generate_signal(
                symbol=symbol,
                df=df,
                timeframe=timeframe
            )

            if signal_result is None:
                return None

            # Cria TimeframeSignal
            tf_signal = TimeframeSignal(
                symbol=symbol,
                timeframe=timeframe,
                direction=signal_result.get('direction', 'neutral'),
                strength=signal_result.get('strength', 0),
                indicators=signal_result.get('indicators', {}),
                timestamp=datetime.now(),
                is_valid=signal_result.get('strength', 0) >= self.config.get('strategy', {}).get('min_signal_strength', 4)
            )

            # Armazena no cache
            self.cache.set(symbol, timeframe, 'signal', tf_signal)

            return tf_signal

        except Exception as e:
            logger.error(f"Erro processando {symbol} {timeframe}: {e}")
            return None

    async def validate_mtf_signal(
        self,
        entry_signal: TimeframeSignal,
        timeframe_signals: Dict[str, TimeframeSignal]
    ) -> Tuple[bool, float, List[str]]:
        """
        Valida sinal de entrada com timeframes superiores

        Returns:
            (is_valid, bonus_strength, aligned_timeframes)
        """
        entry_tf = entry_signal.timeframe
        required_tfs = self.mtf_validation.get(f"{entry_tf}_requires", [])

        aligned = []
        bonus = 0.0

        for tf in required_tfs:
            tf_signal = timeframe_signals.get(tf)

            if tf_signal is None:
                continue

            # Verifica alinhamento
            if tf_signal.direction == entry_signal.direction:
                aligned.append(tf)
                # Bonus por timeframe alinhado
                bonus += 1.5
            elif tf_signal.direction == 'neutral':
                # Neutro nao bloqueia, mas nao da bonus
                pass
            else:
                # Direcao oposta - reduz strength
                bonus -= 1.0

        # Para scalping (5m) - relaxado para permitir mais sinais
        # if entry_tf == '5m' and len(aligned) < 1:
        #     return False, 0, aligned

        # Valido se nao tem penalidade excessiva
        is_valid = bonus >= -0.5

        return is_valid, max(0, bonus), aligned

    async def process_symbol_mtf(
        self,
        symbol: str,
        dataframes: Dict[str, Any] = None
    ) -> List[MTFSignal]:
        """
        Processa todos os timeframes para um simbolo e retorna sinais validados
        """
        mtf_signals = []

        try:
            # Obtem direcao HTF como filtro
            htf_direction = await self.get_htf_direction(symbol)

            # Processa cada timeframe ativo
            timeframe_signals: Dict[str, TimeframeSignal] = {}

            tasks = []
            for tf in self.active_timeframes:
                df = dataframes.get(tf) if dataframes else None
                tasks.append(self.process_timeframe(symbol, tf, df))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for tf, result in zip(self.active_timeframes, results):
                if isinstance(result, Exception):
                    logger.error(f"Erro em {symbol} {tf}: {result}")
                    continue
                if result is not None:
                    timeframe_signals[tf] = result

            # Valida e gera MTFSignals
            for tf in self.active_timeframes:
                tf_signal = timeframe_signals.get(tf)

                if tf_signal is None or not tf_signal.is_valid:
                    continue

                # Filtra por HTF (desativado - muito restritivo)
                # if htf_direction != 'neutral':
                #     expected_dir = 'long' if htf_direction == 'bullish' else 'short'
                #     if tf_signal.direction != expected_dir:
                #         self.stats['signals_rejected'] += 1
                #         continue

                # Valida com MTF
                is_valid, bonus, aligned = await self.validate_mtf_signal(
                    tf_signal,
                    timeframe_signals
                )

                if not is_valid:
                    self.stats['signals_rejected'] += 1
                    continue

                # Calcula strength final
                final_strength = tf_signal.strength + bonus

                # Cria MTFSignal
                indicators = tf_signal.indicators
                entry_price = indicators.get('close', 0)
                atr = indicators.get('atr', entry_price * 0.02)

                sl_mult = self.config.get('strategy', {}).get('sl_atr_mult', 2.5)
                tp_mult = self.config.get('strategy', {}).get('tp_atr_mult', 4.0)

                if tf_signal.direction == 'long':
                    stop_loss = entry_price - (atr * sl_mult)
                    take_profit = entry_price + (atr * tp_mult)
                else:
                    stop_loss = entry_price + (atr * sl_mult)
                    take_profit = entry_price - (atr * tp_mult)

                mtf_signal = MTFSignal(
                    symbol=symbol,
                    entry_timeframe=tf,
                    direction=tf_signal.direction,
                    base_strength=tf_signal.strength,
                    mtf_bonus=bonus,
                    final_strength=final_strength,
                    aligned_timeframes=aligned,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    timeframe_signals=timeframe_signals,
                    confidence=min(1.0, final_strength / 12.0)
                )

                mtf_signals.append(mtf_signal)
                self.stats['signals_validated'] += 1
                self.stats['by_timeframe'][tf] += 1

            self.stats['signals_generated'] += len(mtf_signals)

        except Exception as e:
            logger.error(f"Erro em process_symbol_mtf {symbol}: {e}")

        return mtf_signals

    async def process_batch(
        self,
        symbols: List[str],
        max_concurrent: int = 10
    ) -> Dict[str, List[MTFSignal]]:
        """
        Processa batch de simbolos com limite de concorrencia
        """
        results: Dict[str, List[MTFSignal]] = {}
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_limit(symbol: str):
            async with semaphore:
                return symbol, await self.process_symbol_mtf(symbol)

        tasks = [process_with_limit(s) for s in symbols]

        for coro in asyncio.as_completed(tasks):
            try:
                symbol, signals = await coro
                if signals:
                    results[symbol] = signals
            except Exception as e:
                logger.error(f"Erro no batch: {e}")

        return results

    def get_best_signal(self, signals: List[MTFSignal]) -> Optional[MTFSignal]:
        """Retorna o melhor sinal baseado em strength e confidence"""
        if not signals:
            return None

        # Ordena por final_strength * confidence
        return max(signals, key=lambda s: s.final_strength * s.confidence)

    def prioritize_signals(
        self,
        all_signals: Dict[str, List[MTFSignal]],
        max_signals: int = 15
    ) -> List[MTFSignal]:
        """
        Prioriza sinais de todos os simbolos
        Retorna os melhores para execucao
        """
        # Flatten todos os sinais
        flat_signals = []
        for symbol, signals in all_signals.items():
            flat_signals.extend(signals)

        if not flat_signals:
            return []

        # Ordena por score composto
        def signal_score(s: MTFSignal) -> float:
            base = s.final_strength * s.confidence
            # Bonus para mais timeframes alinhados
            tf_bonus = len(s.aligned_timeframes) * 0.5
            # Preferencia por timeframes maiores (mais confiaveis)
            tf_priority = {'1h': 1.2, '15m': 1.0, '5m': 0.8}.get(s.entry_timeframe, 1.0)
            return base * tf_priority + tf_bonus

        sorted_signals = sorted(flat_signals, key=signal_score, reverse=True)

        return sorted_signals[:max_signals]

    def get_stats(self) -> Dict:
        """Retorna estatisticas do engine"""
        return {
            **self.stats,
            'htf_directions': dict(self._htf_directions),
            'cache_size': sum(len(v) for v in self.cache._cache.values())
        }

    def cleanup(self):
        """Limpa recursos"""
        self.cache.cleanup()
