"""
Data Module - Market Data Handling
Fetch and process market data from exchange

VERSÃO: 2.1
- Cache com TTL ativo e limpeza automática
- Logging adequado
- Thread-safe cache com Lock
- TTL dinâmico por timeframe
"""
import pandas as pd
import numpy as np
import logging
import threading
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketData:
    """
    Handle market data fetching and processing.
    Inclui cache com TTL e limpeza automática.
    """

    # TTL dinâmico por timeframe (em segundos)
    TTL_BY_TIMEFRAME = {
        '1m': 30,     # 30s para 1m
        '5m': 120,    # 2 min para 5m
        '15m': 300,   # 5 min para 15m
        '30m': 600,   # 10 min para 30m
        '1h': 1800,   # 30 min para 1h
        '4h': 3600,   # 1h para 4h
        '1d': 7200,   # 2h para 1d
    }

    def __init__(self, exchange):
        self.exchange = exchange
        self.cache = {}
        self.cache_ttl = 60  # default TTL (seconds)
        self._cache_lock = threading.Lock()  # Thread-safe lock
        self._last_cleanup = datetime.now()
        self._cleanup_interval = 300  # Limpar cache a cada 5 minutos

    def _cleanup_cache(self):
        """Remove entradas expiradas do cache (thread-safe)."""
        now = datetime.now()

        # Só limpar a cada _cleanup_interval segundos
        if (now - self._last_cleanup).total_seconds() < self._cleanup_interval:
            return

        with self._cache_lock:
            self._last_cleanup = now
            expired_keys = []

            for key, (cached_time, _, ttl) in self.cache.items():
                age = (now - cached_time).total_seconds()
                if age > ttl * 10:  # Remover após 10x o TTL
                    expired_keys.append(key)

            for key in expired_keys:
                del self.cache[key]

            if expired_keys:
                logger.debug(f"Cache cleanup: {len(expired_keys)} entradas removidas")
    
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = '5m',
        limit: int = 500,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data and return as DataFrame.
        
        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
            limit: Number of candles
            use_cache: Use cached data if available
        
        Returns:
            DataFrame with OHLCV data
        """
        # Limpeza periódica do cache
        self._cleanup_cache()

        cache_key = f"{symbol}_{timeframe}_{limit}"

        # TTL dinâmico por timeframe
        ttl = self.TTL_BY_TIMEFRAME.get(timeframe, self.cache_ttl)

        # Check cache (thread-safe)
        with self._cache_lock:
            if use_cache and cache_key in self.cache:
                cached_time, cached_data, cached_ttl = self.cache[cache_key]
                if (datetime.now() - cached_time).total_seconds() < cached_ttl:
                    return cached_data.copy()

        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

            if not ohlcv:
                return pd.DataFrame()

            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            # Cache result (thread-safe) com TTL dinâmico
            with self._cache_lock:
                self.cache[cache_key] = (datetime.now(), df.copy(), ttl)

            return df
            
        except Exception as e:
            logger.warning(f"Error fetching OHLCV for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_multi_timeframe(
        self,
        symbol: str,
        timeframes: List[str] = ['5m', '1h'],
        limit: int = 500
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple timeframes.
        
        Args:
            symbol: Trading pair
            timeframes: List of timeframes
            limit: Number of candles per timeframe
        
        Returns:
            Dict of timeframe -> DataFrame
        """
        result = {}
        
        for tf in timeframes:
            df = self.fetch_ohlcv(symbol, tf, limit)
            if not df.empty:
                result[tf] = df
        
        return result
    
    def fetch_historical(
        self,
        symbol: str,
        timeframe: str = '5m',
        days: int = 30,
        max_requests: int = 10
    ) -> pd.DataFrame:
        """
        Fetch extended historical data.
        
        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
            days: Number of days to fetch
            max_requests: Max API requests to make
        
        Returns:
            DataFrame with historical data
        """
        all_data = []
        
        # Calculate milliseconds per candle
        tf_minutes = self._timeframe_to_minutes(timeframe)
        ms_per_candle = tf_minutes * 60 * 1000
        candles_per_request = 1000  # Binance limit
        
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = end_time - (days * 24 * 60 * 60 * 1000)
        
        current_end = end_time
        requests = 0
        
        while current_end > start_time and requests < max_requests:
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol, timeframe,
                    limit=candles_per_request,
                    params={'endTime': current_end}
                )
                
                if not ohlcv:
                    break
                
                all_data.extend(ohlcv)
                
                # Move to earlier data
                current_end = ohlcv[0][0] - 1
                requests += 1
                
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                logger.warning(f"Error fetching historical data: {e}")
                break
        
        if not all_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(
            all_data,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes."""
        multipliers = {
            'm': 1,
            'h': 60,
            'd': 1440,
            'w': 10080
        }
        
        unit = timeframe[-1]
        value = int(timeframe[:-1])
        
        return value * multipliers.get(unit, 1)
    
    def get_current_price(self, symbol: str) -> float:
        """Get current price for symbol."""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return float(ticker.get('last', 0))
        except Exception as e:
            logging.debug(f"Erro ao obter preço de {symbol}: {e}")
            return 0.0
    
    def get_orderbook(self, symbol: str, limit: int = 10) -> Dict[str, Any]:
        """Get orderbook for symbol."""
        try:
            ob = self.exchange.fetch_order_book(symbol, limit)
            return {
                'bids': ob.get('bids', [])[:limit],
                'asks': ob.get('asks', [])[:limit],
                'spread': (ob['asks'][0][0] - ob['bids'][0][0]) if ob['asks'] and ob['bids'] else 0
            }
        except Exception as e:
            logging.debug(f"Erro ao obter orderbook de {symbol}: {e}")
            return {'bids': [], 'asks': [], 'spread': 0}


def fetch_ohlcv_multi(
    symbols: List[str],
    timeframe: str = '1h',
    days: int = 180,
    limit: int = 1000,
    exchange: Any = None
) -> Dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data for multiple symbols.

    Args:
        symbols: List of trading pairs
        timeframe: Candle timeframe
        days: Number of days to fetch
        limit: Max candles per request (used as hint)
        exchange: CCXT exchange instance (optional, creates one if None)

    Returns:
        Dict of symbol -> DataFrame with OHLCV data
    """
    # Create exchange if not provided
    if exchange is None:
        try:
            import ccxt
            from core.config import Config

            use_testnet = Config.get('api.use_testnet', True)

            exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',
                    'adjustForTimeDifference': True
                }
            })

            if use_testnet:
                exchange.set_sandbox_mode(True)

        except Exception as e:
            logger.error(f"Failed to create exchange: {e}")
            return {}

    result = {}
    md = MarketData(exchange)

    for symbol in symbols:
        try:
            df = md.fetch_historical(symbol, timeframe, days, max_requests=20)
            if not df.empty:
                result[symbol] = df
                logger.info(f"Fetched {len(df)} candles for {symbol}")
            else:
                logger.warning(f"No data for {symbol}")
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            continue

        time.sleep(0.2)  # Rate limiting

    return result


def download_data(
    exchange,
    symbol: str,
    timeframe: str = '5m',
    days: int = 90,
    output_file: Optional[str] = None
) -> pd.DataFrame:
    """
    Download historical data and optionally save to file.
    
    Args:
        exchange: CCXT exchange instance
        symbol: Trading pair
        timeframe: Candle timeframe
        days: Number of days
        output_file: Optional file path to save data
    
    Returns:
        DataFrame with historical data
    """
    md = MarketData(exchange)
    df = md.fetch_historical(symbol, timeframe, days)
    
    if output_file and not df.empty:
        df.to_csv(output_file, index=False)
        logger.info(f"Data saved to {output_file}: {len(df)} candles")
    
    return df
