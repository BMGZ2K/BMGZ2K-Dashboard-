import pandas as pd
import os
import logging
from datetime import datetime

logger = logging.getLogger()

class DataCollector:
    def __init__(self, filename='ml_data.csv'):
        self.filename = filename

    def log_data(self, symbol, df):
        """Logs features for ML training."""
        if df.empty:
            return

        candle = df.iloc[-1]
        # Calculate/Extract features
        features = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'open': candle['open'],
            'high': candle['high'],
            'low': candle['low'],
            'close': candle['close'],
            'volume': candle['volume'],
            'rsi': candle.get('rsi_14', 0),
            'adx': candle.get('adx_14', 0),
            'ema_9': candle.get('ema_9', 0),
            'ema_21': candle.get('ema_21', 0),
            'bb_high': candle.get('bb_high_20', 0),
            'bb_low': candle.get('bb_low_20', 0),
            'atr': candle.get('atr_14', 0)
        }
        
        # Save to CSV
        file_exists = os.path.isfile(self.filename)
        try:
            df_log = pd.DataFrame([features])
            df_log.to_csv(self.filename, mode='a', header=not file_exists, index=False)
            logger.info(f"📝 DataCollector: Logged data for {symbol}")
        except Exception as e:
            logger.error(f"Failed to log ML data: {e}")
