import ccxt
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import config
from indicators import Indicators
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

def fetch_training_data(symbol, timeframe, limit=1000):
    """Fetches historical data for training."""
    exchange = ccxt.binance()
    logger.info(f"Fetching {limit} candles for {symbol} {timeframe}...")
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def prepare_features(df):
    """Calculates technical indicators and LAGGED features."""
    # Base Indicators
    Indicators.add_ema(df, 9)
    Indicators.add_ema(df, 21)
    Indicators.add_rsi(df, 14)
    Indicators.add_atr(df, 14)
    Indicators.add_adx(df, 14)
    Indicators.add_bb(df, 20)
    Indicators.add_heikin_ashi(df)
    
    # --- FEATURE ENGINEERING: LAGGED VALUES ---
    # Give the model context of the "past" (t-1, t-2)
    for col in ['rsi_14', 'close', 'volume', 'adx_14']:
        df[f'{col}_lag1'] = df[col].shift(1)
        df[f'{col}_lag2'] = df[col].shift(2)
        
    # Price Change Features
    df['price_change'] = df['close'].pct_change()
    df['price_change_lag1'] = df['price_change'].shift(1)

    # Drop NaN from shifting
    df.dropna(inplace=True)
    
    # Select Features for Model
    feature_cols = [
        'rsi_14', 'atr_14', 'adx_14', 
        'ema_9', 'ema_21', 
        'bb_high_20', 'bb_low_20',
        'ha_close', 'ha_open', 'volume',
        'rsi_14_lag1', 'adx_14_lag1', 'price_change', 'price_change_lag1'
    ]
    
    return df, feature_cols

def create_target(df, lookahead=5, threshold_atr_mult=1.0):
    """
    Creates target labels: 
    1 (Buy) if price increases by > ATR * mult in 'lookahead' candles.
    -1 (Sell) if price decreases by > ATR * mult.
    0 (Hold) otherwise.
    """
    df['target'] = 0
    
    for i in range(len(df) - lookahead):
        current_close = df.iloc[i]['close']
        atr = df.iloc[i]['atr_14']
        future_high = df.iloc[i+1:i+lookahead+1]['high'].max()
        future_low = df.iloc[i+1:i+lookahead+1]['low'].min()
        
        threshold = atr * threshold_atr_mult
        
        if future_high > current_close + threshold:
            df.at[df.index[i], 'target'] = 1
        elif future_low < current_close - threshold:
            df.at[df.index[i], 'target'] = -1
            
    return df

def train_model():
    # 1. Fetch Data
    df = fetch_training_data(config.SYMBOL, config.TIMEFRAME, limit=2000)
    
    # 2. Prepare Features
    df, feature_cols = prepare_features(df)
    
    # 3. Create Target
    df = create_target(df)
    
    # Drop last 'lookahead' rows where target can't be calculated
    df = df.iloc[:-5]
    
    X = df[feature_cols]
    y = df['target']
    
    # 4. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # 5. Train Model (Gradient Boosting for better performance)
    logger.info("Training Gradient Boosting Classifier...")
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # 6. Evaluate
    logger.info("Evaluating Model...")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # 7. Save Model & Feature List (Important to know which features to use during inference)
    joblib.dump({'model': model, 'features': feature_cols}, 'ml_model.pkl')
    logger.info("✅ Model saved to ml_model.pkl")

if __name__ == "__main__":
    train_model()
