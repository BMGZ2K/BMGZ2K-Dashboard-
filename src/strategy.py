from abc import ABC, abstractmethod
import pandas as pd

class Strategy(ABC):
    @abstractmethod
    def generate_signal(self, df):
        """
        Returns a signal: 'BUY', 'SELL', or 'NEUTRAL'.
        """
        pass

class HybridStrategy(Strategy):
    def __init__(self, ema_short=9, ema_long=21, rsi_period=14, adx_threshold=20, bb_window=20):
        self.ema_short = ema_short
        self.ema_long = ema_long
        self.rsi_period = rsi_period
        self.adx_threshold = adx_threshold
        self.bb_window = bb_window
        self.model = None
        self.last_ml_confidence = 0.0
        self.load_model()

    def load_model(self):
        import joblib
        import os
        try:
            if os.path.exists('ml_model.pkl'):
                data = joblib.load('ml_model.pkl')
                if isinstance(data, dict):
                    self.model = data['model']
                    # We could also load 'features' here to verify
                else:
                    self.model = data
                print("🧠 ML Model Loaded!")
        except Exception as e:
            print(f"⚠️ Failed to load ML model: {e}")

    def generate_signal(self, df, trend_direction='NEUTRAL'):
        if df.empty:
            return 'NEUTRAL'
        
        candle = df.iloc[-1]
        
        # Indicators
        ema_s = candle.get(f'ema_{self.ema_short}', 0)
        ema_l = candle.get(f'ema_{self.ema_long}', 0)
        rsi = candle.get(f'rsi_{self.rsi_period}', 50)
        adx = candle.get(f'adx_14', 0)
        bb_high = candle.get(f'bb_high_{self.bb_window}', 0)
        bb_low = candle.get(f'bb_low_{self.bb_window}', 0)
        close = candle['close']
        
        signal = 'NEUTRAL'
        
        # Logic Switch
        if adx > self.adx_threshold:
            # TREND MODE (Momentum)
            if ema_s > ema_l and rsi < 70:
                signal = 'BUY'
            elif ema_s < ema_l and rsi > 30:
                signal = 'SELL'
        else:
            # RANGE MODE (Mean Reversion / Bollinger Bounce)
            if close <= bb_low and rsi < 30:
                signal = 'BUY'
            elif close >= bb_high and rsi > 70:
                signal = 'SELL'
        
        # MTF Confirmation (Filter)
        if trend_direction != 'NEUTRAL':
            if signal == 'BUY' and trend_direction == 'DOWN':
                return 'NEUTRAL'
            if signal == 'SELL' and trend_direction == 'UP':
                return 'NEUTRAL'
        
        # Heikin Ashi Confirmation
        if 'ha_close' in df.columns:
            ha_close = candle['ha_close']
            ha_open = candle['ha_open']
            
            if signal == 'BUY':
                if ha_close <= ha_open: 
                    return 'NEUTRAL'
            elif signal == 'SELL':
                if ha_close >= ha_open: 
                    return 'NEUTRAL'

        # --- ML CONFIRMATION ---
        if self.model and signal != 'NEUTRAL':
            try:
                # Prepare features for single prediction
                # MUST MATCH TRAINING FEATURES EXACTLY (Order matters!)
                
                # Calculate Lagged Features on the fly (requires history)
                # We need at least 3 rows to get lag1 and lag2
                if len(df) < 3:
                    return signal # Not enough data for ML
                
                # Helper to get value at index
                def get_val(col, idx):
                    return df.iloc[idx][col] if col in df.columns else 0
                
                # Current Index is -1
                idx = -1
                
                # Features:
                # 'rsi_14', 'atr_14', 'adx_14', 'ema_9', 'ema_21', 'bb_high_20', 'bb_low_20',
                # 'ha_close', 'ha_open', 'volume',
                # 'rsi_14_lag1', 'adx_14_lag1', 'price_change', 'price_change_lag1'
                
                features = [
                    candle.get('rsi_14', 50),
                    candle.get('atr_14', 0),
                    candle.get('adx_14', 0),
                    candle.get('ema_9', 0),
                    candle.get('ema_21', 0),
                    candle.get('bb_high_20', 0),
                    candle.get('bb_low_20', 0),
                    candle.get('ha_close', 0),
                    candle.get('ha_open', 0),
                    candle.get('volume', 0),
                    
                    # Lagged Features
                    df.iloc[-2]['rsi_14'], # rsi_lag1
                    df.iloc[-2]['adx_14'] if 'adx_14' in df.columns else 0, # adx_lag1
                    
                    # Price Change
                    (df.iloc[-1]['close'] - df.iloc[-2]['close']) / df.iloc[-2]['close'], # price_change
                    (df.iloc[-2]['close'] - df.iloc[-3]['close']) / df.iloc[-3]['close']  # price_change_lag1
                ]
                
                # Reshape for sklearn (1, -1)
                prediction = self.model.predict([features])[0]
                probability = self.model.predict_proba([features])[0]
                
                classes = self.model.classes_
                pred_index = list(classes).index(prediction)
                confidence = probability[pred_index]
                
                self.last_ml_confidence = confidence
                
                # 1 = Buy, -1 = Sell, 0 = Hold
                if signal == 'BUY' and prediction != 1:
                    print(f"🤖 ML rejected BUY signal (Pred: {prediction}, Conf: {confidence:.2f})")
                    return 'NEUTRAL'
                elif signal == 'SELL' and prediction != -1:
                    print(f"🤖 ML rejected SELL signal (Pred: {prediction}, Conf: {confidence:.2f})")
                    return 'NEUTRAL'
                else:
                    print(f"🤖 ML confirmed {signal} signal! (Conf: {confidence:.2f})")
                    
            except Exception as e:
                print(f"ML Prediction Error: {e}")
                self.last_ml_confidence = 0.0

        return signal

        return signal
