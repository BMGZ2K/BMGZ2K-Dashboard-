"""
Machine Learning Engine for Signal Confirmation
XGBoost, Random Forest, and ensemble methods
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import joblib
import os
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from .indicators import calculate_indicators


class MLEngine:
    """
    Machine Learning engine for signal confirmation and filtering.
    """
    
    def __init__(
        self,
        model_type: str = 'xgboost',
        model_path: str = 'models/ml_model.pkl',
        confidence_threshold: float = 0.6,
        lookback_periods: int = 3
    ):
        """
        Initialize ML engine.
        
        Args:
            model_type: 'xgboost', 'random_forest', or 'gradient_boosting'
            model_path: Path to save/load model
            confidence_threshold: Minimum confidence for signal confirmation
            lookback_periods: Periods to look back for lagged features
        """
        self.model_type = model_type
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.lookback_periods = lookback_periods
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        
        self._load_model()
    
    def _create_model(self):
        """Create ML model based on type."""
        if self.model_type == 'xgboost' and HAS_XGBOOST:
            return XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric='mlogloss'
            )
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        else:
            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
    
    def _load_model(self):
        """Load trained model from disk."""
        if os.path.exists(self.model_path):
            try:
                data = joblib.load(self.model_path)
                if isinstance(data, dict):
                    self.model = data.get('model')
                    self.scaler = data.get('scaler', StandardScaler())
                    self.feature_names = data.get('features', [])
                else:
                    self.model = data
                self.is_trained = True
                print(f"ML Model loaded from {self.model_path}")
            except Exception as e:
                print(f"Failed to load ML model: {e}")
    
    def _save_model(self):
        """Save trained model to disk."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        data = {
            'model': self.model,
            'scaler': self.scaler,
            'features': self.feature_names,
            'trained_at': datetime.now().isoformat(),
            'model_type': self.model_type
        }
        
        joblib.dump(data, self.model_path)
        print(f"ML Model saved to {self.model_path}")
    
    def prepare_features(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """
        Prepare feature matrix from OHLCV data.
        
        Args:
            df: OHLCV DataFrame with indicators
            params: Strategy parameters
        
        Returns:
            DataFrame with features
        """
        # Calculate indicators if not present
        if 'rsi' not in df.columns:
            inds = calculate_indicators(df.copy(), params)
            df = inds.get('df', df)
        
        features = pd.DataFrame(index=df.index)
        
        # Core features
        features['rsi'] = df.get('rsi', 50)
        features['adx'] = df.get('adx', 20)
        features['atr'] = df.get('atr', 0)
        features['volume_ratio'] = df.get('volume_ratio', 1)
        features['bb_width'] = df.get('bb_width', 0)
        features['stoch_k'] = df.get('stoch_k', 50)
        features['macd_hist'] = df.get('macd_hist', 0)
        features['momentum'] = df.get('momentum', 0)
        
        # Trend features
        features['ema_cross'] = (df.get('ema_fast', 0) - df.get('ema_slow', 0)) / df['close']
        features['price_vs_ema200'] = (df['close'] - df.get('ema_200', df['close'])) / df['close']
        features['supertrend_dir'] = df.get('supertrend_dir', 0)
        
        # Volatility features
        features['candle_body'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 0.0001)
        features['upper_shadow'] = (df['high'] - df[['close', 'open']].max(axis=1)) / (df['high'] - df['low'] + 0.0001)
        features['lower_shadow'] = (df[['close', 'open']].min(axis=1) - df['low']) / (df['high'] - df['low'] + 0.0001)
        
        # Lagged features
        for lag in range(1, self.lookback_periods + 1):
            features[f'rsi_lag{lag}'] = features['rsi'].shift(lag)
            features[f'adx_lag{lag}'] = features['adx'].shift(lag)
            features[f'volume_ratio_lag{lag}'] = features['volume_ratio'].shift(lag)
        
        # Price changes
        features['price_change'] = df['close'].pct_change()
        features['price_change_lag1'] = features['price_change'].shift(1)
        features['price_change_lag2'] = features['price_change'].shift(2)
        
        # Rolling statistics
        features['rsi_std_10'] = features['rsi'].rolling(10).std()
        features['volume_std_10'] = features['volume_ratio'].rolling(10).std()
        
        # Remove NaN
        features = features.dropna()
        
        return features
    
    def create_labels(self, df: pd.DataFrame, lookahead: int = 5, threshold: float = 0.01) -> pd.Series:
        """
        Create labels for supervised learning.
        
        Args:
            df: OHLCV DataFrame
            lookahead: Periods to look ahead
            threshold: Minimum return threshold
        
        Returns:
            Series with labels: 1 (buy), -1 (sell), 0 (hold)
        """
        future_return = df['close'].shift(-lookahead) / df['close'] - 1
        
        labels = pd.Series(0, index=df.index)
        labels[future_return > threshold] = 1   # Buy signal
        labels[future_return < -threshold] = -1  # Sell signal
        
        return labels
    
    def train(
        self,
        df: pd.DataFrame,
        params: Dict,
        lookahead: int = 5,
        threshold: float = 0.01,
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train ML model on historical data.
        
        Args:
            df: Historical OHLCV data
            params: Strategy parameters
            lookahead: Periods to look ahead for labels
            threshold: Return threshold for labeling
            validation_split: Validation set ratio
        
        Returns:
            Training results and metrics
        """
        print("Preparing training data...")
        
        # Prepare features and labels
        features = self.prepare_features(df.copy(), params)
        labels = self.create_labels(df, lookahead, threshold)
        
        # Align indices
        common_idx = features.index.intersection(labels.dropna().index)
        X = features.loc[common_idx]
        y = labels.loc[common_idx]
        
        # Remove lookahead period from end
        X = X.iloc[:-lookahead]
        y = y.iloc[:-lookahead]
        
        if len(X) < 500:
            return {'success': False, 'error': 'Insufficient training data'}
        
        # Time series split
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Create and train model
        print(f"Training {self.model_type} model...")
        self.model = self._create_model()
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_val_scaled)
        
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_val, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_val, y_pred, average='weighted', zero_division=0)
        }
        
        # Class distribution
        class_dist = y_train.value_counts().to_dict()
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance = dict(zip(self.feature_names, self.model.feature_importances_))
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        else:
            top_features = []
        
        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=tscv, scoring='accuracy')
        
        self.is_trained = True
        self._save_model()
        
        print(f"Training complete. Accuracy: {metrics['accuracy']:.2%}")
        
        return {
            'success': True,
            'metrics': metrics,
            'cv_scores': cv_scores.tolist(),
            'cv_mean': cv_scores.mean(),
            'class_distribution': class_dist,
            'top_features': top_features,
            'training_samples': len(X_train),
            'validation_samples': len(X_val)
        }
    
    def predict(
        self,
        df: pd.DataFrame,
        params: Dict
    ) -> Tuple[int, float]:
        """
        Make prediction on current market state.
        
        Args:
            df: Recent OHLCV data
            params: Strategy parameters
        
        Returns:
            (prediction, confidence): prediction is 1, -1, or 0
        """
        if not self.is_trained or self.model is None:
            return 0, 0.0
        
        try:
            features = self.prepare_features(df.copy(), params)
            
            if len(features) == 0:
                return 0, 0.0
            
            # Get last row
            X = features.iloc[[-1]]
            
            # Ensure correct feature order
            if self.feature_names:
                missing = set(self.feature_names) - set(X.columns)
                for col in missing:
                    X[col] = 0
                X = X[self.feature_names]
            
            # Scale
            X_scaled = self.scaler.transform(X)
            
            # Predict
            prediction = self.model.predict(X_scaled)[0]
            probabilities = self.model.predict_proba(X_scaled)[0]
            
            classes = self.model.classes_
            pred_idx = list(classes).index(prediction)
            confidence = probabilities[pred_idx]
            
            return int(prediction), float(confidence)
            
        except Exception as e:
            print(f"ML prediction error: {e}")
            return 0, 0.0
    
    def confirm_signal(
        self,
        signal_direction: str,
        df: pd.DataFrame,
        params: Dict
    ) -> Tuple[bool, float, str]:
        """
        Confirm or reject a trading signal using ML.
        
        Args:
            signal_direction: 'long' or 'short'
            df: Recent OHLCV data
            params: Strategy parameters
        
        Returns:
            (confirmed, confidence, reason)
        """
        if not self.is_trained:
            return True, 0.5, "ML not trained"
        
        prediction, confidence = self.predict(df, params)
        
        expected = 1 if signal_direction == 'long' else -1
        
        if prediction == expected:
            if confidence >= self.confidence_threshold:
                return True, confidence, f"ML confirmed ({confidence:.2%})"
            else:
                return True, confidence, f"ML weak confirm ({confidence:.2%})"
        elif prediction == 0:
            return True, confidence, f"ML neutral ({confidence:.2%})"
        else:
            if confidence >= 0.7:
                return False, confidence, f"ML rejected ({confidence:.2%})"
            else:
                return True, confidence, f"ML weak reject ({confidence:.2%})"


class DataCollector:
    """
    Collect and store data for ML training.
    """
    
    def __init__(self, data_file: str = 'models/ml_data.csv', buffer_size: int = 100):
        self.data_file = data_file
        self.buffer_size = buffer_size
        self.buffer = []
    
    def log_data(
        self,
        symbol: str,
        indicators: Dict[str, float],
        signal: str,
        outcome: Optional[float] = None
    ):
        """Log data point for training."""
        row = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'signal': signal,
            'outcome': outcome,
            **indicators
        }
        
        self.buffer.append(row)
        
        if len(self.buffer) >= self.buffer_size:
            self.flush()
    
    def flush(self):
        """Write buffer to disk."""
        if not self.buffer:
            return
        
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        
        df = pd.DataFrame(self.buffer)
        
        if os.path.exists(self.data_file):
            df.to_csv(self.data_file, mode='a', header=False, index=False)
        else:
            df.to_csv(self.data_file, index=False)
        
        self.buffer = []
    
    def load_data(self) -> pd.DataFrame:
        """Load collected data."""
        if os.path.exists(self.data_file):
            return pd.read_csv(self.data_file)
        return pd.DataFrame()


def auto_retrain_model(
    data_file: str,
    model_path: str,
    params: Dict,
    min_samples: int = 1000
) -> Dict[str, Any]:
    """
    Automatically retrain ML model with new data.
    
    Args:
        data_file: Path to collected data
        model_path: Path to save model
        params: Strategy parameters
        min_samples: Minimum samples required
    
    Returns:
        Training results
    """
    if not os.path.exists(data_file):
        return {'success': False, 'error': 'No data file found'}
    
    df = pd.read_csv(data_file)
    
    if len(df) < min_samples:
        return {'success': False, 'error': f'Insufficient data: {len(df)} < {min_samples}'}
    
    ml = MLEngine(model_path=model_path)
    
    # Convert collected data to OHLCV format for training
    # This assumes the data file contains necessary indicator columns
    
    result = ml.train(df, params)
    
    return result
