"""
Indicators Module - Complete Technical Analysis Suite
Combines best from both projects with optimizations
"""
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, Any, Optional


def calculate_indicators(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate all technical indicators.
    Returns dictionary with current values and adds columns to dataframe.
    """
    if len(df) < 200:
        return _empty_indicators(df)
    
    # Work on a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Heikin Ashi Transformation
    df = _add_heikin_ashi(df)
    
    # Use HA values for smoother signals
    calc_high = df['ha_high']
    calc_low = df['ha_low']
    calc_close = df['ha_close']
    
    # ATR (Always use raw for true volatility)
    atr_len = params.get('atr_length', 14)
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=atr_len)
    
    # RSI
    rsi_len = params.get('rsi_length', 14)
    df['rsi'] = ta.rsi(calc_close, length=rsi_len)
    df['rsi_sma'] = ta.sma(df['rsi'], length=3)  # Smoothed RSI
    
    # ADX
    adx_result = ta.adx(calc_high, calc_low, calc_close, length=14)
    if adx_result is not None:
        adx_col = [c for c in adx_result.columns if c.startswith('ADX')][0]
        dmp_col = [c for c in adx_result.columns if c.startswith('DMP')][0]
        dmn_col = [c for c in adx_result.columns if c.startswith('DMN')][0]
        df['adx'] = adx_result[adx_col]
        df['di_plus'] = adx_result[dmp_col]
        df['di_minus'] = adx_result[dmn_col]
    
    # SuperTrend (Fast)
    st_len = params.get('supertrend_length', 10)
    st_mult = params.get('supertrend_multiplier', 2.0)
    st = ta.supertrend(calc_high, calc_low, calc_close, length=st_len, multiplier=st_mult)
    if st is not None:
        st_dir = [c for c in st.columns if c.startswith('SUPERTd')][0]
        st_val = [c for c in st.columns if c.startswith('SUPERT_') and not c.startswith('SUPERTd')][0]
        df['supertrend'] = st[st_val]
        df['supertrend_dir'] = st[st_dir]
    
    # SuperTrend (Slow - Higher Timeframe Proxy)
    st_slow_len = params.get('supertrend_slow_length', 60)
    st_slow_mult = params.get('supertrend_slow_multiplier', 3.0)
    st_slow = ta.supertrend(calc_high, calc_low, calc_close, length=st_slow_len, multiplier=st_slow_mult)
    if st_slow is not None:
        st_slow_dir = [c for c in st_slow.columns if c.startswith('SUPERTd')][0]
        df['supertrend_slow_dir'] = st_slow[st_slow_dir]
    
    # Bollinger Bands
    bb_len = params.get('bb_length', 20)
    bb_std = params.get('bb_std', 2.0)
    bb = ta.bbands(calc_close, length=bb_len, std=bb_std)
    if bb is not None:
        lower_col = [c for c in bb.columns if c.startswith('BBL')][0]
        upper_col = [c for c in bb.columns if c.startswith('BBU')][0]
        mid_col = [c for c in bb.columns if c.startswith('BBM')][0]
        df['bb_lower'] = bb[lower_col]
        df['bb_upper'] = bb[upper_col]
        df['bb_mid'] = bb[mid_col]
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / calc_close
        df['bb_width_sma'] = ta.sma(df['bb_width'], length=20)
    
    # EMAs
    ema_fast = params.get('ema_fast', 9)
    ema_slow = params.get('ema_slow', 21)
    ema_trend = params.get('ema_trend', 200)
    df['ema_fast'] = ta.ema(calc_close, length=ema_fast)
    df['ema_slow'] = ta.ema(calc_close, length=ema_slow)
    df['ema_200'] = ta.ema(calc_close, length=ema_trend)
    
    # Volume Analysis
    vol_sma_len = params.get('volume_sma_length', 20)
    df['volume_sma'] = ta.sma(df['volume'], length=vol_sma_len)
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # Stochastic RSI
    stoch_k = params.get('stoch_k', 14)
    stoch_d = params.get('stoch_d', 3)
    df = _add_stoch_rsi(df, stoch_k, stoch_d)
    
    # Donchian Channels
    donch_len = params.get('donchian_length', 20)
    df['donchian_high'] = df['high'].rolling(donch_len).max()
    df['donchian_low'] = df['low'].rolling(donch_len).min()
    
    # MACD
    macd = ta.macd(calc_close, fast=12, slow=26, signal=9)
    if macd is not None:
        macd_col = [c for c in macd.columns if 'MACD_' in c and 'MACDh' not in c and 'MACDs' not in c][0]
        signal_col = [c for c in macd.columns if 'MACDs' in c][0]
        hist_col = [c for c in macd.columns if 'MACDh' in c][0]
        df['macd'] = macd[macd_col]
        df['macd_signal'] = macd[signal_col]
        df['macd_hist'] = macd[hist_col]
    
    # Choppiness Index
    df['chop'] = _calculate_chop(df, 14)
    
    # Market Structure
    df['swing_high'] = df['high'].rolling(10).max()
    df['swing_low'] = df['low'].rolling(10).min()
    df['rsi_swing_high'] = df['rsi'].rolling(10).max()
    df['rsi_swing_low'] = df['rsi'].rolling(10).min()
    
    # Volume Price Analysis
    df = _add_vpa(df)
    
    # Price Action
    df['body_size'] = abs(df['close'] - df['open'])
    df['candle_range'] = df['high'] - df['low']
    df['spread_pct'] = df['body_size'] / df['candle_range'].replace(0, 0.0001)
    
    # Momentum
    df['momentum'] = df['close'].pct_change(10) * 100
    df['roc'] = ta.roc(calc_close, length=10)
    
    return _extract_current_values(df, params)


def _add_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """Add Heikin Ashi candles."""
    df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    
    ha_open = [df['open'].iloc[0]]
    ha_close_values = df['ha_close'].values
    
    for i in range(1, len(df)):
        ha_open.append((ha_open[-1] + ha_close_values[i-1]) / 2)
    
    df['ha_open'] = ha_open
    df['ha_high'] = df[['high', 'ha_open', 'ha_close']].max(axis=1)
    df['ha_low'] = df[['low', 'ha_open', 'ha_close']].min(axis=1)
    
    return df


def _add_stoch_rsi(df: pd.DataFrame, k_len: int, d_len: int) -> pd.DataFrame:
    """Calculate Stochastic RSI."""
    if 'rsi' not in df.columns:
        return df
    
    rsi = df['rsi']
    min_rsi = rsi.rolling(k_len).min()
    max_rsi = rsi.rolling(k_len).max()
    
    denom = max_rsi - min_rsi
    denom = denom.replace(0, 0.000001)
    
    stoch = (rsi - min_rsi) / denom
    df['stoch_k'] = stoch.rolling(d_len).mean() * 100
    df['stoch_d'] = df['stoch_k'].rolling(d_len).mean()
    
    return df


def _calculate_chop(df: pd.DataFrame, length: int) -> pd.Series:
    """Calculate Choppiness Index."""
    try:
        tr = ta.true_range(df['high'], df['low'], df['close'])
        sum_atr = tr.rolling(length).sum()
        
        high_n = df['high'].rolling(length).max()
        low_n = df['low'].rolling(length).min()
        range_n = (high_n - low_n).replace(0, 0.0000001)
        
        ratio = sum_atr / range_n
        ratio = ratio.replace(0, 0.0000001)
        
        return 100 * np.log10(ratio) / np.log10(length)
    except:
        return pd.Series([50.0] * len(df), index=df.index)


def _add_vpa(df: pd.DataFrame) -> pd.DataFrame:
    """Volume Price Analysis indicators."""
    df['bullish_volume'] = df['volume'] * (df['close'] > df['open']).astype(int)
    df['bearish_volume'] = df['volume'] * (df['close'] < df['open']).astype(int)
    
    df['obv'] = (df['volume'] * np.sign(df['close'].diff())).cumsum()
    df['obv_sma'] = ta.sma(df['obv'], length=20)
    
    df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    
    return df


def _extract_current_values(df: pd.DataFrame, params: Dict) -> Dict[str, Any]:
    """Extract current indicator values."""
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    
    def safe_get(col, default=0.0):
        val = last.get(col, default)
        return default if pd.isna(val) else val
    
    def safe_get_prev(col, default=0.0):
        val = prev.get(col, default)
        return default if pd.isna(val) else val
    
    return {
        # Price
        'current_price': safe_get('close'),
        'open': safe_get('open'),
        'high': safe_get('high'),
        'low': safe_get('low'),
        
        # ATR
        'atr': safe_get('atr'),
        
        # RSI
        'rsi': safe_get('rsi', 50),
        'rsi_prev': safe_get_prev('rsi', 50),
        'rsi_smooth': safe_get('rsi_sma', 50),
        
        # ADX
        'adx': safe_get('adx'),
        'adx_prev': safe_get_prev('adx'),
        'adx_slope': safe_get('adx') - safe_get_prev('adx'),
        'di_plus': safe_get('di_plus'),
        'di_minus': safe_get('di_minus'),
        
        # SuperTrend
        'supertrend': safe_get('supertrend'),
        'supertrend_dir': safe_get('supertrend_dir'),
        'supertrend_dir_prev': safe_get_prev('supertrend_dir'),
        'supertrend_slow_dir': safe_get('supertrend_slow_dir'),
        'supertrend_slow_dir_prev': safe_get_prev('supertrend_slow_dir'),
        
        # Bollinger Bands
        'bb_lower': safe_get('bb_lower'),
        'bb_upper': safe_get('bb_upper'),
        'bb_mid': safe_get('bb_mid'),
        'bb_width': safe_get('bb_width'),
        'bb_width_sma': safe_get('bb_width_sma'),
        
        # EMAs
        'ema_fast': safe_get('ema_fast'),
        'ema_slow': safe_get('ema_slow'),
        'ema_200': safe_get('ema_200'),
        
        # Volume
        'volume': safe_get('volume'),
        'volume_sma': safe_get('volume_sma'),
        'volume_ratio': safe_get('volume_ratio', 1.0),
        
        # Stochastic
        'stoch_k': safe_get('stoch_k', 50),
        'stoch_d': safe_get('stoch_d', 50),
        'stoch_k_prev': safe_get_prev('stoch_k', 50),
        'stoch_d_prev': safe_get_prev('stoch_d', 50),
        
        # Donchian
        'donchian_high': safe_get('donchian_high'),
        'donchian_low': safe_get('donchian_low'),
        
        # MACD
        'macd': safe_get('macd'),
        'macd_signal': safe_get('macd_signal'),
        'macd_hist': safe_get('macd_hist'),
        'macd_hist_prev': safe_get_prev('macd_hist'),
        
        # Choppiness
        'chop': safe_get('chop', 50),
        
        # Market Structure
        'swing_high': safe_get('swing_high'),
        'swing_low': safe_get('swing_low'),
        
        # Heikin Ashi
        'ha_close': safe_get('ha_close'),
        'ha_open': safe_get('ha_open'),
        'ha_bullish': safe_get('ha_close') > safe_get('ha_open'),
        
        # VPA
        'spread_pct': safe_get('spread_pct'),
        'obv': safe_get('obv'),
        'obv_sma': safe_get('obv_sma'),
        'vwap': safe_get('vwap'),
        
        # Momentum
        'momentum': safe_get('momentum'),
        'roc': safe_get('roc'),
        
        # DataFrame reference
        'df': df
    }


def _empty_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    """Return empty indicators when insufficient data."""
    price = df.iloc[-1]['close'] if len(df) > 0 else 0
    return {
        'current_price': price,
        'atr': 0, 'rsi': 50, 'adx': 0,
        'supertrend_dir': 0, 'supertrend_slow_dir': 0,
        'bb_lower': price, 'bb_upper': price, 'bb_mid': price,
        'ema_fast': price, 'ema_slow': price, 'ema_200': price,
        'volume_ratio': 1, 'stoch_k': 50, 'stoch_d': 50,
        'chop': 50, 'df': df
    }


def calculate_htf_trend(df_htf: pd.DataFrame, params: Dict) -> int:
    """
    Calculate higher timeframe trend.
    Returns: 1 (bullish), -1 (bearish), 0 (neutral)
    """
    if len(df_htf) < 60:
        return 0
    
    df_htf['ema_21'] = ta.ema(df_htf['close'], length=21)
    df_htf['ema_50'] = ta.ema(df_htf['close'], length=50)
    
    last = df_htf.iloc[-1]
    
    ema_21 = last['ema_21']
    ema_50 = last['ema_50']
    
    if pd.isna(ema_21) or pd.isna(ema_50):
        return 0
    
    if ema_21 > ema_50:
        return 1
    elif ema_21 < ema_50:
        return -1
    return 0
