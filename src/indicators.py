import pandas as pd
import ta

class Indicators:
    @staticmethod
    def add_ema(df, window=14):
        indicator = ta.trend.EMAIndicator(close=df['close'], window=window)
        df[f'ema_{window}'] = indicator.ema_indicator()
        return df

    @staticmethod
    def add_rsi(df, window=14):
        indicator = ta.momentum.RSIIndicator(close=df['close'], window=window)
        df[f'rsi_{window}'] = indicator.rsi()
        return df

    @staticmethod
    def add_atr(df, window=14):
        indicator = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=window)
        df[f'atr_{window}'] = indicator.average_true_range()
        return df

    @staticmethod
    def add_adx(df, window=14):
        indicator = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=window)
        df[f'adx_{window}'] = indicator.adx()
        return df

    @staticmethod
    def add_bb(df, window=20, std_dev=2):
        indicator = ta.volatility.BollingerBands(close=df['close'], window=window, window_dev=std_dev)
        df[f'bb_high_{window}'] = indicator.bollinger_hband()
        df[f'bb_low_{window}'] = indicator.bollinger_lband()
        return df

    @staticmethod
    def add_heikin_ashi(df):
        """Calculates Heikin Ashi candles."""
        df_ha = df.copy()
        
        # HA Close = (Open + High + Low + Close) / 4
        df_ha['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        
        # HA Open = (Previous HA Open + Previous HA Close) / 2
        # Initialize first value
        df_ha.at[df_ha.index[0], 'ha_open'] = (df.iloc[0]['open'] + df.iloc[0]['close']) / 2
        
        # Loop for HA Open (Iterative)
        for i in range(1, len(df_ha)):
            df_ha.at[df_ha.index[i], 'ha_open'] = (df_ha.iloc[i-1]['ha_open'] + df_ha.iloc[i-1]['ha_close']) / 2
            
        # HA High = Max(High, HA Open, HA Close)
        df_ha['ha_high'] = df_ha[['high', 'ha_open', 'ha_close']].max(axis=1)
        
        # HA Low = Min(Low, HA Open, HA Close)
        df_ha['ha_low'] = df_ha[['low', 'ha_open', 'ha_close']].min(axis=1)
        
        # Merge back relevant columns
        df['ha_open'] = df_ha['ha_open']
        df['ha_close'] = df_ha['ha_close']
        df['ha_high'] = df_ha['ha_high']
        df['ha_low'] = df_ha['ha_low']
        
        return df
        
    @staticmethod
    def add_all(df):
        """Adds a standard set of indicators."""
        df = Indicators.add_ema(df, window=9)
        df = Indicators.add_ema(df, window=21)
        df = Indicators.add_rsi(df, window=14)
        df = Indicators.add_atr(df, window=14)
        df = Indicators.add_adx(df, window=14)
        df = Indicators.add_bb(df, window=20)
        df = Indicators.add_heikin_ashi(df)
        return df
