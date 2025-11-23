import pandas as pd
from data_collector import DataCollector

def test():
    print("Testing DataCollector...")
    collector = DataCollector()
    
    # Create dummy DF
    data = {
        'open': [100], 'high': [110], 'low': [90], 'close': [105], 'volume': [1000],
        'rsi_14': [50], 'atr_14': [2], 'adx_14': [25],
        'ema_9': [102], 'ema_21': [100],
        'bb_high_20': [110], 'bb_low_20': [90],
        'ha_close': [105], 'ha_open': [100]
    }
    df = pd.DataFrame(data)
    
    collector.log_step(df, 'BUY', 'UP')
    print("Log step called.")
    
    import os
    if os.path.exists('ml_data.csv'):
        print("✅ ml_data.csv created successfully.")
        with open('ml_data.csv', 'r') as f:
            print(f.read())
    else:
        print("❌ ml_data.csv NOT found.")

if __name__ == "__main__":
    test()
