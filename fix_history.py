import json
import os
import shutil
from datetime import datetime

def clean_history():
    filepath = 'state/trade_history.json'
    if not os.path.exists(filepath):
        print("History file not found.")
        return

    # Backup
    shutil.copy(filepath, filepath + '.bak')
    print(f"Backup created at {filepath}.bak")

    with open(filepath, 'r') as f:
        try:
            history = json.load(f)
        except json.JSONDecodeError:
            print("Invalid JSON.")
            return

    print(f"Total trades before cleaning: {len(history)}")

    unique_trades = []
    seen = set()
    
    # Deduplicate based on symbol, entry_time, exit_time
    for trade in history:
        # Create a unique signature
        sig = (
            trade.get('symbol'),
            trade.get('entry_time'),
            trade.get('exit_time'),
            trade.get('side'),
            trade.get('quantity')
        )
        
        if sig not in seen:
            seen.add(sig)
            unique_trades.append(trade)
        else:
            print(f"Removing duplicate: {trade.get('symbol')} {trade.get('entry_time')}")

    # Re-assign IDs to be sequential
    unique_trades.sort(key=lambda x: x.get('exit_time', ''))
    for i, trade in enumerate(unique_trades):
        trade['id'] = i + 1

    print(f"Total trades after cleaning: {len(unique_trades)}")

    with open(filepath, 'w') as f:
        json.dump(unique_trades, f, indent=2)
    print("History saved.")

if __name__ == "__main__":
    clean_history()
