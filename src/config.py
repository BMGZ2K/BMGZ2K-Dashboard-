import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
API_KEY = os.getenv("BINANCE_API_KEY")
SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")
TESTNET = os.getenv("BINANCE_TESTNET", "True").lower() == "true"

# Trading Config
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT'] # Initial Watchlist
TIMEFRAME = '1m'
LEVERAGE = 20
USDT_AMOUNT = 100.0 # Allocation per trade
MAX_POSITIONS = 5 # Max simultaneous positions for safety  # Starting leverage, can be adjusted
