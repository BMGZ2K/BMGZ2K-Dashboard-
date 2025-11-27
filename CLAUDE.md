# CLAUDE.md - Project Guide for AI Assistants

## Project Overview

This is an **Automated Trading System for Binance Futures** with Walk-Forward Optimization (WFO), genetic strategy evolution, and automated execution. The system is written in Python and includes a Flask-based web dashboard.

## Quick Commands

```bash
# Run tests
python tests/test_system.py         # Full test suite
python tests/test_system.py --quick # Quick tests only

# Start components
python dashboard.py         # Web dashboard at http://localhost:5000
python bot.py               # Live trading bot
python auto_evolve.py       # Strategy evolution system
python auto_evolve.py --quick  # Quick evolution (fewer folds)

# Windows batch scripts
run_dashboard.bat           # Start dashboard only
run_bot.bat                 # Start bot only
```

## Architecture

```
/
├── core/                   # Core trading modules
│   ├── signals.py          # Signal generation and technical indicators
│   ├── trader.py           # Trade execution logic
│   ├── binance_fees.py     # Dynamic Binance fee fetching
│   ├── scoring.py          # Signal scoring system (0-15 scale)
│   ├── evolution.py        # Validated strategy storage
│   ├── config.py           # Configuration (API keys, symbols, params)
│   ├── data.py             # OHLCV data fetching
│   ├── metrics.py          # Performance metrics calculation
│   ├── risk.py             # Risk management
│   └── strategy.py         # Strategy engine
│
├── portfolio_wfo.py        # Main backtester with Walk-Forward Optimization
├── auto_evolve.py          # Genetic evolution system
├── bot.py                  # Live trading bot
├── dashboard.py            # Flask web dashboard
│
├── templates/              # HTML templates
│   └── dashboard.html      # Main dashboard UI
│
├── state/                  # Persistent state (gitignored)
│   ├── trader_state.json   # Current bot state
│   ├── trade_history.json  # Trade history
│   └── validated_strategies.json
│
└── tests/                  # Test suite
    └── test_system.py      # System tests
```

## Key Patterns

### Signal Generation
```python
from core.signals import SignalGenerator, Signal

gen = SignalGenerator({
    'strategy': 'stoch_extreme',  # or: rsi_extremes, trend_following, mean_reversion, momentum_burst
    'ema_fast': 9,
    'ema_slow': 21,
    'sl_atr_mult': 2.0,
    'tp_atr_mult': 3.0,
})

df = gen.prepare_data(df_ohlcv)  # Pre-compute indicators
signal = gen.generate_signal(df, precomputed=True)
# signal.direction: 'long', 'short', 'none'
# signal.strength: 0-10
```

### Backtesting with WFO
```python
from portfolio_wfo import PortfolioBacktester

wfo = PortfolioBacktester()
data = wfo.fetch_data(['BTCUSDT', 'ETHUSDT'], '1h', start, end)
result = wfo.run_backtest(data, params)
# Or: best = wfo.run_wfo(symbols, start, end, param_grid, n_folds=6)
```

### Strategy Storage
```python
from core.evolution import get_storage

storage = get_storage()
best = storage.get_best_strategy()
storage.save_strategy(strategy)
```

## Available Trading Strategies

1. `stoch_extreme` - Stochastic oversold/overbought + EMA cross
2. `rsi_extremes` - RSI oversold/overbought
3. `trend_following` - EMA cross + ADX filter
4. `mean_reversion` - Bollinger Bands bounce
5. `momentum_burst` - Strong momentum moves

## Technical Indicators (in core/signals.py)

All indicators validated against pandas_ta with <0.01 precision:
- RSI (Wilder's smoothing)
- ATR (Wilder's smoothing)
- EMA (Standard EWM)
- ADX (Wilder's smoothing)
- Bollinger Bands (Population std, ddof=0)
- Stochastic (smooth_k=3)
- MACD (Standard)

## Dashboard API Endpoints

| Endpoint | Description |
|----------|-------------|
| `/api/status` | General status (balance, PnL) |
| `/api/positions` | Open positions |
| `/api/strategies` | Validated strategies |
| `/api/strategies/<id>` | Strategy details |
| `/api/trades` | Trade history |
| `/api/metrics` | Performance metrics |
| `/api/logs` | System logs |

## Dependencies

Key packages: `ccxt`, `pandas`, `pandas_ta`, `numpy`, `flask`, `plotly`, `scikit-learn`, `numba`

Install: `pip install -r requirements.txt`

## Environment Variables

Create a `.env` file in project root:
```
BINANCE_API_KEY=your_api_key
BINANCE_SECRET_KEY=your_secret_key
```

## Development Workflow

1. Modify strategy logic in `core/signals.py`
2. Run tests: `python tests/test_system.py`
3. Validate with WFO using `portfolio_wfo.py`
4. If approved, add to evolution in `auto_evolve.py`
5. Bot automatically uses best validated strategy

## Important Notes

- The `state/` directory contains runtime state and is gitignored
- The `archive/` directory contains legacy code - do not use
- Backtest speed: ~5,500-6,500 candles/second
- Signal scores range from 0-15 (higher = better)
- Risk per trade and leverage are configured in `core/config.py`
