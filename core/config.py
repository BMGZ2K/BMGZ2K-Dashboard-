"""
Configuration Module - Unified Trading System
High-Performance Crypto Trading with WFO Validation

================================================================================
FONTE ÚNICA DE VERDADE - CONFIGURAÇÃO CENTRALIZADA
================================================================================

Como usar:
    from core.config import get_validated_params, get_param

    # Obter todos os parâmetros validados
    params = get_validated_params()

    # Obter parâmetro específico
    adx_period = get_param('adx_period', default=14)

NÃO FAÇA:
    - Hardcode valores nos módulos
    - Importar WFO_VALIDATED_PARAMS diretamente (use get_validated_params())
    - Definir parâmetros em outros arquivos

VERSÃO: 2.0 - Centralização completa
================================================================================
"""
import os
from dotenv import load_dotenv

load_dotenv(override=True)

# =============================================================================
# API CONFIGURATION
# =============================================================================
API_KEY = os.getenv('BINANCE_API_KEY', os.getenv('Binanceapikey', '')).strip()
SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', os.getenv('BinanceSecretkey', '')).strip()
USE_TESTNET = os.getenv('TESTNET', 'True').lower() == 'true'

# =============================================================================
# TRADING UNIVERSE - Top 60 Liquid Futures
# =============================================================================
SYMBOLS = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'DOGE/USDT',
    'XRP/USDT', 'ADA/USDT', 'AVAX/USDT', 'DOT/USDT', 'LINK/USDT',
    'LTC/USDT', 'TRX/USDT', 'UNI/USDT', 'ATOM/USDT', 'NEAR/USDT',
    'APT/USDT', 'FIL/USDT', 'SUI/USDT', 'ARB/USDT', 'OP/USDT',
    'INJ/USDT', 'GRT/USDT', 'ALGO/USDT', 'XLM/USDT',
    'CRV/USDT', 'RUNE/USDT', 'ETC/USDT',
    'AAVE/USDT', 'EOS/USDT', 'QNT/USDT', 'FTM/USDT',
    'DYDX/USDT', 'APE/USDT', 'XTZ/USDT', 'WIF/USDT',
]

# =============================================================================
# RISK MANAGEMENT
# NOTA: Valores aqui são GLOBAIS. Para trading, use WFO_VALIDATED_PARAMS
# =============================================================================
MAX_POSITIONS = 10  # Alinhado com WFO_VALIDATED_PARAMS['max_positions']
LEVERAGE_CAP = 10   # Alinhado com WFO_VALIDATED_PARAMS['max_leverage']
RISK_PER_TRADE = 0.01  # 1% - Alinhado com WFO_VALIDATED_PARAMS['risk_per_trade']
MAX_PORTFOLIO_RISK = 0.20  # 20% max portfolio risk
CIRCUIT_BREAKER_DRAWDOWN = 0.25  # 25% drawdown triggers halt
COOLDOWN_MINUTES = 3

# Position Limits (Diversification)
MAX_LONGS = 10
MAX_SHORTS = 10
MAX_CORRELATED_POSITIONS = 3  # Max positions in same sector

# =============================================================================
# STRATEGY PARAMETERS - VALIDADOS WFO
# FONTE ÚNICA DE VERDADE - Todos os módulos DEVEM importar daqui
# Para alterar qualquer parâmetro, edite APENAS este arquivo
# =============================================================================
WFO_VALIDATED_PARAMS = {
    # === IDENTIFICAÇÃO ===
    'strategy': 'stoch_extreme',

    # === RSI ===
    'rsi_period': 14,
    'rsi_oversold': 25,
    'rsi_overbought': 75,

    # === STOCHASTIC ===
    'stoch_k': 14,
    'stoch_d': 3,
    'stoch_oversold': 20,
    'stoch_overbought': 80,

    # === MÉDIAS MÓVEIS ===
    'ema_fast': 9,
    'ema_slow': 21,
    'ema_trend': 50,

    # === ADX ===
    'adx_period': 14,
    'adx_min': 20,
    'adx_strong': 25,
    'adx_very_strong': 35,

    # === ATR / STOP-LOSS / TAKE-PROFIT ===
    'atr_period': 14,
    'sl_atr_mult': 3.0,
    'tp_atr_mult': 5.0,

    # === GESTÃO DE RISCO ===
    'risk_per_trade': 0.01,
    'volume_mult': 1.5,
    'max_positions': 10,
    'max_margin_usage': 0.8,
    'max_position_pct': 0.15,
    'min_position_pct': 0.05,
    'max_leverage': 10,
    'use_compounding': True,
    'min_score_to_open': 5.0,

    # === MOMENTUM PARAMS ===
    'momentum_adx_min': 22,
    'momentum_min_move': 0.7,
    'momentum_rsi_long_min': 45,
    'momentum_rsi_long_max': 65,
    'momentum_rsi_short_min': 35,
    'momentum_rsi_short_max': 55,

    # === MEAN REVERSION PARAMS ===
    'mr_adx_max': 22,
    'mr_rsi_long': 35,
    'mr_rsi_short': 65,
    'mr_bb_touch_pct': 0.02,

    # === BIAS PARAMS (baseado em performance real) ===
    'long_bias': 1.1,
    'short_penalty': 0.9,

    # === SIGNAL STRENGTH ===
    'min_strength': 5.0,
    'base_strength': 6.0,
    'min_signal_strength': 5,
    'max_signal_strength': 10,
    'stoch_base_strength': 6.0,
    'adx_aggressive': 30,

    # === FILTER PARAMS ===
    'min_data_points': 50,
    'use_volume_filter': True,
    'volume_mult_threshold': 1.2,

    # === BOLLINGER BANDS ===
    'bb_period': 20,
    'bb_std': 2.0,
}

# =============================================================================
# BACKTEST / PORTFOLIO CONFIG
# =============================================================================
BACKTEST_CONFIG = {
    'initial_capital': 10000,
    'maker_fee': 0.0002,
    'taker_fee': 0.0004,
    'slippage': 0.0002,
    'funding_rate': 0.0001,
    'max_drawdown_halt': 0.20,
    'min_score_to_replace': 2.5,
    'timeframe': '1h',
}

# =============================================================================
# WFO VALIDATION CONFIG
# =============================================================================
WFO_VALIDATION_CONFIG = {
    'min_folds': 6,
    'train_days': 30,
    'test_days': 10,
    'total_months': 4,
    'min_trades_per_fold': 3,
    'min_symbols': 5,
    'train_ratio': 0.7,
}

# =============================================================================
# OPTIMIZATION GRID (para auto_evolve e optimize_strategy)
# =============================================================================
OPTIMIZATION_GRID = {
    'sl_atr_mult': [2.0, 2.5, 3.0, 3.5],
    'tp_atr_mult': [3.0, 4.0, 5.0, 6.0],
    'rsi_oversold': [20, 25, 30],
    'rsi_overbought': [70, 75, 80],
    'adx_min': [18, 20, 25],
    'stoch_oversold': [15, 20, 25],
    'stoch_overbought': [75, 80, 85],
}

# Alias para compatibilidade com código legado
DEFAULT_STRATEGY_PARAMS = WFO_VALIDATED_PARAMS.copy()


def get_validated_params() -> dict:
    """
    Retorna cópia dos parâmetros WFO validados.
    Use esta função em vez de acessar WFO_VALIDATED_PARAMS diretamente.
    """
    return WFO_VALIDATED_PARAMS.copy()


def get_param(name: str, default=None):
    """
    Obtém um parâmetro específico dos valores WFO validados.
    """
    return WFO_VALIDATED_PARAMS.get(name, default)


def get_backtest_config() -> dict:
    """Retorna configurações de backtest."""
    config = BACKTEST_CONFIG.copy()
    # Mesclar com parâmetros validados
    config.update({
        'max_positions': WFO_VALIDATED_PARAMS['max_positions'],
        'max_margin_usage': WFO_VALIDATED_PARAMS['max_margin_usage'],
        'max_position_pct': WFO_VALIDATED_PARAMS['max_position_pct'],
        'min_position_pct': WFO_VALIDATED_PARAMS['min_position_pct'],
        'max_leverage': WFO_VALIDATED_PARAMS['max_leverage'],
        'use_compounding': WFO_VALIDATED_PARAMS['use_compounding'],
        'min_score_to_open': WFO_VALIDATED_PARAMS['min_score_to_open'],
    })
    return config


def get_wfo_config() -> dict:
    """Retorna configurações de WFO validation."""
    return WFO_VALIDATION_CONFIG.copy()


def get_optimization_grid() -> dict:
    """Retorna grid de otimização."""
    return OPTIMIZATION_GRID.copy()

# =============================================================================
# OPTIMIZATION PARAMETERS (WFO)
# =============================================================================
WFO_CONFIG = {
    'num_folds': 5,
    'train_ratio': 0.7,
    'min_trades_per_fold': 30,
    'optimization_metric': 'sharpe_ratio',  # sharpe_ratio, sortino_ratio, profit_factor, calmar_ratio
    'min_win_rate': 0.40,
    'min_profit_factor': 1.2,
    'max_drawdown': 0.20,
}

# Parameter Search Space for Optimization
PARAM_GRID = {
    'supertrend_length': [7, 10, 14],
    'supertrend_multiplier': [1.5, 2.0, 2.5, 3.0],
    'rsi_length': [10, 14, 21],
    'rsi_overbought': [65, 70, 75],
    'rsi_oversold': [25, 30, 35],
    'adx_threshold': [15, 20, 25, 30],
    'tp_atr_multiplier': [2.5, 3.0, 3.5, 4.0, 5.0],
    'sl_atr_multiplier': [1.0, 1.5, 2.0],
    'ema_fast': [5, 9, 12],
    'ema_slow': [21, 34, 50],
}

# =============================================================================
# ML CONFIGURATION
# =============================================================================
ML_CONFIG = {
    'enabled': True,
    'model_type': 'xgboost',  # xgboost, random_forest, lightgbm
    'confidence_threshold': 0.6,
    'retrain_interval_hours': 24,
    'min_samples_for_training': 1000,
    'feature_importance_threshold': 0.01,
}

# =============================================================================
# EXECUTION
# =============================================================================
EXECUTION_CONFIG = {
    'order_type': 'market',
    'retry_attempts': 5,
    'retry_delay_base': 0.5,
    'min_notional': 6.0,
    'slippage_tolerance': 0.002,  # 0.2%
}

# =============================================================================
# FILE PATHS
# =============================================================================
LOG_FILE = "logs/trades.csv"
STATE_FILE = "state/bot_state.json"
PARAMS_FILE = "state/optimized_params.json"
HISTORY_FILE = "logs/balance_history.csv"
EVOLUTION_LOG = "logs/evolution.log"
ML_MODEL_FILE = "models/ml_model.pkl"
BACKTEST_RESULTS = "results/backtest_results.json"

# =============================================================================
# TIMEFRAMES
# =============================================================================
PRIMARY_TIMEFRAME = '1h'  # Otimizado: SOL stoch_extreme validado WFO
HTF_TIMEFRAME = '4h'
LTF_TIMEFRAME = '15m'

# Intervalos de polling (segundos) baseados no timeframe
# Verifica a cada X segundos para não perder sinais
POLLING_INTERVALS = {
    '1m': 10,     # 10 segundos
    '3m': 20,     # 20 segundos
    '5m': 30,     # 30 segundos
    '15m': 60,    # 1 minuto
    '30m': 120,   # 2 minutos
    '1h': 60,     # 1 minuto (verificar frequentemente)
    '4h': 300,    # 5 minutos
    '1d': 900,    # 15 minutos
}

def get_polling_interval(timeframe: str) -> int:
    """Retorna intervalo de polling em segundos para um timeframe."""
    return POLLING_INTERVALS.get(timeframe, 60)

# =============================================================================
# METRICS THRESHOLDS (For Strategy Validation)
# =============================================================================
VALIDATION_THRESHOLDS = {
    'min_sharpe': 1.0,
    'min_sortino': 1.2,
    'min_calmar': 0.5,
    'min_profit_factor': 1.3,
    'min_win_rate': 0.40,
    'max_drawdown': 0.25,
    'min_trades': 50,
    'min_expectancy': 0.5,  # $0.50 per $100 risked
}
