# Core Trading System Module
"""
Core Trading System
====================
Módulos principais do sistema de trading evolutivo.

Módulos ativos:
- config: Configurações centralizadas (API keys, símbolos, WFO_VALIDATED_PARAMS)
- signals: Gerador de sinais de trading + classe Signal
- strategies_optimized: Estratégias otimizadas (usa config centralizado)
- trader: Execução de trades e gestão de posições
- scoring: Sistema de pontuação de sinais
- evolution: Storage de estratégias validadas
- binance_fees: Taxas dinâmicas da Binance
- utils: Utilitários (save_json_atomic, load_json_safe, etc.)
- data: Busca de dados OHLCV

Módulos arquivados (não utilizados):
- archive/deprecated_core/strategy.py
- archive/deprecated_core/indicators.py
"""

from .config import (
    API_KEY, SECRET_KEY, SYMBOLS, USE_TESTNET,
    MAX_POSITIONS, LEVERAGE_CAP, RISK_PER_TRADE,
    DEFAULT_STRATEGY_PARAMS, WFO_CONFIG, PARAM_GRID
)
from .signals import SignalGenerator, Signal, check_exit_signal
from .scoring import ScoringSystem, SignalScore
from .evolution import StrategyStorage, ValidatedStrategy, get_storage

__all__ = [
    # Config
    'API_KEY', 'SECRET_KEY', 'SYMBOLS', 'USE_TESTNET',
    'MAX_POSITIONS', 'LEVERAGE_CAP', 'RISK_PER_TRADE',
    'DEFAULT_STRATEGY_PARAMS', 'WFO_CONFIG', 'PARAM_GRID',
    # Signals
    'SignalGenerator', 'Signal', 'check_exit_signal',
    # Scoring
    'ScoringSystem', 'SignalScore',
    # Evolution
    'StrategyStorage', 'ValidatedStrategy', 'get_storage',
]
