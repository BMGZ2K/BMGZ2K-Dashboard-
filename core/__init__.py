# Core Trading System Module
"""
Core Trading System
====================
Módulos principais do sistema de trading evolutivo.

Módulos ativos:
- config: Configurações (API keys, símbolos, parâmetros)
- signals: Gerador de sinais de trading
- scoring: Sistema de pontuação de sinais
- evolution: Storage de estratégias validadas
- binance_fees: Taxas dinâmicas da Binance
- indicators: Indicadores técnicos
- data: Busca de dados OHLCV
- metrics: Métricas de performance
- risk: Gerenciamento de risco
- strategy: Engine de estratégias
- trader: Execução de trades
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
