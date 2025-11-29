"""
WFO - Walk-Forward Optimization System
=======================================
Sistema robusto de otimizacao de estrategias com validacao WFO.

Componentes:
- backtester: Backtester preciso com custos reais
- optimizers: Otimizadores genetico e bayesiano
- walk_forward: Engine de Walk-Forward Optimization
- metrics: Calculo de metricas robustas
- storage: Armazenamento de estrategias validadas
- validators: Validacao de robustez

Uso:
    from wfo import WFOEngine, run_optimization

    engine = WFOEngine(symbols=['BTC/USDT'], timeframe='1h')
    results = engine.optimize(method='bayesian')
"""

from .backtester import PreciseBacktester
from .optimizers import GeneticOptimizer, BayesianOptimizer
from .walk_forward import WFOEngine
from .metrics import calculate_metrics, RobustnessMetrics
from .storage import StrategyStorage
from .validators import validate_strategy, RobustnessValidator

__all__ = [
    'PreciseBacktester',
    'GeneticOptimizer',
    'BayesianOptimizer',
    'WFOEngine',
    'calculate_metrics',
    'RobustnessMetrics',
    'StrategyStorage',
    'validate_strategy',
    'RobustnessValidator'
]

__version__ = '1.0.0'
