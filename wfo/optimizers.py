"""
Optimizers - Otimizacao Genetica e Bayesiana
=============================================
Metodos avancados de otimizacao para encontrar melhores parametros.
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Callable, Optional, Any
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import Config

log = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Resultado de uma otimizacao."""
    best_params: Dict[str, Any]
    best_score: float
    all_results: List[Tuple[Dict, float]]
    generations: int = 0
    trials: int = 0
    optimization_time: float = 0
    convergence_history: List[float] = field(default_factory=list)


# =============================================================================
# PARAMETROS DE OTIMIZACAO
# =============================================================================
# MELHORIA C4: Parameter space otimizado com ranges mais refinados
# Baseado em testes e validação via WFO
PARAM_SPACE = {
    # RSI - ranges mais estreitos baseados em validação
    'rsi_period': {'type': 'int', 'min': 12, 'max': 16, 'step': 1},       # Era 7-21
    'rsi_oversold': {'type': 'int', 'min': 22, 'max': 28, 'step': 2},     # Era 15-35
    'rsi_overbought': {'type': 'int', 'min': 72, 'max': 78, 'step': 2},   # Era 65-85

    # Stochastic - ranges otimizados
    'stoch_k': {'type': 'int', 'min': 12, 'max': 16, 'step': 1},          # Era 7-21
    'stoch_d': {'type': 'int', 'min': 2, 'max': 4, 'step': 1},            # Era 2-5
    'stoch_oversold': {'type': 'int', 'min': 15, 'max': 25, 'step': 5},   # Era 10-30
    'stoch_overbought': {'type': 'int', 'min': 75, 'max': 85, 'step': 5}, # Era 70-90

    # EMA - ranges focados em valores validados
    'ema_fast': {'type': 'int', 'min': 7, 'max': 12, 'step': 1},          # Era 5-15
    'ema_slow': {'type': 'int', 'min': 18, 'max': 26, 'step': 2},         # Era 15-30
    'ema_trend': {'type': 'int', 'min': 45, 'max': 60, 'step': 5},        # Era 40-100

    # ADX - ranges mais precisos
    'adx_period': {'type': 'int', 'min': 12, 'max': 16, 'step': 1},       # Era 10-20
    'adx_min': {'type': 'int', 'min': 18, 'max': 25, 'step': 1},          # Era 15-30
    'adx_strong': {'type': 'int', 'min': 23, 'max': 30, 'step': 1},       # Era 20-35

    # ATR / SL / TP - ranges otimizados para melhor R:R
    'atr_period': {'type': 'int', 'min': 12, 'max': 16, 'step': 1},       # Era 10-20
    'sl_atr_mult': {'type': 'float', 'min': 2.5, 'max': 4.0, 'step': 0.25}, # Era 1.5-4.0
    'tp_atr_mult': {'type': 'float', 'min': 4.0, 'max': 7.0, 'step': 0.5},  # Era 2.0-6.0

    # Signal - ranges refinados
    'min_signal_strength': {'type': 'int', 'min': 4, 'max': 6, 'step': 1}, # Era 3-7
    'stoch_base_strength': {'type': 'float', 'min': 5.5, 'max': 6.5, 'step': 0.5}, # Era 5.0-7.0

    # Bias - neutro por padrão, pequena variação
    'long_bias': {'type': 'float', 'min': 0.95, 'max': 1.1, 'step': 0.05}, # Era 1.0-1.3
    'short_penalty': {'type': 'float', 'min': 0.9, 'max': 1.05, 'step': 0.05}, # Era 0.7-1.0
}

# Espaço de parâmetros original (para referência e compatibilidade)
PARAM_SPACE_WIDE = {
    'rsi_period': {'type': 'int', 'min': 7, 'max': 21, 'step': 1},
    'rsi_oversold': {'type': 'int', 'min': 15, 'max': 35, 'step': 5},
    'rsi_overbought': {'type': 'int', 'min': 65, 'max': 85, 'step': 5},
    'stoch_k': {'type': 'int', 'min': 7, 'max': 21, 'step': 1},
    'stoch_d': {'type': 'int', 'min': 2, 'max': 5, 'step': 1},
    'stoch_oversold': {'type': 'int', 'min': 10, 'max': 30, 'step': 5},
    'stoch_overbought': {'type': 'int', 'min': 70, 'max': 90, 'step': 5},
    'ema_fast': {'type': 'int', 'min': 5, 'max': 15, 'step': 1},
    'ema_slow': {'type': 'int', 'min': 15, 'max': 30, 'step': 1},
    'ema_trend': {'type': 'int', 'min': 40, 'max': 100, 'step': 10},
    'adx_period': {'type': 'int', 'min': 10, 'max': 20, 'step': 1},
    'adx_min': {'type': 'int', 'min': 15, 'max': 30, 'step': 5},
    'adx_strong': {'type': 'int', 'min': 20, 'max': 35, 'step': 5},
    'atr_period': {'type': 'int', 'min': 10, 'max': 20, 'step': 1},
    'sl_atr_mult': {'type': 'float', 'min': 1.5, 'max': 4.0, 'step': 0.5},
    'tp_atr_mult': {'type': 'float', 'min': 2.0, 'max': 6.0, 'step': 0.5},
    'min_signal_strength': {'type': 'int', 'min': 3, 'max': 7, 'step': 1},
    'stoch_base_strength': {'type': 'float', 'min': 5.0, 'max': 7.0, 'step': 0.5},
    'long_bias': {'type': 'float', 'min': 1.0, 'max': 1.3, 'step': 0.05},
    'short_penalty': {'type': 'float', 'min': 0.7, 'max': 1.0, 'step': 0.05},
}


# =============================================================================
# MELHORIA C1: ADAPTIVE FITNESS FUNCTION
# =============================================================================
def detect_market_regime(df: 'pd.DataFrame') -> str:
    """
    Detecta regime de mercado baseado em ADX e volatilidade.

    Returns:
        'trending': Mercado em tendência forte (ADX > 25)
        'ranging': Mercado lateral (ADX < 20)
        'mixed': Mercado misto/transição
    """
    try:
        import pandas as pd

        if df is None or len(df) < 50:
            return 'mixed'

        # Calcular ADX se não existir
        if 'adx' not in df.columns:
            # Simplificado: usar volatilidade como proxy
            close = df['close']
            returns = close.pct_change().dropna()
            volatility = returns.std() * 100

            # Alta volatilidade = trending, baixa = ranging
            if volatility > 3:
                return 'trending'
            elif volatility < 1.5:
                return 'ranging'
            return 'mixed'

        # Usar ADX para determinar regime
        adx_mean = df['adx'].tail(20).mean()

        if adx_mean > 25:
            return 'trending'
        elif adx_mean < 20:
            return 'ranging'
        return 'mixed'

    except Exception:
        return 'mixed'


def calculate_adaptive_fitness(
    result: Dict,
    market_regime: str = 'mixed',
    trades_normalization: int = 50
) -> float:
    """
    MELHORIA C1: Fitness adaptativa baseada no regime de mercado.

    Em mercados trending: Prioriza Profit Factor (capturar tendências)
    Em mercados ranging: Prioriza Win Rate (trades mais frequentes e precisos)
    Em mercados mistos: Balanceado com foco em Sharpe

    Args:
        result: Dict com métricas (sharpe, profit_factor, win_rate, max_drawdown, total_trades)
        market_regime: 'trending', 'ranging', ou 'mixed'
        trades_normalization: Número de trades para normalização

    Returns:
        Float fitness score
    """
    # Extrair métricas do resultado
    sharpe = result.get('sharpe', result.get('sharpe_ratio', 0)) or 0
    profit_factor = result.get('profit_factor', 1) or 1
    win_rate = result.get('win_rate', 0.5) or 0.5
    max_drawdown = result.get('max_drawdown', 0.1) or 0.1
    total_trades = result.get('total_trades', result.get('trades', 0)) or 0

    # Pesos adaptativos por regime
    if market_regime == 'trending':
        weights = {
            'sharpe': 0.25,
            'profit_factor': 0.35,  # Mais peso em PF - capturar tendências
            'win_rate': 0.15,
            'drawdown': 0.15,
            'trades': 0.10
        }
    elif market_regime == 'ranging':
        weights = {
            'sharpe': 0.15,
            'profit_factor': 0.20,
            'win_rate': 0.35,  # Mais peso em WR - precisão em mercado lateral
            'drawdown': 0.20,
            'trades': 0.10
        }
    else:  # mixed
        weights = {
            'sharpe': 0.30,
            'profit_factor': 0.25,
            'win_rate': 0.20,
            'drawdown': 0.15,
            'trades': 0.10
        }

    # Calcular fitness
    # Sharpe: normalizar para 0-1 (assumindo range típico 0-3)
    sharpe_norm = min(1, max(0, sharpe / 3))

    # Profit Factor: (PF - 1) normalizado (PF de 2 = 1.0)
    pf_norm = min(1, max(0, (profit_factor - 1)))

    # Win Rate: já é 0-1
    wr_norm = min(1, max(0, win_rate))

    # Drawdown: inverter (menor DD = melhor)
    dd_norm = 1 - min(1, max(0, max_drawdown))

    # Trades: normalizar
    trades_norm = min(1, total_trades / trades_normalization)

    fitness = (
        sharpe_norm * weights['sharpe'] +
        pf_norm * weights['profit_factor'] +
        wr_norm * weights['win_rate'] +
        dd_norm * weights['drawdown'] +
        trades_norm * weights['trades']
    )

    return fitness


def random_params(param_space: dict = None) -> dict:
    """Gera parametros aleatorios dentro do espaco."""
    param_space = param_space or PARAM_SPACE
    params = {}

    for name, spec in param_space.items():
        if spec['type'] == 'int':
            values = list(range(spec['min'], spec['max'] + 1, spec.get('step', 1)))
            params[name] = random.choice(values)
        elif spec['type'] == 'float':
            steps = int((spec['max'] - spec['min']) / spec.get('step', 0.1)) + 1
            values = [spec['min'] + i * spec.get('step', 0.1) for i in range(steps)]
            params[name] = round(random.choice(values), 4)

    # Adiciona defaults do Config
    params['strategy'] = Config.get('strategy.name', 'stoch_extreme')
    params['name'] = Config.get('strategy.name', 'stoch_extreme')
    params['risk_per_trade'] = Config.get('risk.risk_per_trade', 0.01)
    params['max_positions'] = Config.get('risk.max_positions', 10)

    return params


def mutate_params(params: dict, mutation_rate: float = 0.2, param_space: dict = None) -> dict:
    """Muta parametros com probabilidade mutation_rate."""
    param_space = param_space or PARAM_SPACE
    new_params = params.copy()

    for name, spec in param_space.items():
        if name not in new_params:
            continue

        if random.random() < mutation_rate:
            current = new_params[name]

            if spec['type'] == 'int':
                delta = random.choice([-spec.get('step', 1), spec.get('step', 1)])
                new_val = current + delta
                new_val = max(spec['min'], min(spec['max'], new_val))
                new_params[name] = new_val
            elif spec['type'] == 'float':
                delta = random.choice([-spec.get('step', 0.1), spec.get('step', 0.1)])
                new_val = current + delta
                new_val = max(spec['min'], min(spec['max'], new_val))
                new_params[name] = round(new_val, 4)

    return new_params


def crossover_params(parent1: dict, parent2: dict) -> dict:
    """Crossover de dois conjuntos de parametros."""
    child = {}

    for key in parent1:
        if key in parent2:
            child[key] = random.choice([parent1[key], parent2[key]])
        else:
            child[key] = parent1[key]

    return child


# =============================================================================
# GENETIC OPTIMIZER
# =============================================================================
class GeneticOptimizer:
    """
    Otimizador usando Algoritmo Genetico.

    Features:
    - Selecao por torneio
    - Crossover e mutacao
    - Elitismo
    - Paralelizacao
    """

    def __init__(
        self,
        population_size: int = None,
        generations: int = None,
        mutation_rate: float = None,
        crossover_rate: float = None,
        elite_size: int = None,
        tournament_size: int = None,
        param_space: dict = None,
        n_jobs: int = None
    ):
        # Load from Config with fallback to defaults
        genetic_config = Config.get_section('wfo').get('genetic', {})

        self.population_size = population_size or genetic_config.get('population_size', 50)
        self.generations = generations or genetic_config.get('generations', 30)
        self.mutation_rate = mutation_rate if mutation_rate is not None else genetic_config.get('mutation_rate', 0.2)
        self.crossover_rate = crossover_rate if crossover_rate is not None else genetic_config.get('crossover_rate', 0.7)
        self.elite_size = elite_size or genetic_config.get('elite_size', 5)
        self.tournament_size = tournament_size or genetic_config.get('tournament_size', 3)
        self.param_space = param_space or PARAM_SPACE
        self.n_jobs = n_jobs or genetic_config.get('n_jobs', 4)

        self.population: List[dict] = []
        self.fitness_scores: List[float] = []
        self.best_individual: dict = {}
        self.best_score: float = float('-inf')
        self.convergence_history: List[float] = []

    def _initialize_population(self):
        """Inicializa populacao aleatoria."""
        self.population = [random_params(self.param_space) for _ in range(self.population_size)]
        self.fitness_scores = [float('-inf')] * self.population_size

    def _tournament_selection(self) -> dict:
        """Seleciona individuo por torneio."""
        tournament_idx = random.sample(range(len(self.population)), self.tournament_size)
        tournament_fitness = [(idx, self.fitness_scores[idx]) for idx in tournament_idx]
        winner_idx = max(tournament_fitness, key=lambda x: x[1])[0]
        return self.population[winner_idx].copy()

    def _evaluate_population(self, fitness_func: Callable[[dict], float]):
        """Avalia fitness de toda populacao."""
        for i, individual in enumerate(self.population):
            if self.fitness_scores[i] == float('-inf'):
                try:
                    score = fitness_func(individual)
                    self.fitness_scores[i] = score if not np.isnan(score) else float('-inf')
                except Exception as e:
                    log.warning(f"Erro avaliando individuo: {e}")
                    self.fitness_scores[i] = float('-inf')

                if self.fitness_scores[i] > self.best_score:
                    self.best_score = self.fitness_scores[i]
                    self.best_individual = individual.copy()

    def _create_next_generation(self):
        """Cria proxima geracao."""
        # Ordena por fitness
        sorted_pop = sorted(
            zip(self.population, self.fitness_scores),
            key=lambda x: x[1],
            reverse=True
        )

        new_population = []
        new_scores = []

        # Elitismo - mantém os melhores
        for i in range(self.elite_size):
            new_population.append(sorted_pop[i][0].copy())
            new_scores.append(sorted_pop[i][1])

        # Preenche resto com crossover e mutacao
        while len(new_population) < self.population_size:
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()

            # Crossover
            if random.random() < self.crossover_rate:
                child = crossover_params(parent1, parent2)
            else:
                child = parent1.copy()

            # Mutacao
            child = mutate_params(child, self.mutation_rate, self.param_space)

            new_population.append(child)
            new_scores.append(float('-inf'))  # Nao avaliado ainda

        self.population = new_population
        self.fitness_scores = new_scores

    def optimize(
        self,
        fitness_func: Callable[[dict], float],
        early_stop_generations: int = 10,
        min_improvement: float = 0.001
    ) -> OptimizationResult:
        """
        Executa otimizacao genetica.

        Args:
            fitness_func: Funcao que avalia parametros e retorna score
            early_stop_generations: Para se nao melhorar em N geracoes
            min_improvement: Melhoria minima para resetar early stop

        Returns:
            OptimizationResult
        """
        import time
        start_time = time.time()

        self._initialize_population()
        no_improvement_count = 0
        prev_best = float('-inf')

        all_results = []

        for gen in range(self.generations):
            # Avalia populacao
            self._evaluate_population(fitness_func)

            # Registra resultados
            for ind, score in zip(self.population, self.fitness_scores):
                if score > float('-inf'):
                    all_results.append((ind.copy(), score))

            # Convergence history
            self.convergence_history.append(self.best_score)

            # Log progresso
            avg_fitness = np.mean([s for s in self.fitness_scores if s > float('-inf')] or [0])
            log.info(f"Gen {gen+1}/{self.generations} | Best: {self.best_score:.4f} | Avg: {avg_fitness:.4f}")

            # Early stopping
            if self.best_score - prev_best > min_improvement:
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= early_stop_generations:
                log.info(f"Early stopping at generation {gen+1}")
                break

            prev_best = self.best_score

            # Proxima geracao
            self._create_next_generation()

        optimization_time = time.time() - start_time

        return OptimizationResult(
            best_params=self.best_individual,
            best_score=self.best_score,
            all_results=all_results,
            generations=gen + 1,
            trials=len(all_results),
            optimization_time=optimization_time,
            convergence_history=self.convergence_history
        )


# =============================================================================
# BAYESIAN OPTIMIZER
# =============================================================================
class BayesianOptimizer:
    """
    Otimizador usando Otimizacao Bayesiana.

    Usa Optuna internamente para TPE (Tree-structured Parzen Estimator).
    Mais eficiente que grid search e random search.
    """

    def __init__(
        self,
        n_trials: int = None,
        timeout: int = None,
        param_space: dict = None,
        n_jobs: int = None,
        sampler: str = None,
        pruner: bool = None
    ):
        # Load from Config with fallback to defaults
        bayesian_config = Config.get_section('wfo').get('bayesian', {})

        self.n_trials = n_trials or bayesian_config.get('n_trials', 100)
        self.timeout = timeout
        self.param_space = param_space or PARAM_SPACE
        self.n_jobs = n_jobs or bayesian_config.get('n_jobs', 1)
        self.sampler_type = sampler or bayesian_config.get('sampler', 'tpe')
        self.use_pruner = pruner if pruner is not None else bayesian_config.get('use_pruner', True)

        self.best_params: dict = {}
        self.best_score: float = float('-inf')
        self.all_results: List[Tuple[dict, float]] = []

    def _create_optuna_study(self):
        """Cria estudo Optuna com sampler apropriado."""
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            if self.sampler_type == 'tpe':
                sampler = optuna.samplers.TPESampler(seed=42)
            elif self.sampler_type == 'cmaes':
                sampler = optuna.samplers.CmaEsSampler(seed=42)
            else:
                sampler = optuna.samplers.RandomSampler(seed=42)

            pruner = optuna.pruners.MedianPruner() if self.use_pruner else optuna.pruners.NopPruner()

            return optuna.create_study(
                direction='maximize',
                sampler=sampler,
                pruner=pruner
            )
        except ImportError:
            log.warning("Optuna nao instalado, usando fallback")
            return None

    def _suggest_params(self, trial) -> dict:
        """Sugere parametros usando Optuna trial."""
        params = {}

        for name, spec in self.param_space.items():
            if spec['type'] == 'int':
                params[name] = trial.suggest_int(
                    name, spec['min'], spec['max'], step=spec.get('step', 1)
                )
            elif spec['type'] == 'float':
                params[name] = trial.suggest_float(
                    name, spec['min'], spec['max'], step=spec.get('step', 0.1)
                )

        # Defaults from Config
        params['strategy'] = Config.get('strategy.name', 'stoch_extreme')
        params['name'] = Config.get('strategy.name', 'stoch_extreme')
        params['risk_per_trade'] = Config.get('risk.risk_per_trade', 0.01)
        params['max_positions'] = Config.get('risk.max_positions', 10)

        return params

    def optimize(
        self,
        fitness_func: Callable[[dict], float]
    ) -> OptimizationResult:
        """
        Executa otimizacao Bayesiana.

        Args:
            fitness_func: Funcao que avalia parametros

        Returns:
            OptimizationResult
        """
        import time
        start_time = time.time()

        study = self._create_optuna_study()

        if study is None:
            # Fallback para random search
            return self._random_search_fallback(fitness_func)

        import optuna

        def objective(trial):
            params = self._suggest_params(trial)

            try:
                score = fitness_func(params)
                if np.isnan(score) or np.isinf(score):
                    return float('-inf')

                self.all_results.append((params.copy(), score))

                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params.copy()

                return score

            except Exception as e:
                log.warning(f"Trial failed: {e}")
                return float('-inf')

        # Executa otimizacao
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs,
            show_progress_bar=True
        )

        optimization_time = time.time() - start_time

        # Extrai melhor resultado (com null check)
        best_trial = study.best_trial
        self.best_params = self._suggest_params_from_dict(best_trial.params)
        self.best_score = best_trial.value if best_trial.value is not None else float('-inf')

        # Convergence history
        convergence = [trial.value for trial in study.trials if trial.value is not None]

        return OptimizationResult(
            best_params=self.best_params,
            best_score=self.best_score,
            all_results=self.all_results,
            trials=len(study.trials),
            optimization_time=optimization_time,
            convergence_history=convergence
        )

    def _suggest_params_from_dict(self, params_dict: dict) -> dict:
        """Converte dict de parametros Optuna para formato padrao."""
        params = params_dict.copy()
        params['strategy'] = Config.get('strategy.name', 'stoch_extreme')
        params['name'] = Config.get('strategy.name', 'stoch_extreme')
        params['risk_per_trade'] = Config.get('risk.risk_per_trade', 0.01)
        params['max_positions'] = Config.get('risk.max_positions', 10)
        return params

    def _random_search_fallback(self, fitness_func: Callable) -> OptimizationResult:
        """Fallback para random search se Optuna nao disponivel."""
        import time
        start_time = time.time()

        for i in range(self.n_trials):
            params = random_params(self.param_space)

            try:
                score = fitness_func(params)
                if not np.isnan(score):
                    self.all_results.append((params.copy(), score))

                    if score > self.best_score:
                        self.best_score = score
                        self.best_params = params.copy()

            except Exception as e:
                log.warning(f"Trial {i} failed: {e}")

            if (i + 1) % 10 == 0:
                log.info(f"Trial {i+1}/{self.n_trials} | Best: {self.best_score:.4f}")

        optimization_time = time.time() - start_time

        return OptimizationResult(
            best_params=self.best_params,
            best_score=self.best_score,
            all_results=self.all_results,
            trials=self.n_trials,
            optimization_time=optimization_time,
            convergence_history=[r[1] for r in self.all_results]
        )


# =============================================================================
# HYBRID OPTIMIZER
# =============================================================================
class HybridOptimizer:
    """
    Otimizador hibrido que combina Bayesiano + Genetico.

    1. Fase 1: Bayesiano para explorar espaco
    2. Fase 2: Genetico para refinar melhores regioes
    """

    def __init__(
        self,
        bayesian_trials: int = None,
        genetic_generations: int = None,
        genetic_population: int = None,
        param_space: dict = None
    ):
        # Load from Config with fallback to defaults
        hybrid_config = Config.get_section('wfo').get('hybrid', {})

        self.bayesian_trials = bayesian_trials or hybrid_config.get('bayesian_trials', 50)
        self.genetic_generations = genetic_generations or hybrid_config.get('genetic_generations', 20)
        self.genetic_population = genetic_population or hybrid_config.get('genetic_population', 30)
        self.param_space = param_space or PARAM_SPACE

        # Additional hybrid config
        self._top_results_inject = hybrid_config.get('top_results_inject', 10)
        self._refinement_elite_size = hybrid_config.get('refinement_elite_size', 3)
        self._refinement_mutation_rate = hybrid_config.get('refinement_mutation_rate', 0.1)

    def optimize(self, fitness_func: Callable[[dict], float]) -> OptimizationResult:
        """Executa otimizacao hibrida."""
        import time
        start_time = time.time()

        all_results = []

        # Fase 1: Bayesiano
        log.info("=== Fase 1: Otimizacao Bayesiana ===")
        bayesian = BayesianOptimizer(
            n_trials=self.bayesian_trials,
            param_space=self.param_space
        )
        bayesian_result = bayesian.optimize(fitness_func)
        all_results.extend(bayesian_result.all_results)

        # Pega top N resultados para inicializar genetico
        top_results = sorted(all_results, key=lambda x: x[1], reverse=True)[:self._top_results_inject]
        initial_population = [r[0] for r in top_results]

        # Fase 2: Genetico para refinar
        log.info("=== Fase 2: Refinamento Genetico ===")
        genetic = GeneticOptimizer(
            population_size=self.genetic_population,
            generations=self.genetic_generations,
            param_space=self.param_space,
            elite_size=self._refinement_elite_size,
            mutation_rate=self._refinement_mutation_rate  # Menor mutacao para refinar
        )

        # Injeta melhores resultados na populacao inicial
        genetic._initialize_population()
        for i, params in enumerate(initial_population):
            if i < len(genetic.population):
                genetic.population[i] = params.copy()

        genetic_result = genetic.optimize(fitness_func)
        all_results.extend(genetic_result.all_results)

        # Melhor resultado geral
        best_result = max(all_results, key=lambda x: x[1])

        optimization_time = time.time() - start_time

        return OptimizationResult(
            best_params=best_result[0],
            best_score=best_result[1],
            all_results=all_results,
            generations=genetic_result.generations,
            trials=len(all_results),
            optimization_time=optimization_time,
            convergence_history=bayesian_result.convergence_history + genetic_result.convergence_history
        )
