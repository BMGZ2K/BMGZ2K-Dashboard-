"""
Optimizer Module - Genetic Algorithm and Grid Search
Continuous strategy evolution and parameter optimization
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import random
import json
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from .backtester import Backtester, WalkForwardOptimizer
from .metrics import PerformanceMetrics


@dataclass
class Individual:
    """Genetic algorithm individual (parameter set)."""
    params: Dict[str, Any]
    fitness: float = 0.0
    metrics: Dict[str, float] = None
    generation: int = 0


class GeneticOptimizer:
    """
    Genetic Algorithm optimizer for strategy parameters.
    Evolves parameters over multiple generations to find optimal settings.
    """
    
    def __init__(
        self,
        param_ranges: Dict[str, Tuple],
        population_size: int = 50,
        generations: int = 30,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        elite_ratio: float = 0.1,
        optimization_metric: str = 'sharpe_ratio'
    ):
        """
        Initialize genetic optimizer.
        
        Args:
            param_ranges: Dict of param_name -> (min, max, step) or list of values
            population_size: Number of individuals per generation
            generations: Number of generations to evolve
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elite_ratio: Ratio of elite individuals to preserve
            optimization_metric: Metric to optimize
        """
        self.param_ranges = param_ranges
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_ratio = elite_ratio
        self.optimization_metric = optimization_metric
        
        self.population: List[Individual] = []
        self.best_individual: Optional[Individual] = None
        self.history: List[Dict] = []
    
    def evolve(
        self,
        df: pd.DataFrame,
        base_params: Dict[str, Any],
        initial_balance: float = 10000.0,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run genetic evolution.
        
        Args:
            df: Historical data
            base_params: Base parameters to extend
            initial_balance: Starting balance
            verbose: Print progress
        
        Returns:
            Dict with best params and evolution history
        """
        # Initialize population
        self._initialize_population(base_params)
        
        for gen in range(self.generations):
            # Evaluate fitness
            self._evaluate_population(df, base_params, initial_balance)
            
            # Sort by fitness
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            
            # Update best
            if self.best_individual is None or self.population[0].fitness > self.best_individual.fitness:
                self.best_individual = self.population[0]
            
            # Record history
            avg_fitness = np.mean([ind.fitness for ind in self.population])
            best_fitness = self.population[0].fitness
            
            self.history.append({
                'generation': gen,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness,
                'best_params': self.population[0].params.copy()
            })
            
            if verbose:
                print(f"Gen {gen+1}/{self.generations}: Best={best_fitness:.4f}, Avg={avg_fitness:.4f}")
            
            # Check for early stopping
            if self._check_convergence():
                if verbose:
                    print(f"Converged at generation {gen+1}")
                break
            
            # Create next generation
            self._evolve_population(base_params)
        
        return {
            'best_params': {**base_params, **self.best_individual.params},
            'best_fitness': self.best_individual.fitness,
            'best_metrics': self.best_individual.metrics,
            'generations_run': len(self.history),
            'history': self.history
        }
    
    def _initialize_population(self, base_params: Dict):
        """Create initial random population."""
        self.population = []
        
        for _ in range(self.population_size):
            params = {}
            for name, range_spec in self.param_ranges.items():
                if isinstance(range_spec, list):
                    params[name] = random.choice(range_spec)
                else:
                    min_val, max_val, step = range_spec
                    if isinstance(step, float):
                        params[name] = round(random.uniform(min_val, max_val), 2)
                    else:
                        params[name] = random.randrange(min_val, max_val + 1, step)
            
            self.population.append(Individual(params=params))
    
    def _evaluate_population(self, df: pd.DataFrame, base_params: Dict, initial_balance: float):
        """Evaluate fitness of all individuals."""
        for ind in self.population:
            if ind.fitness != 0:
                continue
            
            full_params = {**base_params, **ind.params}
            
            try:
                bt = Backtester(full_params, initial_balance)
                result = bt.run(df)
                
                metrics = result.metrics
                ind.metrics = metrics
                
                # Calculate composite fitness
                sharpe = metrics.get('sharpe_ratio', 0)
                pf = metrics.get('profit_factor', 0)
                wr = metrics.get('win_rate', 0) / 100
                dd = metrics.get('max_drawdown', 1)
                trades = metrics.get('total_trades', 0)
                
                # Penalize low trade count
                if trades < 30:
                    ind.fitness = 0
                    continue
                
                # Penalize high drawdown
                if dd > 0.3:
                    ind.fitness = 0
                    continue
                
                # Composite fitness
                ind.fitness = (
                    sharpe * 0.35 +
                    (pf - 1) * 0.25 +
                    wr * 0.2 +
                    (1 - dd) * 0.2
                )
                
            except Exception as e:
                ind.fitness = 0
    
    def _evolve_population(self, base_params: Dict):
        """Create next generation through selection, crossover, and mutation."""
        new_population = []
        
        # Elite preservation
        elite_count = int(self.population_size * self.elite_ratio)
        elites = self.population[:elite_count]
        new_population.extend(elites)
        
        # Generate rest through crossover and mutation
        while len(new_population) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_select()
            parent2 = self._tournament_select()
            
            # Crossover
            if random.random() < self.crossover_rate:
                child_params = self._crossover(parent1.params, parent2.params)
            else:
                child_params = parent1.params.copy()
            
            # Mutation
            child_params = self._mutate(child_params)
            
            child = Individual(params=child_params, generation=len(self.history))
            new_population.append(child)
        
        self.population = new_population
    
    def _tournament_select(self, tournament_size: int = 3) -> Individual:
        """Select individual through tournament selection."""
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=lambda x: x.fitness)
    
    def _crossover(self, params1: Dict, params2: Dict) -> Dict:
        """Uniform crossover between two parameter sets."""
        child = {}
        for key in params1:
            if random.random() < 0.5:
                child[key] = params1[key]
            else:
                child[key] = params2[key]
        return child
    
    def _mutate(self, params: Dict) -> Dict:
        """Mutate parameters."""
        mutated = params.copy()
        
        for name, value in mutated.items():
            if random.random() < self.mutation_rate:
                range_spec = self.param_ranges.get(name)
                if range_spec is None:
                    continue
                
                if isinstance(range_spec, list):
                    mutated[name] = random.choice(range_spec)
                else:
                    min_val, max_val, step = range_spec
                    if isinstance(step, float):
                        # Gaussian mutation
                        new_val = value + random.gauss(0, (max_val - min_val) * 0.1)
                        mutated[name] = round(max(min_val, min(max_val, new_val)), 2)
                    else:
                        # Step mutation
                        direction = random.choice([-1, 1])
                        new_val = value + direction * step
                        mutated[name] = max(min_val, min(max_val, new_val))
        
        return mutated
    
    def _check_convergence(self, window: int = 5, threshold: float = 0.001) -> bool:
        """Check if evolution has converged."""
        if len(self.history) < window:
            return False
        
        recent = self.history[-window:]
        fitness_values = [h['best_fitness'] for h in recent]
        
        if max(fitness_values) - min(fitness_values) < threshold:
            return True
        
        return False


class EvolutionEngine:
    """
    Continuous evolution engine that runs in the background.
    Periodically reoptimizes parameters based on recent performance.
    """
    
    def __init__(
        self,
        param_ranges: Dict[str, Tuple],
        base_params: Dict[str, Any],
        params_file: str = 'state/optimized_params.json',
        history_file: str = 'logs/evolution_history.csv'
    ):
        self.param_ranges = param_ranges
        self.base_params = base_params
        self.params_file = params_file
        self.history_file = history_file
        
        self.current_params = self._load_params()
    
    def _load_params(self) -> Dict:
        """Load current optimized parameters."""
        if os.path.exists(self.params_file):
            try:
                with open(self.params_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return self.base_params.copy()
    
    def _save_params(self, params: Dict):
        """Save optimized parameters."""
        os.makedirs(os.path.dirname(self.params_file), exist_ok=True)
        with open(self.params_file, 'w') as f:
            json.dump(params, f, indent=2)
    
    def _log_evolution(self, result: Dict):
        """Log evolution result to CSV."""
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        
        file_exists = os.path.exists(self.history_file)
        
        with open(self.history_file, 'a') as f:
            if not file_exists:
                headers = ['timestamp', 'fitness', 'sharpe', 'profit_factor', 'win_rate', 'max_drawdown']
                headers.extend(list(result['best_params'].keys()))
                f.write(','.join(headers) + '\n')
            
            metrics = result.get('best_metrics', {})
            values = [
                datetime.now().isoformat(),
                str(result['best_fitness']),
                str(metrics.get('sharpe_ratio', 0)),
                str(metrics.get('profit_factor', 0)),
                str(metrics.get('win_rate', 0)),
                str(metrics.get('max_drawdown', 0))
            ]
            values.extend([str(v) for v in result['best_params'].values()])
            f.write(','.join(values) + '\n')
    
    def run_evolution_cycle(
        self,
        df: pd.DataFrame,
        initial_balance: float = 10000.0,
        use_wfo: bool = True
    ) -> Dict[str, Any]:
        """
        Run one evolution cycle.
        
        Args:
            df: Historical data
            initial_balance: Starting balance
            use_wfo: Use Walk-Forward Optimization for validation
        
        Returns:
            Evolution results
        """
        print(f"Starting evolution cycle at {datetime.now()}")
        
        if use_wfo:
            # Use WFO for robust optimization
            wfo = WalkForwardOptimizer(
                param_grid=self._ranges_to_grid(),
                initial_balance=initial_balance,
                num_folds=3,
                optimization_metric='sharpe_ratio'
            )
            
            result = wfo.run(df, base_params=self.base_params)
            
            if result['best_params'] and result.get('final_validation', {}).get('passed', False):
                self.current_params = result['best_params']
                self._save_params(self.current_params)
                
                print(f"New champion found! Score: {result['best_score']:.4f}")
                self._log_evolution({
                    'best_params': result['best_params'],
                    'best_fitness': result['best_score'],
                    'best_metrics': result.get('final_validation', {}).get('metrics', {})
                })
                
                return {
                    'success': True,
                    'method': 'WFO',
                    'result': result
                }
        else:
            # Use genetic algorithm
            ga = GeneticOptimizer(
                param_ranges=self.param_ranges,
                population_size=30,
                generations=20,
                optimization_metric='sharpe_ratio'
            )
            
            result = ga.evolve(df, self.base_params, initial_balance)
            
            if result['best_fitness'] > 0:
                self.current_params = result['best_params']
                self._save_params(self.current_params)
                
                print(f"New champion found! Fitness: {result['best_fitness']:.4f}")
                self._log_evolution(result)
                
                return {
                    'success': True,
                    'method': 'GA',
                    'result': result
                }
        
        return {
            'success': False,
            'message': 'No improvement found'
        }
    
    def _ranges_to_grid(self) -> Dict[str, List]:
        """Convert param ranges to grid for WFO."""
        grid = {}
        for name, range_spec in self.param_ranges.items():
            if isinstance(range_spec, list):
                grid[name] = range_spec
            else:
                min_val, max_val, step = range_spec
                if isinstance(step, float):
                    grid[name] = list(np.arange(min_val, max_val + step, step))
                else:
                    grid[name] = list(range(min_val, max_val + 1, step))
        return grid


def optimize_for_symbol(
    symbol: str,
    df: pd.DataFrame,
    base_params: Dict,
    param_grid: Dict[str, List],
    initial_balance: float = 10000.0
) -> Dict[str, Any]:
    """
    Optimize parameters for a specific symbol.
    
    Args:
        symbol: Trading pair
        df: Historical data
        base_params: Base parameters
        param_grid: Parameter search space
        initial_balance: Starting balance
    
    Returns:
        Optimization results
    """
    wfo = WalkForwardOptimizer(
        param_grid=param_grid,
        initial_balance=initial_balance,
        num_folds=3
    )
    
    result = wfo.run(df, base_params=base_params)
    result['symbol'] = symbol
    
    return result


def multi_symbol_optimization(
    symbol_data: Dict[str, pd.DataFrame],
    base_params: Dict,
    param_grid: Dict[str, List],
    max_workers: int = 4
) -> Dict[str, Dict]:
    """
    Optimize parameters for multiple symbols in parallel.
    
    Args:
        symbol_data: Dict of symbol -> DataFrame
        base_params: Base parameters
        param_grid: Parameter search space
        max_workers: Number of parallel workers
    
    Returns:
        Dict of symbol -> optimization results
    """
    results = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                optimize_for_symbol,
                symbol, df, base_params, param_grid
            ): symbol
            for symbol, df in symbol_data.items()
        }
        
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                results[symbol] = future.result()
            except Exception as e:
                results[symbol] = {'error': str(e)}
    
    return results
