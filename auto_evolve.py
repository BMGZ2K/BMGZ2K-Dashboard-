"""
Sistema de Evolução Automática
==============================
Evolui estratégias, testa via WFO, valida e faz deploy automático.

Fluxo:
1. Carregar baseline atual
2. Gerar variações (mutações)
3. Testar via WFO
4. Se melhor que baseline: salvar como novo baseline
5. Deploy automático no bot
"""
import os
import sys
import json
import random
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from portfolio_wfo import PortfolioBacktester, PortfolioBacktestResult
from core.config import SYMBOLS, get_validated_params, get_optimization_grid
from core.evolution import get_storage


@dataclass
class EvolutionResult:
    """Resultado de uma evolução."""
    generation: int
    params: Dict
    metrics: Dict
    is_improvement: bool
    improvement_pct: float
    fold_details: List[Dict] = None
    symbols: List[str] = None
    timeframe: str = '1h'


class StrategyEvolver:
    """
    Evoluidor de estratégias usando algoritmo genético simplificado.
    
    Processo:
    1. População inicial = baseline + mutações
    2. Avaliar cada indivíduo via backtest
    3. Selecionar os melhores
    4. Cruzar e mutar
    5. Repetir até convergência ou limite
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.backtester = PortfolioBacktester()
        
        # Histórico de evoluções
        self.history: List[EvolutionResult] = []
        self.current_best: Dict = {}
        self.baseline_score: float = 0
    
    def _default_config(self) -> Dict:
        """Configuração usando valores centralizados."""
        # Usar grid de otimização do config
        opt_grid = get_optimization_grid()

        return {
            # Evolução
            'population_size': 10,
            'generations': 5,
            'mutation_rate': 0.25,
            'elite_count': 2,

            # Melhoria mínima para aceitar
            'min_improvement_pct': 3.0,

            # Parâmetros e seus ranges - do config centralizado
            'param_ranges': {
                'sl_atr_mult': (min(opt_grid['sl_atr_mult']), max(opt_grid['sl_atr_mult']), 0.5),
                'tp_atr_mult': (min(opt_grid['tp_atr_mult']), max(opt_grid['tp_atr_mult']), 1.0),
                'rsi_oversold': (min(opt_grid['rsi_oversold']), max(opt_grid['rsi_oversold']), 5),
                'rsi_overbought': (min(opt_grid['rsi_overbought']), max(opt_grid['rsi_overbought']), 5),
                'adx_min': (min(opt_grid['adx_min']), max(opt_grid['adx_min']), 2),
                'max_positions': (5, 15, 1),
                'max_margin_usage': (0.70, 0.90, 0.05),
                'risk_per_trade': (0.01, 0.03, 0.005),
                'max_leverage': (5, 12, 1),
            },

            # Estratégias disponíveis
            'strategies': [
                'stoch_extreme',
                'rsi_extreme',
                'momentum_burst',
                'trend_adx',
                'pullback'
            ],

            # Paths
            'baseline_path': 'state/current_best.json',
            'baselines_dir': 'state/baselines',
            'bot_config_path': 'state/trader_state.json',
        }
    
    def load_baseline(self) -> Dict:
        """Carregar baseline atual."""
        path = self.config['baseline_path']

        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            self.current_best = data.get('best_params', data.get('params', {}))
            self.baseline_score = data.get('metrics', {}).get('score', data.get('metrics', {}).get('wfo_score', 0))
            print(f"Baseline carregado: score = {self.baseline_score:.2f}")
            return data
        else:
            # Baseline padrão - do config centralizado
            self.current_best = get_validated_params()
            self.baseline_score = 0
            print("Usando baseline do config centralizado")
            return {'best_params': self.current_best}
    
    def generate_mutation(self, params: Dict) -> Dict:
        """Gerar mutação de um conjunto de parâmetros."""
        new_params = params.copy()
        ranges = self.config['param_ranges']
        
        # Mutar alguns parâmetros aleatoriamente
        for param, (min_val, max_val, step) in ranges.items():
            if random.random() < self.config['mutation_rate']:
                if isinstance(min_val, int):
                    new_params[param] = random.randint(min_val, max_val)
                else:
                    # Gerar valores no range com step
                    values = np.arange(min_val, max_val + step, step)
                    new_params[param] = float(random.choice(values))
        
        # Possível mudança de estratégia
        if random.random() < self.config['mutation_rate'] * 0.5:
            new_params['strategy'] = random.choice(self.config['strategies'])
        
        return new_params
    
    def crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Cruzar dois conjuntos de parâmetros."""
        child = {}
        
        for key in parent1.keys():
            if random.random() < 0.5:
                child[key] = parent1[key]
            else:
                child[key] = parent2.get(key, parent1[key])
        
        return child
    
    def evaluate_params(
        self,
        params: Dict,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> Tuple[float, Dict]:
        """
        Avaliar um conjunto de parâmetros.
        
        Returns:
            (score, metrics)
        """
        # Buscar dados
        data = self.backtester.fetch_data(
            symbols,
            self.backtester.config['timeframe'],
            start_date,
            end_date
        )
        
        if len(data) < 3:
            return 0, {}
        
        # Executar backtest
        result = self.backtester.run_backtest(data, params)
        
        # Calcular score composto
        # Score combina múltiplas métricas para avaliar qualidade da estratégia
        score = (
            result.sharpe_ratio * 3 +
            result.total_return_pct / 50 -
            result.max_drawdown_pct / 3 +
            result.win_rate / 25 +
            result.profit_factor * 2
        )
        
        metrics = {
            'return_pct': result.total_return_pct,
            'sharpe': result.sharpe_ratio,
            'max_dd': result.max_drawdown_pct,
            'win_rate': result.win_rate,
            'profit_factor': result.profit_factor,
            'total_trades': result.total_trades,
            'score': score
        }
        
        return score, metrics
    
    def evolve(
        self,
        symbols: List[str] = None,
        days_back: int = 90
    ) -> EvolutionResult:
        """
        Executar evolução completa.
        
        Returns:
            Melhor resultado encontrado
        """
        print("\n" + "=" * 60)
        print("EVOLUÇÃO AUTOMÁTICA DE ESTRATÉGIA")
        print("=" * 60)
        
        # Configuração
        symbols = symbols or SYMBOLS[:15]
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Carregar baseline
        self.load_baseline()
        
        print(f"\nPeríodo: {start_date.date()} a {end_date.date()}")
        print(f"Símbolos: {len(symbols)}")
        print(f"Baseline score: {self.baseline_score:.2f}")
        
        # Gerar população inicial
        population = [self.current_best.copy()]
        
        for _ in range(self.config['population_size'] - 1):
            mutant = self.generate_mutation(self.current_best)
            population.append(mutant)
        
        print(f"\nPopulação inicial: {len(population)} indivíduos")
        
        best_score = self.baseline_score
        best_params = self.current_best.copy()
        best_metrics = {}
        
        # Evoluir
        for gen in range(self.config['generations']):
            print(f"\n--- Geração {gen + 1}/{self.config['generations']} ---")
            
            # Avaliar população
            scores = []
            for i, params in enumerate(population):
                score, metrics = self.evaluate_params(
                    params, symbols, start_date, end_date
                )
                scores.append((score, params, metrics))
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    best_metrics = metrics.copy()
                    print(f"  [!] Novo melhor: score = {score:.2f}")
            
            # Ordenar por score
            scores.sort(key=lambda x: x[0], reverse=True)
            
            print(f"  Melhor desta geração: {scores[0][0]:.2f}")
            print(f"  Média: {np.mean([s[0] for s in scores]):.2f}")
            
            # Selecionar elite
            elite = [s[1] for s in scores[:self.config['elite_count']]]
            
            # Nova população
            new_population = elite.copy()
            
            while len(new_population) < self.config['population_size']:
                # Selecionar pais (torneio)
                tournament = random.sample(scores, min(3, len(scores)))
                parent1 = max(tournament, key=lambda x: x[0])[1]
                
                tournament = random.sample(scores, min(3, len(scores)))
                parent2 = max(tournament, key=lambda x: x[0])[1]
                
                # Cruzar
                child = self.crossover(parent1, parent2)
                
                # Mutar
                child = self.generate_mutation(child)
                
                new_population.append(child)
            
            population = new_population
        
        # Calcular melhoria
        if self.baseline_score > 0:
            improvement = (best_score - self.baseline_score) / self.baseline_score * 100
        else:
            improvement = 100 if best_score > 0 else 0
        
        is_improvement = improvement >= self.config['min_improvement_pct']
        
        print("\n" + "=" * 60)
        print("RESULTADO DA EVOLUÇÃO")
        print("=" * 60)
        print(f"Score baseline: {self.baseline_score:.2f}")
        print(f"Score final: {best_score:.2f}")
        print(f"Melhoria: {improvement:.1f}%")
        print(f"É melhoria significativa: {'SIM' if is_improvement else 'NAO'}")
        
        if best_metrics:
            print(f"\nMétricas:")
            print(f"  Retorno: {best_metrics.get('return_pct', 0):.2f}%")
            print(f"  Sharpe: {best_metrics.get('sharpe', 0):.2f}")
            print(f"  Max DD: {best_metrics.get('max_dd', 0):.2f}%")
            print(f"  Win Rate: {best_metrics.get('win_rate', 0):.1f}%")
            print(f"  Trades: {best_metrics.get('total_trades', 0)}")
        
        # Fazer WFO completo no melhor para obter detalhes dos folds
        fold_details = []
        if is_improvement and best_metrics:
            print("\nExecutando WFO completo para validação...")
            wfo_result = self.backtester.run_wfo(
                symbols, start_date, end_date,
                {k: [v] for k, v in best_params.items()}  # Grid com apenas os melhores params
            )
            fold_details = wfo_result.get('fold_details', [])

        result = EvolutionResult(
            generation=self.config['generations'],
            params=best_params,
            metrics=best_metrics,
            is_improvement=is_improvement,
            improvement_pct=improvement,
            fold_details=fold_details,
            symbols=symbols,
            timeframe=self.backtester.config['timeframe']
        )

        # Salvar se for melhoria
        if is_improvement:
            self.save_evolution(result)
            self.deploy_to_bot(best_params)

        return result
    
    def save_evolution(self, result: EvolutionResult):
        """Salvar resultado da evolução."""
        # Salvar como novo baseline
        output = {
            'timestamp': datetime.now().isoformat(),
            'best_params': result.params,
            'metrics': result.metrics,
            'improvement_pct': result.improvement_pct,
            'previous_score': self.baseline_score,
            'fold_details': result.fold_details or [],
            'symbols': result.symbols or [],
            'timeframe': result.timeframe
        }

        # Backup do baseline anterior
        baseline_path = self.config['baseline_path']
        if os.path.exists(baseline_path):
            backup_name = f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            backup_path = os.path.join(self.config['baselines_dir'], backup_name)
            shutil.copy(baseline_path, backup_path)
            print(f"Baseline anterior salvo em: {backup_path}")

        # Salvar novo baseline
        with open(baseline_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        print(f"Novo baseline salvo em: {baseline_path}")

        # Salvar também no histórico
        history_path = os.path.join(
            self.config['baselines_dir'],
            f"evolution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(history_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        # Salvar no sistema de armazenamento de estratégias validadas
        if result.fold_details:
            try:
                storage = get_storage()
                strategy = storage.add_strategy(
                    strategy_type=result.params.get('strategy', 'unknown'),
                    symbols=result.symbols or [],
                    timeframe=result.timeframe,
                    params=result.params,
                    fold_results=result.fold_details,  # fold_results é o nome do parâmetro
                    wfo_score=result.metrics.get('score', 0)
                )
                # Marcar como ativa
                storage.set_active(strategy.id, True)
                print(f"\n[STORAGE] Estratégia validada salva: {strategy.name}")
                print(f"  ID: {strategy.id}")
                print(f"  Robustez: {strategy.robustness_score}%")
                print(f"  Folds: {strategy.num_folds}")
            except Exception as e:
                print(f"Erro salvando no storage: {e}")
    
    def deploy_to_bot(self, params: Dict):
        """Fazer deploy dos parâmetros no bot."""
        bot_config_path = self.config['bot_config_path']
        
        # Carregar config atual do bot
        if os.path.exists(bot_config_path):
            with open(bot_config_path, 'r') as f:
                bot_state = json.load(f)
        else:
            bot_state = {}
        
        # Atualizar parâmetros
        bot_state['params'] = params
        bot_state['params_updated'] = datetime.now().isoformat()
        
        # Salvar
        with open(bot_config_path, 'w') as f:
            json.dump(bot_state, f, indent=2)
        
        print(f"\n[DEPLOY] Parâmetros atualizados no bot!")
        print(f"  Path: {bot_config_path}")
        print(f"  Estratégia: {params.get('strategy', 'N/A')}")
        
        # Criar arquivo de flag para o bot recarregar
        flag_path = 'state/reload_params.flag'
        with open(flag_path, 'w') as f:
            f.write(datetime.now().isoformat())
        
        print(f"  Flag de reload criada: {flag_path}")


def quick_test():
    """Teste rápido da evolução."""
    print("\n" + "=" * 60)
    print("TESTE RÁPIDO DE EVOLUÇÃO")
    print("=" * 60)
    
    evolver = StrategyEvolver({
        'population_size': 5,
        'generations': 2,
        'mutation_rate': 0.3,
        'elite_count': 1,
        'min_improvement_pct': 1.0,
        'param_ranges': {
            'sl_mult': (1.5, 2.5, 0.5),
            'tp_mult': (2.0, 4.0, 0.5),
            'max_positions': (5, 10, 1),
        },
        'strategies': ['stoch_extreme'],
        'baseline_path': 'state/current_best.json',
        'baselines_dir': 'state/baselines',
        'bot_config_path': 'state/trader_state.json',
    })
    
    result = evolver.evolve(
        symbols=SYMBOLS[:5],  # Apenas 5 símbolos para teste
        days_back=120  # 120 dias para ter folds suficientes
    )
    
    print("\n" + "=" * 60)
    print("TESTE CONCLUÍDO")
    print("=" * 60)
    print(f"Melhoria: {result.improvement_pct:.1f}%")
    print(f"Score: {result.metrics.get('score', 0):.2f}")
    
    return result


def full_evolution():
    """Evolução completa."""
    evolver = StrategyEvolver()
    
    result = evolver.evolve(
        symbols=SYMBOLS[:15],
        days_back=90
    )
    
    return result


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evolução automática de estratégias')
    parser.add_argument('--quick', action='store_true', help='Teste rápido')
    parser.add_argument('--full', action='store_true', help='Evolução completa')
    
    args = parser.parse_args()
    
    if args.quick:
        quick_test()
    else:
        full_evolution()
