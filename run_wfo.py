#!/usr/bin/env python3
"""
WFO Runner - Walk-Forward Optimization System
==============================================
Script principal para executar otimizacao de estrategias com validacao WFO.

Uso:
    python run_wfo.py --method hybrid --symbols BTC/USDT,ETH/USDT --timeframe 1h

Metodos disponiveis:
    - genetic: Algoritmo genetico (rapido, bom para exploracao)
    - bayesian: Otimizacao Bayesiana (eficiente, bom para refinamento)
    - hybrid: Combina Bayesiano + Genetico (melhor resultado geral)

Exemplo completo:
    python run_wfo.py \\
        --method hybrid \\
        --symbols BTC/USDT,ETH/USDT,SOL/USDT \\
        --timeframe 1h \\
        --windows 5 \\
        --is-ratio 0.7 \\
        --population 100 \\
        --generations 50 \\
        --bayesian-trials 200 \\
        --export
"""

import argparse
import sys
import json
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Any

import numpy as np
import pandas as pd

# Adiciona path do projeto
sys.path.insert(0, str(Path(__file__).parent))

from wfo import (
    WFOEngine,
    PreciseBacktester,
    GeneticOptimizer,
    BayesianOptimizer,
    StrategyStorage,
    calculate_metrics,
    RobustnessMetrics,
    validate_strategy,
    RobustnessValidator
)
from wfo.metrics import create_strategy_metrics
from wfo.optimizers import HybridOptimizer

from core.config import Config
from core.signals import SignalGenerator


def parse_args():
    """Parse argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description="WFO - Walk-Forward Optimization System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Metodo de otimizacao
    parser.add_argument(
        '--method',
        type=str,
        default='hybrid',
        choices=['genetic', 'bayesian', 'hybrid'],
        help='Metodo de otimizacao (default: hybrid)'
    )

    # Simbolos
    parser.add_argument(
        '--symbols',
        type=str,
        default=None,
        help='Simbolos separados por virgula (ex: BTC/USDT,ETH/USDT)'
    )

    # Timeframe
    parser.add_argument(
        '--timeframe',
        type=str,
        default='1h',
        help='Timeframe para otimizacao (default: 1h)'
    )

    # WFO Config
    parser.add_argument(
        '--windows',
        type=int,
        default=5,
        help='Numero de janelas WFO (default: 5)'
    )

    parser.add_argument(
        '--is-ratio',
        type=float,
        default=0.7,
        help='Ratio In-Sample/Total (default: 0.7)'
    )

    # Genetic Config
    parser.add_argument(
        '--population',
        type=int,
        default=100,
        help='Tamanho da populacao genetica (default: 100)'
    )

    parser.add_argument(
        '--generations',
        type=int,
        default=50,
        help='Numero de geracoes geneticas (default: 50)'
    )

    # Bayesian Config
    parser.add_argument(
        '--bayesian-trials',
        type=int,
        default=200,
        help='Numero de trials Bayesianos (default: 200)'
    )

    # Output
    parser.add_argument(
        '--export',
        action='store_true',
        help='Exporta melhor estrategia para producao'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='config/best_strategy.json',
        help='Arquivo de saida para exportacao'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Modo verbose com mais detalhes'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Executa sem salvar resultados'
    )

    return parser.parse_args()


def get_symbols(args_symbols: Optional[str]) -> List[str]:
    """Obtem lista de simbolos."""
    if args_symbols:
        return [s.strip() for s in args_symbols.split(',')]

    # Usa simbolos do config
    config = Config.get_all()
    return config.get('symbols', ['BTC/USDT', 'ETH/USDT'])


def print_header():
    """Imprime header do sistema."""
    print("\n" + "=" * 70)
    print(" WFO - WALK-FORWARD OPTIMIZATION SYSTEM")
    print(" Sistema Robusto de Treinamento de Estrategias")
    print("=" * 70)
    print(f" Iniciado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")


def print_config(args, symbols: List[str]):
    """Imprime configuracao atual."""
    print("CONFIGURACAO:")
    print("-" * 50)
    print(f"  Metodo:           {args.method.upper()}")
    print(f"  Simbolos:         {', '.join(symbols)}")
    print(f"  Timeframe:        {args.timeframe}")
    print(f"  Janelas WFO:      {args.windows}")
    print(f"  IS Ratio:         {args.is_ratio * 100:.0f}%")

    if args.method in ['genetic', 'hybrid']:
        print(f"  Populacao:        {args.population}")
        print(f"  Geracoes:         {args.generations}")

    if args.method in ['bayesian', 'hybrid']:
        print(f"  Trials Bayesian:  {args.bayesian_trials}")

    print("-" * 50 + "\n")


def print_results(result: Dict, verbose: bool = False):
    """Imprime resultados da otimizacao."""
    print("\n" + "=" * 70)
    print(" RESULTADOS DA OTIMIZACAO")
    print("=" * 70)

    # Resultado geral
    is_robust = result.get('robustness', {}).get('is_robust', False)
    status = "ROBUSTA" if is_robust else "NAO ROBUSTA"
    status_emoji = "[OK]" if is_robust else "[!!]"

    print(f"\n{status_emoji} Estrategia: {status}")
    print(f"    Score Composto: {result.get('composite_score', 0):.2f}/100")

    # Performance
    print("\nPERFORMANCE:")
    print("-" * 50)
    perf = result.get('performance', {})
    print(f"  Retorno Total:    {perf.get('total_return', 0) * 100:.2f}%")
    print(f"  Sharpe Ratio:     {perf.get('sharpe_ratio', 0):.4f}")
    print(f"  Sortino Ratio:    {perf.get('sortino_ratio', 0):.4f}")
    print(f"  Profit Factor:    {perf.get('profit_factor', 0):.2f}")
    print(f"  Win Rate:         {perf.get('win_rate', 0) * 100:.1f}%")
    print(f"  Max Drawdown:     {perf.get('max_drawdown', 0) * 100:.2f}%")
    print(f"  Total Trades:     {perf.get('total_trades', 0)}")

    # Robustez
    print("\nROBUSTEZ WFO:")
    print("-" * 50)
    rob = result.get('robustness', {})
    print(f"  OOS/IS Ratio:     {rob.get('oos_is_ratio', 0):.2f}")
    print(f"  Consistencia:     {rob.get('consistency_score', 0) * 100:.1f}%")
    print(f"  Estabilidade:     {rob.get('stability_score', 0) * 100:.1f}%")
    print(f"  Degradacao:       {rob.get('degradation', 0) * 100:.1f}%")
    print(f"  MC Confidence:    {rob.get('monte_carlo_confidence', 0) * 100:.1f}%")

    # Parametros
    if verbose:
        print("\nPARAMETROS OTIMIZADOS:")
        print("-" * 50)
        params = result.get('params', {})
        for key, value in sorted(params.items()):
            print(f"  {key}: {value}")

    # Janelas WFO
    windows = result.get('windows', [])
    if windows and verbose:
        print("\nJANELAS WFO:")
        print("-" * 50)
        for i, w in enumerate(windows, 1):
            is_ret = w.get('is_return', 0) * 100
            oos_ret = w.get('oos_return', 0) * 100
            is_sharpe = w.get('is_sharpe', 0)
            oos_sharpe = w.get('oos_sharpe', 0)
            print(f"  Janela {i}: IS {is_ret:+.2f}% (Sharpe {is_sharpe:.2f}) | "
                  f"OOS {oos_ret:+.2f}% (Sharpe {oos_sharpe:.2f})")

    print("\n" + "=" * 70)


def run_optimization(args) -> Optional[Dict]:
    """Executa otimizacao WFO."""
    symbols = get_symbols(args.symbols)

    print_config(args, symbols)

    try:
        # Cria engine WFO
        print("[1/4] Inicializando WFO Engine...")
        engine = WFOEngine(
            symbols=symbols,
            timeframe=args.timeframe,
            n_windows=args.windows,
            is_ratio=args.is_ratio
        )

        # Configura metodo de otimizacao
        print(f"[2/4] Configurando otimizador {args.method.upper()}...")

        if args.method == 'genetic':
            engine.optimizer = GeneticOptimizer(
                population_size=args.population,
                n_generations=args.generations
            )
        elif args.method == 'bayesian':
            engine.optimizer = BayesianOptimizer(
                n_trials=args.bayesian_trials
            )
        else:  # hybrid
            engine.optimizer = HybridOptimizer(
                bayesian_trials=args.bayesian_trials,
                genetic_generations=args.generations,
                genetic_population=args.population
            )

        # Executa otimizacao
        print("[3/4] Executando Walk-Forward Optimization...")
        print("      (isso pode demorar alguns minutos...)\n")

        result = engine.run()

        if result is None:
            print("[ERRO] Otimizacao falhou - nenhum resultado retornado")
            return None

        # Converte resultado para dict se necessario
        if hasattr(result, 'to_dict'):
            result_dict = result.to_dict()
        else:
            result_dict = result

        print("[4/4] Otimizacao concluida!")

        return result_dict

    except Exception as e:
        print(f"\n[ERRO] Falha na otimizacao: {e}")
        traceback.print_exc()
        return None


def save_results(result: Dict, args):
    """Salva resultados no storage."""
    if args.dry_run:
        print("\n[DRY-RUN] Resultados nao salvos")
        return

    try:
        storage = StrategyStorage()

        # Cria metricas completas
        from wfo.metrics import PerformanceMetrics, StrategyMetrics

        perf_data = result.get('performance', {})
        rob_data = result.get('robustness', {})

        perf = PerformanceMetrics(**{k: v for k, v in perf_data.items()
                                     if k in PerformanceMetrics.__dataclass_fields__})
        rob = RobustnessMetrics(**{k: v for k, v in rob_data.items()
                                   if k in RobustnessMetrics.__dataclass_fields__})

        metrics = StrategyMetrics(
            performance=perf,
            robustness=rob,
            strategy_name=result.get('strategy_name', 'stoch_extreme'),
            params=result.get('best_params', result.get('params', {})),
            symbols=result.get('symbols', []),
            timeframe=result.get('timeframe', '1h'),
            composite_score=result.get('composite_score', 0),
            created_at=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat()
        )

        strategy_id = storage.save_strategy(metrics)
        print(f"\n[SALVO] Estrategia armazenada com ID: {strategy_id}")

        # Exporta se solicitado
        if args.export:
            storage.export_for_production(output_file=args.output)
            print(f"[EXPORTADO] Estrategia exportada para: {args.output}")

    except Exception as e:
        print(f"\n[ERRO] Falha ao salvar: {e}")
        traceback.print_exc()


def main():
    """Funcao principal."""
    args = parse_args()

    print_header()

    # Executa otimizacao
    result = run_optimization(args)

    if result:
        # Imprime resultados
        print_results(result, verbose=args.verbose)

        # Salva resultados
        save_results(result, args)

        # Status final
        is_robust = result.get('robustness', {}).get('is_robust', False)
        if is_robust:
            print("\n[SUCESSO] Estrategia robusta encontrada!")
            return 0
        else:
            print("\n[AVISO] Estrategia nao passou nos criterios de robustez")
            return 1
    else:
        print("\n[FALHA] Otimizacao nao produziu resultados validos")
        return 2


if __name__ == '__main__':
    sys.exit(main())
