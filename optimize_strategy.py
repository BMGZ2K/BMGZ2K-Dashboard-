"""
Otimização de Estratégia com WFO
=================================
Busca os melhores parâmetros usando Walk-Forward Optimization.

Testa múltiplas combinações e valida com 6+ folds.
"""
import sys
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List
from itertools import product
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from portfolio_wfo import PortfolioBacktester
from core.config import SYMBOLS, get_validated_params, get_optimization_grid, get_wfo_config
from run_wfo_validation import WFOValidator


def optimize_strategy():
    """Otimizar parâmetros da estratégia com WFO."""
    print("\n" + "=" * 60)
    print("OTIMIZAÇÃO DE ESTRATÉGIA COM WFO")
    print("=" * 60 + "\n")

    # Configuração
    symbols = SYMBOLS[:10]
    timeframe = '1h'

    # Grid de parâmetros - do config centralizado
    opt_grid = get_optimization_grid()
    param_grid = {
        'sl_atr_mult': opt_grid['sl_atr_mult'],
        'tp_atr_mult': opt_grid['tp_atr_mult'],
        'rsi_oversold': opt_grid['rsi_oversold'],
        'rsi_overbought': opt_grid['rsi_overbought'],
        'adx_min': opt_grid['adx_min'],
    }

    # Parâmetros base - do config centralizado
    base_params = get_validated_params()

    # Gerar combinações
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))

    print(f"Símbolos: {len(symbols)}")
    print(f"Timeframe: {timeframe}")
    print(f"Combinações a testar: {len(combinations)}")
    print()

    # Configurar validador - usa config centralizado com override para otimização mais rápida
    wfo_config = get_wfo_config()
    wfo_config['min_folds'] = 4  # Menos folds para otimização inicial
    wfo_config['total_months'] = 3
    wfo_config['min_trades_per_fold'] = 2
    wfo_config['min_symbols'] = 3
    validator = WFOValidator(wfo_config)

    # Testar cada combinação
    results = []

    for i, combo in enumerate(combinations):
        params = {**base_params, **dict(zip(param_names, combo))}

        print(f"\n[{i+1}/{len(combinations)}] Testando: sl={params.get('sl_atr_mult')}, tp={params.get('tp_atr_mult')}, "
              f"rsi_os={params['rsi_oversold']}, rsi_ob={params['rsi_overbought']}, adx={params['adx_min']}")

        try:
            result = validator.validate_strategy(
                strategy_type='stoch_extreme',
                params=params,
                symbols=symbols,
                timeframe=timeframe
            )

            if result:
                results.append({
                    'params': dict(zip(param_names, combo)),
                    'full_params': params,
                    'wfo_score': result['wfo_score'],
                    'robustness': result['robustness_score'],
                    'avg_return': result['metrics']['avg_return_pct'],
                    'avg_sharpe': result['metrics']['avg_sharpe'],
                    'max_dd': result['metrics']['max_drawdown_pct'],
                    'win_rate': result['metrics']['avg_win_rate'],
                    'num_folds': result['num_folds'],
                    'result': result,
                })
                print(f"  Score: {result['wfo_score']:.2f} | Return: {result['metrics']['avg_return_pct']:.1f}% | "
                      f"Sharpe: {result['metrics']['avg_sharpe']:.2f}")
        except Exception as e:
            print(f"  Erro: {e}")
            continue

    if not results:
        print("\nNenhum resultado válido encontrado!")
        return None

    # Ordenar por score
    results.sort(key=lambda x: x['wfo_score'], reverse=True)

    # Mostrar top 5
    print("\n" + "=" * 60)
    print("TOP 5 CONFIGURAÇÕES")
    print("=" * 60)

    for i, r in enumerate(results[:5]):
        print(f"\n{i+1}. Score: {r['wfo_score']:.2f}")
        print(f"   Params: {r['params']}")
        print(f"   Return: {r['avg_return']:.1f}% | Sharpe: {r['avg_sharpe']:.2f} | "
              f"DD: {r['max_dd']:.1f}% | WR: {r['win_rate']:.0f}%")

    # Validar melhor configuração com WFO completo (6+ folds)
    best = results[0]
    print("\n" + "=" * 60)
    print("VALIDAÇÃO FINAL DO MELHOR RESULTADO")
    print("=" * 60)

    # Validador com configuração completa
    full_validator = WFOValidator({
        'min_folds': 6,
        'train_days': 30,
        'test_days': 10,
        'total_months': 4,
        'min_trades_per_fold': 3,
        'min_symbols': 5,
    })

    final_result = full_validator.validate_strategy(
        strategy_type='stoch_extreme',
        params=best['full_params'],
        symbols=symbols,
        timeframe=timeframe
    )

    if final_result:
        # Salvar resultado
        strategy_id = full_validator.save_validated_strategy(final_result)
        full_validator.activate_strategy(strategy_id)
        full_validator.save_as_current_best(final_result)

        print("\n" + "=" * 60)
        print("OTIMIZAÇÃO CONCLUÍDA")
        print("=" * 60)
        print(f"Melhor configuração encontrada e validada!")
        print(f"Strategy ID: {strategy_id}")
        print(f"WFO Score: {final_result['wfo_score']:.2f}")
        print(f"Parâmetros otimizados:")
        for k, v in best['params'].items():
            print(f"  {k}: {v}")

        return final_result
    else:
        print("\nValidação final falhou!")
        return None


if __name__ == '__main__':
    optimize_strategy()
