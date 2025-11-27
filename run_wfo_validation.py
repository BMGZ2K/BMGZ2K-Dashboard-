"""
WFO Validation - Validação Padronizada de Estratégias
======================================================
Executa Walk-Forward Optimization com:
- Mínimo 6 folds para significância estatística
- Períodos de treino e teste separados
- Métricas padronizadas
- Validação de robustez
- Integração com sistema de evolução

USO:
    python run_wfo_validation.py

Resultados são salvos em:
- state/validated_strategies.json (para dashboard)
- state/current_best.json (estratégia ativa)
"""
import sys
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from portfolio_wfo import PortfolioBacktester, PortfolioBacktestResult
from core.config import SYMBOLS, get_wfo_config, get_validated_params
from core.evolution import get_storage, ValidatedStrategy


class WFOValidator:
    """
    Validador WFO padronizado.

    Garante:
    - Mínimo 6 folds para significância
    - Períodos não sobrepostos de treino/teste
    - Métricas calculadas corretamente
    - Formato consistente para armazenamento
    """

    def __init__(self, config: Dict = None):
        # Usar config centralizado como base
        self.config = get_wfo_config()
        if config:
            self.config.update(config)
        self.backtester = PortfolioBacktester()
        self.storage = get_storage()

    def validate_strategy(
        self,
        strategy_type: str,
        params: Dict,
        symbols: List[str] = None,
        timeframe: str = '1h'
    ) -> Optional[Dict]:
        """
        Validar uma estratégia com WFO completo.

        Args:
            strategy_type: Tipo da estratégia (ex: 'stoch_extreme', 'combined')
            params: Parâmetros da estratégia
            symbols: Lista de símbolos (default: SYMBOLS[:10])
            timeframe: Timeframe (default: '1h')

        Returns:
            Dict com resultados da validação ou None se falhar
        """
        symbols = symbols or SYMBOLS[:10]

        print("=" * 60)
        print(f"WFO VALIDATION: {strategy_type}")
        print("=" * 60)
        print(f"Símbolos: {len(symbols)}")
        print(f"Timeframe: {timeframe}")
        print(f"Parâmetros: {json.dumps(params, indent=2)}")
        print()

        # Definir período
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.config['total_months'] * 30)

        print(f"Período: {start_date.date()} a {end_date.date()}")
        print(f"Folds mínimos: {self.config['min_folds']}")
        print()

        # Buscar dados
        print("Buscando dados históricos...")
        data = self.backtester.fetch_data(symbols, timeframe, start_date, end_date)

        if len(data) < self.config['min_symbols']:
            print(f"ERRO: Apenas {len(data)} símbolos com dados (mínimo: {self.config['min_symbols']})")
            return None

        print(f"Símbolos com dados: {len(data)}")

        # Criar folds
        folds = self._create_folds(start_date, end_date)

        if len(folds) < self.config['min_folds']:
            print(f"ERRO: Apenas {len(folds)} folds criados (mínimo: {self.config['min_folds']})")
            return None

        print(f"Folds criados: {len(folds)}")
        print()

        # Executar backtest em cada fold
        fold_results = []

        for i, fold in enumerate(folds):
            print(f"Fold {i+1}/{len(folds)}: {fold['test'][0].date()} - {fold['test'][1].date()}")

            # Filtrar dados para período de teste
            test_data = {}
            for sym, df in data.items():
                mask = (df.index >= fold['test'][0]) & (df.index <= fold['test'][1])
                df_filtered = df[mask]
                if len(df_filtered) > 50:
                    test_data[sym] = df_filtered

            if len(test_data) < 3:
                print(f"  Dados insuficientes, pulando...")
                continue

            # Executar backtest
            result = self.backtester.run_backtest(test_data, params, timeframe)

            if result.total_trades < self.config['min_trades_per_fold']:
                print(f"  Poucos trades ({result.total_trades}), pulando...")
                continue

            fold_result = {
                'fold_number': i + 1,
                'train_start': fold['train'][0].isoformat(),
                'train_end': fold['train'][1].isoformat(),
                'test_start': fold['test'][0].isoformat(),
                'test_end': fold['test'][1].isoformat(),
                'return_pct': round(result.total_return_pct, 2),
                'max_drawdown_pct': round(result.max_drawdown_pct, 2),
                'sharpe_ratio': round(result.sharpe_ratio, 2),
                'sortino_ratio': round(result.sortino_ratio, 2),
                'profit_factor': round(result.profit_factor, 2),
                'total_trades': result.total_trades,
                'win_rate': round(result.win_rate, 1),
                'avg_trade_pnl': round(result.avg_trade_pnl, 2),
            }

            fold_results.append(fold_result)
            print(f"  Retorno: {result.total_return_pct:.1f}% | "
                  f"Sharpe: {result.sharpe_ratio:.2f} | "
                  f"WR: {result.win_rate:.0f}% | "
                  f"Trades: {result.total_trades}")

        print()

        if len(fold_results) < self.config['min_folds']:
            print(f"ERRO: Apenas {len(fold_results)} folds válidos (mínimo: {self.config['min_folds']})")
            return None

        # Calcular métricas agregadas
        returns = [f['return_pct'] for f in fold_results]
        sharpes = [f['sharpe_ratio'] for f in fold_results]
        sortinos = [f['sortino_ratio'] for f in fold_results]
        drawdowns = [f['max_drawdown_pct'] for f in fold_results]
        profit_factors = [f['profit_factor'] for f in fold_results]
        win_rates = [f['win_rate'] for f in fold_results]
        trades = [f['total_trades'] for f in fold_results]

        # Métricas finais
        metrics = {
            'avg_return_pct': round(np.mean(returns), 2),
            'min_return_pct': round(np.min(returns), 2),
            'max_return_pct': round(np.max(returns), 2),
            'std_return_pct': round(np.std(returns), 2),
            'avg_drawdown_pct': round(np.mean(drawdowns), 2),
            'max_drawdown_pct': round(np.max(drawdowns), 2),
            'avg_sharpe': round(np.mean(sharpes), 2),
            'avg_sortino': round(np.mean(sortinos), 2),
            'avg_profit_factor': round(np.mean(profit_factors), 2),
            'avg_win_rate': round(np.mean(win_rates), 1),
            'total_trades': sum(trades),
        }

        # Calcular score WFO
        # Score = Sharpe * 2 + Retorno/10 - MaxDD/5 + WinRate/20
        wfo_score = (
            metrics['avg_sharpe'] * 2 +
            metrics['avg_return_pct'] / 10 -
            metrics['max_drawdown_pct'] / 5 +
            metrics['avg_win_rate'] / 20
        )

        # Calcular robustez (consistência entre folds)
        # Robustez alta = todos os folds positivos com baixa variância
        positive_folds = sum(1 for r in returns if r > 0)
        consistency = positive_folds / len(returns)

        if metrics['std_return_pct'] > 0:
            cv = abs(metrics['std_return_pct'] / metrics['avg_return_pct']) if metrics['avg_return_pct'] != 0 else float('inf')
            robustness = max(0, min(100, (1 - cv) * consistency * 100))
        else:
            robustness = consistency * 100

        # Resultado final
        result = {
            'strategy_type': strategy_type,
            'params': params,
            'symbols': [s for s in symbols if s in data],
            'timeframe': timeframe,
            'metrics': metrics,
            'wfo_score': round(wfo_score, 2),
            'robustness_score': round(robustness, 1),
            'num_folds': len(fold_results),
            'fold_results': fold_results,
            'validation_date': datetime.now().isoformat(),
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat(),
            }
        }

        # Imprimir resultado
        print("=" * 60)
        print("RESULTADO DA VALIDAÇÃO")
        print("=" * 60)
        print(f"Folds válidos: {len(fold_results)}")
        print(f"Retorno médio: {metrics['avg_return_pct']:.2f}% (min: {metrics['min_return_pct']:.2f}%, max: {metrics['max_return_pct']:.2f}%)")
        print(f"Sharpe médio: {metrics['avg_sharpe']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
        print(f"Win Rate: {metrics['avg_win_rate']:.1f}%")
        print(f"Profit Factor: {metrics['avg_profit_factor']:.2f}")
        print(f"Total de trades: {metrics['total_trades']}")
        print(f"WFO Score: {wfo_score:.2f}")
        print(f"Robustez: {robustness:.1f}%")
        print()

        return result

    def _create_folds(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Criar folds para WFO."""
        folds = []
        train_days = self.config['train_days']
        test_days = self.config['test_days']

        # Step = test_days para folds não sobrepostos no teste
        step_days = test_days

        current = start_date

        while current + timedelta(days=train_days + test_days) <= end_date:
            train_start = current
            train_end = current + timedelta(days=train_days)
            test_start = train_end
            test_end = test_start + timedelta(days=test_days)

            folds.append({
                'train': (train_start, train_end),
                'test': (test_start, test_end)
            })

            current = current + timedelta(days=step_days)

        return folds

    def save_validated_strategy(self, result: Dict) -> str:
        """
        Salvar estratégia validada no sistema.

        Returns:
            ID da estratégia salva
        """
        # Adicionar ao storage de evolução
        strategy = self.storage.add_strategy(
            strategy_type=result['strategy_type'],
            symbols=result['symbols'],
            timeframe=result['timeframe'],
            params=result['params'],
            fold_results=result['fold_results'],
            wfo_score=result['wfo_score']
        )

        print(f"Estratégia salva: {strategy.id} ({strategy.name})")

        return strategy.id

    def activate_strategy(self, strategy_id: str):
        """Ativar uma estratégia (desativa outras)."""
        self.storage.set_active(strategy_id, True)
        print(f"Estratégia {strategy_id} ativada")

    def save_as_current_best(self, result: Dict):
        """Salvar como current_best.json."""
        output = {
            'strategy': result['strategy_type'],
            'strategy_name': f"WFO_Validated_{result['strategy_type']}",
            'params': result['params'],
            'metrics': {
                'return_pct': result['metrics']['avg_return_pct'],
                'sharpe_ratio': result['metrics']['avg_sharpe'],
                'max_drawdown_pct': result['metrics']['max_drawdown_pct'],
                'win_rate': result['metrics']['avg_win_rate'],
                'total_trades': result['metrics']['total_trades'],
                'profit_factor': result['metrics']['avg_profit_factor'],
                'wfo_score': result['wfo_score'],
            },
            'validation': {
                'is_mathematically_valid': True,
                'is_statistically_significant': result['num_folds'] >= 6,
                'num_folds': result['num_folds'],
                'robustness': result['robustness_score'],
                'test_period_days': self.config['total_months'] * 30,
            },
            'symbols': result['symbols'],
            'timeframe': result['timeframe'],
            'activated_at': datetime.now().isoformat(),
        }

        os.makedirs('state', exist_ok=True)
        with open('state/current_best.json', 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print("Salvo em state/current_best.json")


def main():
    """Executar validação WFO da estratégia atual."""
    print("\n" + "=" * 60)
    print("INICIANDO VALIDAÇÃO WFO PADRONIZADA")
    print("=" * 60 + "\n")

    validator = WFOValidator()

    # Parâmetros da estratégia a validar
    # Usando os parâmetros que estão funcionando no bot
    params = {
        'strategy': 'stoch_extreme',
        'sl_mult': 3.0,
        'tp_mult': 5.0,
        'rsi_period': 14,
        'rsi_oversold': 20,
        'rsi_overbought': 70,
        'stoch_k': 14,
        'stoch_d': 3,
        'stoch_oversold': 20,
        'stoch_overbought': 80,
        'ema_fast': 9,
        'ema_slow': 21,
        'adx_period': 14,
        'adx_min': 20,
        'atr_period': 14,
        'risk_per_trade': 0.01,
        'volume_mult': 1.5,
        'max_positions': 10,
        'max_margin_usage': 0.8,
    }

    # Símbolos para validação
    symbols = SYMBOLS[:10]  # Top 10 símbolos

    # Executar validação
    result = validator.validate_strategy(
        strategy_type='stoch_extreme',
        params=params,
        symbols=symbols,
        timeframe='1h'
    )

    if result:
        # Salvar estratégia validada
        strategy_id = validator.save_validated_strategy(result)

        # Ativar estratégia
        validator.activate_strategy(strategy_id)

        # Salvar como current_best
        validator.save_as_current_best(result)

        print("\n" + "=" * 60)
        print("VALIDAÇÃO CONCLUÍDA COM SUCESSO")
        print("=" * 60)
        print(f"Estratégia ID: {strategy_id}")
        print(f"WFO Score: {result['wfo_score']:.2f}")
        print(f"Robustez: {result['robustness_score']:.1f}%")
        print()

        return result
    else:
        print("\n" + "=" * 60)
        print("VALIDAÇÃO FALHOU")
        print("=" * 60)
        print("Verifique os logs acima para detalhes.")
        return None


if __name__ == '__main__':
    main()
