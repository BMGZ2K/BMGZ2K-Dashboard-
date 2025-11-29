"""
Walk-Forward Optimization Engine
================================
Sistema robusto de validacao out-of-sample.

WFO divide dados em periodos:
- In-Sample (IS): Usado para otimizacao
- Out-of-Sample (OOS): Usado para validacao

Metodologia:
1. Divide dados em N janelas
2. Para cada janela: otimiza em IS, valida em OOS
3. Concatena resultados OOS para metricas finais
4. Estrategia e robusta se performance OOS e consistente
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import Config
from .backtester import PreciseBacktester, BacktestResult
from .optimizers import GeneticOptimizer, BayesianOptimizer, HybridOptimizer

log = logging.getLogger(__name__)


@dataclass
class WFOWindow:
    """Representa uma janela WFO."""
    window_id: int
    is_start: datetime
    is_end: datetime
    oos_start: datetime
    oos_end: datetime
    is_params: Dict = field(default_factory=dict)
    is_score: float = 0
    oos_result: Optional[BacktestResult] = None
    oos_score: float = 0


@dataclass
class WFOResult:
    """Resultado completo do WFO."""
    # Metricas agregadas OOS
    total_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    profit_factor: float
    win_rate: float
    total_trades: int
    calmar_ratio: float
    recovery_factor: float

    # Robustez
    oos_is_ratio: float  # OOS/IS performance ratio
    consistency_score: float  # % janelas lucrativas
    stability_score: float  # Desvio padrao dos retornos
    degradation_pct: float  # Queda IS -> OOS

    # Detalhes
    windows: List[WFOWindow] = field(default_factory=list)
    best_params: Dict = field(default_factory=dict)
    combined_equity_curve: List[float] = field(default_factory=list)
    combined_trades: List = field(default_factory=list)

    # Metadata
    optimization_method: str = ''
    total_optimization_time: float = 0
    is_robust: bool = False


class WFOEngine:
    """
    Walk-Forward Optimization Engine.

    Executa otimizacao robusta com validacao out-of-sample.
    """

    def __init__(
        self,
        symbols: List[str] = None,
        timeframe: str = None,
        initial_capital: float = None,
        n_windows: int = None,
        is_ratio: float = None,
        min_trades_per_window: int = None,
        optimization_method: str = 'hybrid',  # 'genetic', 'bayesian', 'hybrid'
        # Criterios de robustez
        min_oos_is_ratio: float = None,
        min_consistency: float = None,
        max_degradation: float = None,
        min_sharpe: float = None,
        min_profit_factor: float = None,
        max_drawdown: float = None
    ):
        # Load WFO config
        wfo_config = Config.get_section('wfo')
        robustness_config = wfo_config.get('robustness', {})
        fitness_config = wfo_config.get('fitness', {})

        # Core settings from Config
        self.symbols = symbols or Config.get('symbols', ['BTC/USDT', 'ETH/USDT'])
        self.timeframe = timeframe or Config.get('timeframes.primary', '1h')
        self.initial_capital = initial_capital or Config.get('backtest.initial_capital', 10000)
        self.n_windows = n_windows or wfo_config.get('n_windows', 5)
        self.is_ratio = is_ratio if is_ratio is not None else wfo_config.get('is_ratio', 0.7)
        self.min_trades_per_window = min_trades_per_window or wfo_config.get('min_trades_per_window', 10)
        self.optimization_method = optimization_method

        # Criterios de robustez from Config
        self.min_oos_is_ratio = min_oos_is_ratio if min_oos_is_ratio is not None else robustness_config.get('min_oos_is_ratio', 0.6)
        self.min_consistency = min_consistency if min_consistency is not None else robustness_config.get('min_consistency', 0.6)
        self.max_degradation = max_degradation if max_degradation is not None else robustness_config.get('max_degradation', 0.5)
        self.min_sharpe = min_sharpe if min_sharpe is not None else robustness_config.get('min_sharpe', 1.0)
        self.min_profit_factor = min_profit_factor if min_profit_factor is not None else robustness_config.get('min_profit_factor', 1.3)
        self.max_drawdown = max_drawdown if max_drawdown is not None else robustness_config.get('max_drawdown', 0.25)

        # Fitness weights from Config
        self._fitness_weights = {
            'sharpe': fitness_config.get('sharpe_weight', 0.4),
            'profit_factor': fitness_config.get('profit_factor_weight', 0.2),
            'win_rate': fitness_config.get('win_rate_weight', 0.2),
            'trades_factor': fitness_config.get('trades_factor_weight', 0.1),
            'drawdown_penalty': fitness_config.get('drawdown_penalty_weight', 0.1),
            'trades_normalization': fitness_config.get('trades_normalization', 50)
        }

        # Data days and annualization
        self._data_days = wfo_config.get('data_days', 180)
        self._warmup_period = wfo_config.get('warmup_period', 50)
        self._annualization_factor = wfo_config.get('annualization', {}).get('periods_per_year_1h', 8760)

        # Data
        self.data_dict: Dict[str, pd.DataFrame] = {}
        self.windows: List[WFOWindow] = []

    def load_data(self, data_dict: Dict[str, pd.DataFrame] = None, days: int = None):
        """
        Carrega dados para otimizacao.

        Args:
            data_dict: Dict de symbol -> DataFrame (se None, busca da API)
            days: Dias de historico para buscar (default: from Config)
        """
        if data_dict:
            self.data_dict = data_dict
            return

        # Use config value if not specified
        days = days or self._data_days

        # Busca dados da Binance
        from core.data import fetch_ohlcv_multi

        log.info(f"Buscando {days} dias de dados para {len(self.symbols)} symbols...")
        self.data_dict = fetch_ohlcv_multi(
            symbols=self.symbols,
            timeframe=self.timeframe,
            limit=days * 24  # 24 candles por dia para 1h
        )
        log.info(f"Dados carregados: {len(self.data_dict)} symbols")

    def _create_windows(self) -> List[WFOWindow]:
        """Cria janelas WFO baseado nos dados."""
        if not self.data_dict:
            raise ValueError("Dados nao carregados. Chame load_data() primeiro.")

        # Encontra range de datas comum
        all_dates = set()
        for df in self.data_dict.values():
            if len(df) > 0:
                all_dates.update(df.index.tolist())

        all_dates = sorted(all_dates)
        if len(all_dates) < 100:
            raise ValueError("Dados insuficientes para WFO")

        total_periods = len(all_dates)
        window_size = total_periods // self.n_windows

        windows = []
        for i in range(self.n_windows):
            start_idx = i * window_size
            end_idx = min((i + 1) * window_size + window_size // 2, total_periods)

            window_data = all_dates[start_idx:end_idx]
            is_size = int(len(window_data) * self.is_ratio)

            is_start = window_data[0]
            is_end = window_data[is_size - 1]
            oos_start = window_data[is_size]
            oos_end = window_data[-1]

            windows.append(WFOWindow(
                window_id=i + 1,
                is_start=is_start,
                is_end=is_end,
                oos_start=oos_start,
                oos_end=oos_end
            ))

        return windows

    def _get_window_data(self, window: WFOWindow, is_period: bool) -> Dict[str, pd.DataFrame]:
        """Extrai dados para um periodo especifico."""
        if is_period:
            start, end = window.is_start, window.is_end
        else:
            start, end = window.oos_start, window.oos_end

        window_data = {}
        for symbol, df in self.data_dict.items():
            mask = (df.index >= start) & (df.index <= end)
            subset = df[mask].copy()
            if len(subset) >= 50:
                window_data[symbol] = subset

        return window_data

    def _create_fitness_function(self, data_dict: Dict[str, pd.DataFrame]) -> Callable:
        """Cria funcao de fitness para otimizacao."""

        # Capture config values for closure
        initial_capital = self.initial_capital
        min_trades = self.min_trades_per_window
        max_dd = self.max_drawdown
        weights = self._fitness_weights

        def fitness_func(params: dict) -> float:
            """
            Avalia parametros usando score composto ponderado.
            Pesos configuráveis via Config.
            """
            try:
                # Backtester uses Config internally for fees
                backtester = PreciseBacktester(
                    initial_capital=initial_capital
                )

                from core.signals import SignalGenerator
                result = backtester.run_multi_symbol(
                    data_dict=data_dict,
                    signals_func=SignalGenerator,
                    params=params,
                    risk_per_trade=params.get('risk_per_trade', Config.get('risk.risk_per_trade', 0.01)),
                    max_positions=params.get('max_positions', Config.get('risk.max_positions', 10))
                )

                if result.total_trades < min_trades:
                    return float('-inf')

                if result.max_drawdown_pct > max_dd * 100:
                    return float('-inf')

                # Score composto com pesos do Config
                sharpe = max(0, result.sharpe_ratio)
                pf = max(0, result.profit_factor - 1)  # Bonus acima de 1
                wr = result.win_rate / 100
                trades_factor = np.sqrt(result.total_trades / weights['trades_normalization'])
                dd_penalty = 1 - (result.max_drawdown_pct / 100)

                score = (
                    sharpe * weights['sharpe'] +
                    pf * weights['profit_factor'] +
                    wr * weights['win_rate'] +
                    trades_factor * weights['trades_factor'] +
                    dd_penalty * weights['drawdown_penalty']
                )

                return score

            except Exception as e:
                log.debug(f"Fitness error: {e}")
                return float('-inf')

        return fitness_func

    def _optimize_window(self, window: WFOWindow) -> Tuple[dict, float]:
        """Otimiza parametros para uma janela IS."""
        is_data = self._get_window_data(window, is_period=True)

        if len(is_data) < 3:
            log.warning(f"Window {window.window_id}: Dados IS insuficientes")
            return {}, float('-inf')

        fitness_func = self._create_fitness_function(is_data)

        # Escolhe otimizador - todos usam Config internamente para defaults
        if self.optimization_method == 'genetic':
            optimizer = GeneticOptimizer()
        elif self.optimization_method == 'bayesian':
            optimizer = BayesianOptimizer()
        else:  # hybrid
            optimizer = HybridOptimizer()

        result = optimizer.optimize(fitness_func)

        return result.best_params, result.best_score

    def _validate_window(self, window: WFOWindow, params: dict) -> BacktestResult:
        """Valida parametros em dados OOS."""
        oos_data = self._get_window_data(window, is_period=False)

        if len(oos_data) < 3:
            log.warning(f"Window {window.window_id}: Dados OOS insuficientes")
            return None

        # Backtester uses Config internally for fees
        backtester = PreciseBacktester(
            initial_capital=self.initial_capital
        )

        from core.signals import SignalGenerator
        result = backtester.run_multi_symbol(
            data_dict=oos_data,
            signals_func=SignalGenerator,
            params=params,
            risk_per_trade=params.get('risk_per_trade', Config.get('risk.risk_per_trade', 0.01)),
            max_positions=params.get('max_positions', Config.get('risk.max_positions', 10))
        )

        return result

    def _validate_window_is(self, window: WFOWindow, params: dict) -> BacktestResult:
        """
        Valida parametros em dados IS (para obter metricas comparaveis com OOS).

        CORRIGIDO: Usado para calcular OOS/IS ratio com mesma metrica (Sharpe).
        """
        is_data = self._get_window_data(window, is_period=True)

        if len(is_data) < 3:
            return None

        backtester = PreciseBacktester(
            initial_capital=self.initial_capital
        )

        from core.signals import SignalGenerator
        result = backtester.run_multi_symbol(
            data_dict=is_data,
            signals_func=SignalGenerator,
            params=params,
            risk_per_trade=params.get('risk_per_trade', Config.get('risk.risk_per_trade', 0.01)),
            max_positions=params.get('max_positions', Config.get('risk.max_positions', 10))
        )

        return result

    def run(self) -> WFOResult:
        """
        Executa Walk-Forward Optimization completo.

        Returns:
            WFOResult com metricas agregadas e validacao de robustez
        """
        import time
        start_time = time.time()

        log.info("="*60)
        log.info("WALK-FORWARD OPTIMIZATION")
        log.info("="*60)
        log.info(f"Symbols: {len(self.symbols)}")
        log.info(f"Timeframe: {self.timeframe}")
        log.info(f"Windows: {self.n_windows}")
        log.info(f"IS/OOS Ratio: {self.is_ratio:.0%}/{1-self.is_ratio:.0%}")
        log.info(f"Optimization: {self.optimization_method}")
        log.info("="*60)

        # Carrega dados se necessario
        if not self.data_dict:
            self.load_data(days=self._data_days)

        # Cria janelas
        self.windows = self._create_windows()
        log.info(f"Criadas {len(self.windows)} janelas WFO")

        # Processa cada janela
        all_is_scores = []
        all_is_results = []  # CORRIGIDO: Armazenar resultados IS para métricas comparáveis
        all_oos_results = []
        combined_trades = []
        combined_equity = [self.initial_capital]

        for window in self.windows:
            log.info(f"\n--- Window {window.window_id}/{len(self.windows)} ---")
            log.info(f"IS: {window.is_start} -> {window.is_end}")
            log.info(f"OOS: {window.oos_start} -> {window.oos_end}")

            # Otimiza em IS
            log.info("Otimizando em In-Sample...")
            params, is_score = self._optimize_window(window)

            if not params:
                log.warning("Otimizacao falhou, pulando janela")
                continue

            window.is_params = params
            window.is_score = is_score
            all_is_scores.append(is_score)

            log.info(f"IS Score: {is_score:.4f}")

            # CORRIGIDO: Rodar backtest IS com params otimizados para obter Sharpe IS comparável
            is_result = self._validate_window_is(window, params)
            if is_result and is_result.total_trades > 0:
                all_is_results.append(is_result)
                log.info(f"IS Backtest: Sharpe={is_result.sharpe_ratio:.2f}")

            # Valida em OOS
            log.info("Validando em Out-of-Sample...")
            oos_result = self._validate_window(window, params)

            if oos_result and oos_result.total_trades > 0:
                window.oos_result = oos_result
                window.oos_score = oos_result.sharpe_ratio
                all_oos_results.append(oos_result)
                combined_trades.extend(oos_result.trades)

                # Atualiza equity combinada
                if oos_result.equity_curve:
                    scale = combined_equity[-1] / oos_result.equity_curve[0]
                    scaled_equity = [e * scale for e in oos_result.equity_curve[1:]]
                    combined_equity.extend(scaled_equity)

                log.info(f"OOS: Return={oos_result.total_return_pct:.2f}% | "
                        f"Sharpe={oos_result.sharpe_ratio:.2f} | "
                        f"WR={oos_result.win_rate:.1f}% | "
                        f"Trades={oos_result.total_trades}")
            else:
                log.warning("Validacao OOS sem trades")

        # Calcula metricas agregadas
        if not all_oos_results:
            log.error("Nenhum resultado OOS valido")
            return self._empty_result()

        wfo_result = self._calculate_aggregate_metrics(
            all_is_scores,
            all_is_results,  # CORRIGIDO: Passar IS results para métricas comparáveis
            all_oos_results,
            combined_equity,
            combined_trades
        )

        wfo_result.windows = self.windows
        wfo_result.optimization_method = self.optimization_method
        wfo_result.total_optimization_time = time.time() - start_time

        # Encontra melhor params (mais consistente)
        best_window = max(
            [w for w in self.windows if w.oos_result],
            key=lambda w: w.oos_result.sharpe_ratio if w.oos_result else 0
        )
        wfo_result.best_params = best_window.is_params

        # Valida robustez
        wfo_result.is_robust = self._validate_robustness(wfo_result)

        self._print_summary(wfo_result)

        return wfo_result

    def _calculate_aggregate_metrics(
        self,
        is_scores: List[float],
        is_results: List[BacktestResult],  # CORRIGIDO: IS backtest results para comparação justa
        oos_results: List[BacktestResult],
        combined_equity: List[float],
        combined_trades: List
    ) -> WFOResult:
        """Calcula metricas agregadas de todos os periodos OOS."""

        # Metricas agregadas
        total_return = (combined_equity[-1] - combined_equity[0]) / combined_equity[0] * 100

        # Sharpe da equity combinada
        returns = []
        for i in range(1, len(combined_equity)):
            r = (combined_equity[i] - combined_equity[i-1]) / combined_equity[i-1]
            returns.append(r)

        if returns:
            annualization = self._annualization_factor
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(annualization) if np.std(returns) > 0 else 0
            neg_returns = [r for r in returns if r < 0]
            sortino = np.mean(returns) / np.std(neg_returns) * np.sqrt(annualization) if neg_returns else sharpe
        else:
            sharpe = sortino = 0

        # Max drawdown
        peak = combined_equity[0]
        max_dd = 0
        for eq in combined_equity:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak
            if dd > max_dd:
                max_dd = dd

        # Profit factor e win rate
        wins = [t for t in combined_trades if t.pnl > 0]
        losses = [t for t in combined_trades if t.pnl <= 0]
        total_win = sum(t.pnl for t in wins) if wins else 0
        total_loss = abs(sum(t.pnl for t in losses)) if losses else 0
        pf = total_win / total_loss if total_loss > 0 else float('inf')
        wr = len(wins) / len(combined_trades) * 100 if combined_trades else 0

        # Calmar ratio
        calmar = total_return / (max_dd * 100) if max_dd > 0 else float('inf')

        # Recovery factor
        recovery = total_return / (max_dd * 100) if max_dd > 0 else float('inf')

        # Robustez - CORRIGIDO: Usar mesma métrica (Sharpe) para IS e OOS
        # Filtra IS results válidos (não None)
        valid_is_results = [r for r in is_results if r is not None]
        avg_is_sharpe = np.mean([r.sharpe_ratio for r in valid_is_results]) if valid_is_results else 0
        avg_oos_sharpe = np.mean([r.sharpe_ratio for r in oos_results])
        oos_is_ratio = avg_oos_sharpe / avg_is_sharpe if avg_is_sharpe > 0 else 0

        # Consistency (% janelas lucrativas)
        profitable_windows = sum(1 for r in oos_results if r.total_return > 0)
        consistency = profitable_windows / len(oos_results) if oos_results else 0

        # Stability (desvio dos retornos)
        window_returns = [r.total_return_pct for r in oos_results]
        stability = 1 / (1 + np.std(window_returns)) if window_returns else 0

        # Degradation IS -> OOS - CORRIGIDO: Usa retorno IS real
        avg_is_return = np.mean([r.total_return_pct for r in valid_is_results]) if valid_is_results else 0
        avg_oos_return = total_return / len(oos_results) if oos_results else 0
        degradation = max(0, 1 - avg_oos_return / avg_is_return) if avg_is_return > 0 else 1

        return WFOResult(
            total_return_pct=total_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown_pct=max_dd * 100,
            profit_factor=pf,
            win_rate=wr,
            total_trades=len(combined_trades),
            calmar_ratio=calmar,
            recovery_factor=recovery,
            oos_is_ratio=oos_is_ratio,
            consistency_score=consistency,
            stability_score=stability,
            degradation_pct=degradation * 100,
            combined_equity_curve=combined_equity,
            combined_trades=combined_trades
        )

    def _validate_robustness(self, result: WFOResult) -> bool:
        """Verifica se estrategia passa nos criterios de robustez."""
        checks = {
            'Sharpe >= min': result.sharpe_ratio >= self.min_sharpe,
            'PF >= min': result.profit_factor >= self.min_profit_factor,
            'DD <= max': result.max_drawdown_pct <= self.max_drawdown * 100,
            'OOS/IS >= min': result.oos_is_ratio >= self.min_oos_is_ratio,
            'Consistency >= min': result.consistency_score >= self.min_consistency,
            'Degradation <= max': result.degradation_pct <= self.max_degradation * 100
        }

        log.info("\n=== Validacao de Robustez ===")
        all_passed = True
        for check, passed in checks.items():
            status = "PASS" if passed else "FAIL"
            log.info(f"  {check}: {status}")
            if not passed:
                all_passed = False

        return all_passed

    def _print_summary(self, result: WFOResult):
        """Imprime resumo dos resultados."""
        log.info("\n" + "="*60)
        log.info("RESULTADOS WFO")
        log.info("="*60)
        log.info(f"Total Return: {result.total_return_pct:.2f}%")
        log.info(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        log.info(f"Sortino Ratio: {result.sortino_ratio:.2f}")
        log.info(f"Max Drawdown: {result.max_drawdown_pct:.2f}%")
        log.info(f"Profit Factor: {result.profit_factor:.2f}")
        log.info(f"Win Rate: {result.win_rate:.1f}%")
        log.info(f"Total Trades: {result.total_trades}")
        log.info(f"Calmar Ratio: {result.calmar_ratio:.2f}")
        log.info("-"*40)
        log.info(f"OOS/IS Ratio: {result.oos_is_ratio:.2f}")
        log.info(f"Consistency: {result.consistency_score:.1%}")
        log.info(f"Stability: {result.stability_score:.2f}")
        log.info(f"Degradation: {result.degradation_pct:.1f}%")
        log.info("-"*40)
        log.info(f"IS ROBUST: {'YES' if result.is_robust else 'NO'}")
        log.info(f"Optimization Time: {result.total_optimization_time:.1f}s")
        log.info("="*60)

    def _empty_result(self) -> WFOResult:
        """Retorna resultado vazio."""
        return WFOResult(
            total_return_pct=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            max_drawdown_pct=0,
            profit_factor=0,
            win_rate=0,
            total_trades=0,
            calmar_ratio=0,
            recovery_factor=0,
            oos_is_ratio=0,
            consistency_score=0,
            stability_score=0,
            degradation_pct=100,
            is_robust=False
        )
