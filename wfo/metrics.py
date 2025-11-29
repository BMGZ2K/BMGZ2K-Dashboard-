"""
WFO Metrics Module
==================
Calculo completo e preciso de metricas de performance e robustez.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


@dataclass
class RobustnessMetrics:
    """Metricas de robustez WFO."""
    oos_is_ratio: float = 0.0          # Ratio OOS/IS performance
    consistency_score: float = 0.0      # % janelas lucrativas
    stability_score: float = 0.0        # 1 - coef variacao returns
    degradation: float = 0.0            # Degradacao OOS vs IS
    regime_stability: float = 0.0       # Estabilidade entre regimes
    parameter_stability: float = 0.0    # Estabilidade dos parametros
    monte_carlo_confidence: float = 0.0 # Confianca Monte Carlo
    is_robust: bool = False

    def to_dict(self) -> Dict:
        return {
            'oos_is_ratio': round(self.oos_is_ratio, 4),
            'consistency_score': round(self.consistency_score, 4),
            'stability_score': round(self.stability_score, 4),
            'degradation': round(self.degradation, 4),
            'regime_stability': round(self.regime_stability, 4),
            'parameter_stability': round(self.parameter_stability, 4),
            'monte_carlo_confidence': round(self.monte_carlo_confidence, 4),
            'is_robust': self.is_robust
        }


@dataclass
class PerformanceMetrics:
    """Metricas completas de performance."""
    # Returns
    total_return: float = 0.0
    annual_return: float = 0.0
    monthly_return: float = 0.0

    # Risk-adjusted
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    omega_ratio: float = 0.0

    # Drawdown
    max_drawdown: float = 0.0
    avg_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    recovery_factor: float = 0.0

    # Win/Loss
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    payoff_ratio: float = 0.0

    # Trade stats
    total_trades: int = 0
    avg_trade: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0

    # Time analysis
    avg_trade_duration: float = 0.0
    trades_per_day: float = 0.0

    # Risk metrics
    var_95: float = 0.0  # Value at Risk 95%
    cvar_95: float = 0.0  # Conditional VaR 95%
    tail_ratio: float = 0.0

    def to_dict(self) -> Dict:
        return {
            'total_return': round(self.total_return, 4),
            'annual_return': round(self.annual_return, 4),
            'monthly_return': round(self.monthly_return, 4),
            'sharpe_ratio': round(self.sharpe_ratio, 4),
            'sortino_ratio': round(self.sortino_ratio, 4),
            'calmar_ratio': round(self.calmar_ratio, 4),
            'omega_ratio': round(self.omega_ratio, 4),
            'max_drawdown': round(self.max_drawdown, 4),
            'avg_drawdown': round(self.avg_drawdown, 4),
            'max_drawdown_duration': self.max_drawdown_duration,
            'recovery_factor': round(self.recovery_factor, 4),
            'win_rate': round(self.win_rate, 4),
            'profit_factor': round(self.profit_factor, 4),
            'expectancy': round(self.expectancy, 4),
            'payoff_ratio': round(self.payoff_ratio, 4),
            'total_trades': self.total_trades,
            'avg_trade': round(self.avg_trade, 4),
            'avg_win': round(self.avg_win, 4),
            'avg_loss': round(self.avg_loss, 4),
            'largest_win': round(self.largest_win, 4),
            'largest_loss': round(self.largest_loss, 4),
            'consecutive_wins': self.consecutive_wins,
            'consecutive_losses': self.consecutive_losses,
            'avg_trade_duration': round(self.avg_trade_duration, 2),
            'trades_per_day': round(self.trades_per_day, 4),
            'var_95': round(self.var_95, 4),
            'cvar_95': round(self.cvar_95, 4),
            'tail_ratio': round(self.tail_ratio, 4)
        }


@dataclass
class StrategyMetrics:
    """Metricas completas de estrategia."""
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    robustness: RobustnessMetrics = field(default_factory=RobustnessMetrics)

    # Meta info
    strategy_name: str = ""
    params: Dict = field(default_factory=dict)
    symbols: List[str] = field(default_factory=list)
    timeframe: str = "1h"

    # Timestamps
    created_at: str = ""
    last_updated: str = ""

    # Composite score
    composite_score: float = 0.0

    def to_dict(self) -> Dict:
        return {
            'strategy_name': self.strategy_name,
            'params': self.params,
            'symbols': self.symbols,
            'timeframe': self.timeframe,
            'performance': self.performance.to_dict(),
            'robustness': self.robustness.to_dict(),
            'composite_score': round(self.composite_score, 4),
            'created_at': self.created_at,
            'last_updated': self.last_updated
        }


def calculate_returns(equity_curve: np.ndarray) -> np.ndarray:
    """Calcula retornos a partir da curva de equity."""
    if len(equity_curve) < 2:
        return np.array([0.0])

    returns = np.diff(equity_curve) / equity_curve[:-1]
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    return returns


def calculate_drawdown_series(equity_curve: np.ndarray) -> np.ndarray:
    """Calcula serie de drawdowns."""
    if len(equity_curve) == 0:
        return np.array([0.0])

    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    return np.nan_to_num(drawdown, nan=0.0)


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0,
                           periods_per_year: int = 8760) -> float:
    """
    Calcula Sharpe Ratio anualizado.

    Args:
        returns: Array de retornos
        risk_free_rate: Taxa livre de risco anual
        periods_per_year: Periodos por ano (8760 para hourly)
    """
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - risk_free_rate / periods_per_year
    mean_return = np.mean(excess_returns)
    std_return = np.std(returns, ddof=1)

    if std_return == 0:
        return 0.0

    sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)
    return np.clip(sharpe, -10, 10)


def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0,
                            periods_per_year: int = 8760) -> float:
    """Calcula Sortino Ratio (usa apenas volatilidade negativa)."""
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - risk_free_rate / periods_per_year
    mean_return = np.mean(excess_returns)

    negative_returns = returns[returns < 0]
    if len(negative_returns) == 0:
        return 10.0 if mean_return > 0 else 0.0

    downside_std = np.std(negative_returns, ddof=1)
    if downside_std == 0:
        return 0.0

    sortino = (mean_return / downside_std) * np.sqrt(periods_per_year)
    return np.clip(sortino, -10, 10)


def calculate_calmar_ratio(annual_return: float, max_drawdown: float) -> float:
    """Calcula Calmar Ratio (return / max_drawdown)."""
    if abs(max_drawdown) < 0.001:
        return 10.0 if annual_return > 0 else 0.0

    calmar = annual_return / abs(max_drawdown)
    return np.clip(calmar, -10, 10)


def calculate_omega_ratio(returns: np.ndarray, threshold: float = 0.0) -> float:
    """
    Calcula Omega Ratio.

    Omega = soma(ganhos acima threshold) / soma(perdas abaixo threshold)
    """
    if len(returns) == 0:
        return 0.0

    gains = returns[returns > threshold] - threshold
    losses = threshold - returns[returns < threshold]

    total_gains = np.sum(gains) if len(gains) > 0 else 0.0
    total_losses = np.sum(losses) if len(losses) > 0 else 0.0

    if total_losses == 0:
        return 10.0 if total_gains > 0 else 1.0

    omega = total_gains / total_losses
    return np.clip(omega, 0, 10)


def calculate_var(returns: np.ndarray, confidence: float = 0.95) -> float:
    """
    Calcula Value at Risk (VaR).

    Retorna a perda maxima esperada com X% de confianca.
    """
    if len(returns) < 10:
        return 0.0

    var = np.percentile(returns, (1 - confidence) * 100)
    return var


def calculate_cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
    """
    Calcula Conditional Value at Risk (CVaR / Expected Shortfall).

    Media das perdas que excedem o VaR.
    """
    if len(returns) < 10:
        return 0.0

    var = calculate_var(returns, confidence)
    cvar = np.mean(returns[returns <= var])
    return cvar if not np.isnan(cvar) else var


def calculate_tail_ratio(returns: np.ndarray) -> float:
    """
    Calcula Tail Ratio.

    Ratio entre o percentil 95 (ganhos) e percentil 5 (perdas).
    """
    if len(returns) < 20:
        return 1.0

    p95 = np.percentile(returns, 95)
    p05 = np.percentile(returns, 5)

    if abs(p05) < 0.0001:
        return 10.0 if p95 > 0 else 1.0

    ratio = abs(p95 / p05)
    return np.clip(ratio, 0, 10)


def calculate_max_consecutive(pnls: List[float], positive: bool = True) -> int:
    """Calcula maior sequencia de trades positivos ou negativos."""
    if len(pnls) == 0:
        return 0

    max_streak = 0
    current_streak = 0

    for pnl in pnls:
        if (positive and pnl > 0) or (not positive and pnl < 0):
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0

    return max_streak


def calculate_metrics(
    trades: List[Dict],
    equity_curve: np.ndarray,
    initial_capital: float = 10000.0,
    periods_per_year: int = 8760  # Hourly
) -> PerformanceMetrics:
    """
    Calcula todas as metricas de performance a partir dos trades.

    Args:
        trades: Lista de trades com 'pnl', 'entry_time', 'exit_time', etc
        equity_curve: Curva de equity
        initial_capital: Capital inicial
        periods_per_year: Periodos por ano para anualizacao

    Returns:
        PerformanceMetrics com todas as metricas calculadas
    """
    metrics = PerformanceMetrics()

    if len(trades) == 0 or len(equity_curve) < 2:
        return metrics

    # Extrai PnLs
    pnls = [t.get('pnl', t.get('pnl_pct', 0)) for t in trades]
    pnls = [p for p in pnls if p is not None and not np.isnan(p)]

    if len(pnls) == 0:
        return metrics

    # Returns da equity curve
    returns = calculate_returns(equity_curve)

    # ========== RETORNOS ==========
    final_equity = equity_curve[-1]
    metrics.total_return = (final_equity - initial_capital) / initial_capital

    # Calcula periodo em anos
    n_periods = len(equity_curve)
    years = n_periods / periods_per_year

    if years > 0:
        metrics.annual_return = (1 + metrics.total_return) ** (1 / years) - 1
        metrics.monthly_return = (1 + metrics.annual_return) ** (1/12) - 1

    # ========== RISK-ADJUSTED ==========
    metrics.sharpe_ratio = calculate_sharpe_ratio(returns, periods_per_year=periods_per_year)
    metrics.sortino_ratio = calculate_sortino_ratio(returns, periods_per_year=periods_per_year)
    metrics.omega_ratio = calculate_omega_ratio(returns)

    # ========== DRAWDOWN ==========
    drawdown_series = calculate_drawdown_series(equity_curve)
    metrics.max_drawdown = abs(np.min(drawdown_series))
    metrics.avg_drawdown = abs(np.mean(drawdown_series[drawdown_series < 0])) if np.any(drawdown_series < 0) else 0.0

    # Duracao maxima do drawdown
    in_drawdown = False
    current_duration = 0
    max_duration = 0

    for dd in drawdown_series:
        if dd < 0:
            current_duration += 1
            in_drawdown = True
        else:
            if in_drawdown:
                max_duration = max(max_duration, current_duration)
                current_duration = 0
                in_drawdown = False

    max_duration = max(max_duration, current_duration)
    metrics.max_drawdown_duration = max_duration

    # Calmar e Recovery Factor
    metrics.calmar_ratio = calculate_calmar_ratio(metrics.annual_return, metrics.max_drawdown)

    if metrics.max_drawdown > 0:
        metrics.recovery_factor = metrics.total_return / metrics.max_drawdown

    # ========== WIN/LOSS ==========
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    metrics.total_trades = len(pnls)
    metrics.win_rate = len(wins) / len(pnls) if len(pnls) > 0 else 0.0

    total_wins = sum(wins) if wins else 0.0
    total_losses = abs(sum(losses)) if losses else 0.0

    metrics.profit_factor = total_wins / total_losses if total_losses > 0 else (10.0 if total_wins > 0 else 0.0)
    metrics.profit_factor = min(metrics.profit_factor, 10.0)

    metrics.avg_trade = np.mean(pnls)
    metrics.avg_win = np.mean(wins) if wins else 0.0
    metrics.avg_loss = abs(np.mean(losses)) if losses else 0.0

    # Payoff Ratio (avg_win / avg_loss)
    if metrics.avg_loss > 0:
        metrics.payoff_ratio = metrics.avg_win / metrics.avg_loss
    else:
        metrics.payoff_ratio = 10.0 if metrics.avg_win > 0 else 0.0

    # Expectancy (media de retorno por trade)
    metrics.expectancy = (metrics.win_rate * metrics.avg_win) - ((1 - metrics.win_rate) * metrics.avg_loss)

    # Largest win/loss
    metrics.largest_win = max(wins) if wins else 0.0
    metrics.largest_loss = abs(min(losses)) if losses else 0.0

    # Consecutive
    metrics.consecutive_wins = calculate_max_consecutive(pnls, positive=True)
    metrics.consecutive_losses = calculate_max_consecutive(pnls, positive=False)

    # ========== TIME ANALYSIS ==========
    # Duracao media dos trades
    durations = []
    for t in trades:
        entry = t.get('entry_time')
        exit_t = t.get('exit_time')
        if entry and exit_t:
            if isinstance(entry, str):
                entry = datetime.fromisoformat(entry)
            if isinstance(exit_t, str):
                exit_t = datetime.fromisoformat(exit_t)
            if hasattr(entry, 'timestamp') and hasattr(exit_t, 'timestamp'):
                durations.append((exit_t - entry).total_seconds() / 3600)  # Em horas

    metrics.avg_trade_duration = np.mean(durations) if durations else 0.0

    # Trades por dia
    if years > 0:
        metrics.trades_per_day = len(pnls) / (years * 365)

    # ========== RISK METRICS ==========
    metrics.var_95 = calculate_var(returns, 0.95)
    metrics.cvar_95 = calculate_cvar(returns, 0.95)
    metrics.tail_ratio = calculate_tail_ratio(returns)

    return metrics


def calculate_robustness_metrics(
    is_results: List[Dict],
    oos_results: List[Dict],
    params_history: List[Dict] = None
) -> RobustnessMetrics:
    """
    Calcula metricas de robustez WFO.

    Args:
        is_results: Resultados In-Sample por janela
        oos_results: Resultados Out-of-Sample por janela
        params_history: Historico de parametros otimizados

    Returns:
        RobustnessMetrics com todas as metricas
    """
    metrics = RobustnessMetrics()

    if len(is_results) == 0 or len(oos_results) == 0:
        return metrics

    # Extrai retornos e sharpes
    is_returns = [r.get('total_return', 0) for r in is_results]
    oos_returns = [r.get('total_return', 0) for r in oos_results]

    is_sharpes = [r.get('sharpe_ratio', 0) for r in is_results]
    oos_sharpes = [r.get('sharpe_ratio', 0) for r in oos_results]

    # ========== OOS/IS RATIO ==========
    # Ratio entre performance OOS e IS
    avg_is_return = np.mean(is_returns) if is_returns else 0
    avg_oos_return = np.mean(oos_returns) if oos_returns else 0

    if abs(avg_is_return) > 0.001:
        metrics.oos_is_ratio = avg_oos_return / avg_is_return
    else:
        metrics.oos_is_ratio = 1.0 if avg_oos_return >= 0 else 0.0

    # ========== CONSISTENCY SCORE ==========
    # % de janelas OOS com retorno positivo
    profitable_windows = sum(1 for r in oos_returns if r > 0)
    metrics.consistency_score = profitable_windows / len(oos_returns) if oos_returns else 0.0

    # ========== STABILITY SCORE ==========
    # 1 - coeficiente de variacao dos retornos OOS
    if len(oos_returns) > 1:
        mean_oos = np.mean(oos_returns)
        std_oos = np.std(oos_returns)

        if abs(mean_oos) > 0.001:
            cv = std_oos / abs(mean_oos)
            metrics.stability_score = max(0, 1 - min(cv, 2) / 2)
        else:
            metrics.stability_score = 0.5

    # ========== DEGRADATION ==========
    # Quanto a performance degrada de IS para OOS
    if abs(avg_is_return) > 0.001:
        metrics.degradation = (avg_is_return - avg_oos_return) / abs(avg_is_return)
    else:
        metrics.degradation = 0.0

    # ========== REGIME STABILITY ==========
    # Correlacao entre retornos IS e OOS (idealmente baixa degradacao consistente)
    if len(is_returns) > 2 and len(oos_returns) > 2:
        min_len = min(len(is_returns), len(oos_returns))
        corr = np.corrcoef(is_returns[:min_len], oos_returns[:min_len])[0, 1]

        if not np.isnan(corr):
            # Alta correlacao positiva = bom (performance similar)
            metrics.regime_stability = (corr + 1) / 2  # Normaliza para 0-1
        else:
            metrics.regime_stability = 0.5

    # ========== PARAMETER STABILITY ==========
    # Quao estaveis sao os parametros otimizados entre janelas
    if params_history and len(params_history) > 1:
        param_variations = []

        # Para cada parametro, calcula coeficiente de variacao
        all_keys = set()
        for p in params_history:
            all_keys.update(p.keys())

        for key in all_keys:
            values = [p.get(key, 0) for p in params_history if key in p]
            if len(values) > 1 and isinstance(values[0], (int, float)):
                mean_val = np.mean(values)
                std_val = np.std(values)

                if abs(mean_val) > 0.001:
                    cv = std_val / abs(mean_val)
                    param_variations.append(cv)

        if param_variations:
            avg_cv = np.mean(param_variations)
            metrics.parameter_stability = max(0, 1 - min(avg_cv, 2) / 2)
    else:
        metrics.parameter_stability = 0.5

    # ========== MONTE CARLO CONFIDENCE ==========
    # Simula distribuicao de retornos para estimar confianca
    if len(oos_returns) >= 3:
        # Bootstrap dos retornos OOS
        n_simulations = 1000
        simulated_means = []

        for _ in range(n_simulations):
            sample = np.random.choice(oos_returns, size=len(oos_returns), replace=True)
            simulated_means.append(np.mean(sample))

        # % de simulacoes com retorno positivo
        positive_simulations = sum(1 for m in simulated_means if m > 0)
        metrics.monte_carlo_confidence = positive_simulations / n_simulations

    # ========== IS ROBUST ==========
    # Criterios para considerar estrategia robusta
    metrics.is_robust = (
        metrics.oos_is_ratio >= 0.5 and           # OOS >= 50% do IS
        metrics.consistency_score >= 0.6 and       # >= 60% janelas lucrativas
        metrics.stability_score >= 0.4 and         # Estabilidade razoavel
        metrics.degradation <= 0.5 and             # Degradacao <= 50%
        metrics.monte_carlo_confidence >= 0.6      # >= 60% confianca MC
    )

    return metrics


def calculate_composite_score(
    performance: PerformanceMetrics,
    robustness: RobustnessMetrics,
    weights: Dict[str, float] = None
) -> float:
    """
    Calcula score composto combinando performance e robustez.

    Args:
        performance: Metricas de performance
        robustness: Metricas de robustez
        weights: Pesos customizados (opcional)

    Returns:
        Score composto normalizado (0-100)
    """
    if weights is None:
        weights = {
            # Performance (50%)
            'sharpe_ratio': 0.10,
            'sortino_ratio': 0.08,
            'profit_factor': 0.08,
            'win_rate': 0.06,
            'total_return': 0.08,
            'max_drawdown': 0.10,

            # Robustness (50%)
            'oos_is_ratio': 0.12,
            'consistency_score': 0.12,
            'stability_score': 0.10,
            'parameter_stability': 0.08,
            'monte_carlo_confidence': 0.08
        }

    score = 0.0

    # ========== PERFORMANCE SCORES ==========

    # Sharpe (normalizado: -2 a 4 -> 0 a 100)
    sharpe_score = np.clip((performance.sharpe_ratio + 2) / 6 * 100, 0, 100)
    score += sharpe_score * weights.get('sharpe_ratio', 0)

    # Sortino (normalizado: -2 a 5 -> 0 a 100)
    sortino_score = np.clip((performance.sortino_ratio + 2) / 7 * 100, 0, 100)
    score += sortino_score * weights.get('sortino_ratio', 0)

    # Profit Factor (normalizado: 0 a 3 -> 0 a 100)
    pf_score = np.clip(performance.profit_factor / 3 * 100, 0, 100)
    score += pf_score * weights.get('profit_factor', 0)

    # Win Rate (ja em 0-1, converte para 0-100)
    wr_score = performance.win_rate * 100
    score += wr_score * weights.get('win_rate', 0)

    # Total Return (normalizado: -50% a 200% -> 0 a 100)
    return_score = np.clip((performance.total_return + 0.5) / 2.5 * 100, 0, 100)
    score += return_score * weights.get('total_return', 0)

    # Max Drawdown (invertido: menor = melhor)
    # Normalizado: 50% a 0% -> 0 a 100
    dd_score = np.clip((0.5 - performance.max_drawdown) / 0.5 * 100, 0, 100)
    score += dd_score * weights.get('max_drawdown', 0)

    # ========== ROBUSTNESS SCORES ==========

    # OOS/IS Ratio (normalizado: 0 a 1.5 -> 0 a 100)
    oos_is_score = np.clip(robustness.oos_is_ratio / 1.5 * 100, 0, 100)
    score += oos_is_score * weights.get('oos_is_ratio', 0)

    # Consistency (ja em 0-1)
    consistency_score = robustness.consistency_score * 100
    score += consistency_score * weights.get('consistency_score', 0)

    # Stability (ja em 0-1)
    stability_score = robustness.stability_score * 100
    score += stability_score * weights.get('stability_score', 0)

    # Parameter Stability (ja em 0-1)
    param_score = robustness.parameter_stability * 100
    score += param_score * weights.get('parameter_stability', 0)

    # Monte Carlo Confidence (ja em 0-1)
    mc_score = robustness.monte_carlo_confidence * 100
    score += mc_score * weights.get('monte_carlo_confidence', 0)

    return np.clip(score, 0, 100)


def create_strategy_metrics(
    strategy_name: str,
    params: Dict,
    symbols: List[str],
    timeframe: str,
    trades: List[Dict],
    equity_curve: np.ndarray,
    is_results: List[Dict],
    oos_results: List[Dict],
    params_history: List[Dict] = None,
    initial_capital: float = 10000.0
) -> StrategyMetrics:
    """
    Cria objeto completo de metricas para uma estrategia.

    Args:
        strategy_name: Nome da estrategia
        params: Parametros otimizados
        symbols: Simbolos usados
        timeframe: Timeframe
        trades: Lista de trades
        equity_curve: Curva de equity
        is_results: Resultados In-Sample
        oos_results: Resultados Out-of-Sample
        params_history: Historico de parametros
        initial_capital: Capital inicial

    Returns:
        StrategyMetrics completo
    """
    # Calcula metricas de performance
    performance = calculate_metrics(trades, equity_curve, initial_capital)

    # Calcula metricas de robustez
    robustness = calculate_robustness_metrics(is_results, oos_results, params_history)

    # Calcula score composto
    composite = calculate_composite_score(performance, robustness)

    # Cria objeto final
    now = datetime.now().isoformat()

    return StrategyMetrics(
        performance=performance,
        robustness=robustness,
        strategy_name=strategy_name,
        params=params,
        symbols=symbols,
        timeframe=timeframe,
        created_at=now,
        last_updated=now,
        composite_score=composite
    )
