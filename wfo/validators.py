"""
WFO Validators Module
=====================
Validacao de robustez e qualidade de estrategias.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


@dataclass
class ValidationResult:
    """Resultado de validacao."""
    is_valid: bool = False
    score: float = 0.0
    passed_tests: List[str] = field(default_factory=list)
    failed_tests: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'is_valid': self.is_valid,
            'score': round(self.score, 4),
            'passed_tests': self.passed_tests,
            'failed_tests': self.failed_tests,
            'warnings': self.warnings,
            'details': self.details
        }


class RobustnessValidator:
    """
    Validador de robustez para estrategias.

    Testes implementados:
    1. Walk-Forward Validation
    2. Monte Carlo Simulation
    3. Parameter Stability
    4. Regime Detection
    5. Overfitting Detection
    6. Statistical Significance
    """

    def __init__(
        self,
        min_oos_is_ratio: float = 0.5,
        min_consistency: float = 0.6,
        min_sharpe: float = 0.5,
        max_drawdown: float = 0.3,
        min_trades: int = 30,
        confidence_level: float = 0.95
    ):
        """
        Inicializa o validador.

        Args:
            min_oos_is_ratio: Ratio minimo OOS/IS
            min_consistency: Consistencia minima (% janelas lucrativas)
            min_sharpe: Sharpe minimo
            max_drawdown: Drawdown maximo permitido
            min_trades: Numero minimo de trades
            confidence_level: Nivel de confianca estatistica
        """
        self.min_oos_is_ratio = min_oos_is_ratio
        self.min_consistency = min_consistency
        self.min_sharpe = min_sharpe
        self.max_drawdown = max_drawdown
        self.min_trades = min_trades
        self.confidence_level = confidence_level

    def validate(
        self,
        is_results: List[Dict],
        oos_results: List[Dict],
        trades: List[Dict],
        equity_curve: np.ndarray,
        params_history: List[Dict] = None
    ) -> ValidationResult:
        """
        Executa validacao completa de robustez.

        Args:
            is_results: Resultados In-Sample por janela
            oos_results: Resultados Out-of-Sample por janela
            trades: Lista de trades
            equity_curve: Curva de equity
            params_history: Historico de parametros otimizados

        Returns:
            ValidationResult com resultado da validacao
        """
        result = ValidationResult()
        tests_passed = 0
        total_tests = 8

        # ========== TEST 1: Numero minimo de trades ==========
        if len(trades) >= self.min_trades:
            result.passed_tests.append("min_trades")
            tests_passed += 1
            result.details['trades_count'] = len(trades)
        else:
            result.failed_tests.append("min_trades")
            result.details['trades_count'] = len(trades)
            result.warnings.append(f"Trades insuficientes: {len(trades)} < {self.min_trades}")

        # ========== TEST 2: OOS/IS Ratio ==========
        oos_is_ratio = self._calculate_oos_is_ratio(is_results, oos_results)
        result.details['oos_is_ratio'] = oos_is_ratio

        if oos_is_ratio >= self.min_oos_is_ratio:
            result.passed_tests.append("oos_is_ratio")
            tests_passed += 1
        else:
            result.failed_tests.append("oos_is_ratio")
            result.warnings.append(f"OOS/IS ratio baixo: {oos_is_ratio:.2f} < {self.min_oos_is_ratio}")

        # ========== TEST 3: Consistencia ==========
        consistency = self._calculate_consistency(oos_results)
        result.details['consistency'] = consistency

        if consistency >= self.min_consistency:
            result.passed_tests.append("consistency")
            tests_passed += 1
        else:
            result.failed_tests.append("consistency")
            result.warnings.append(f"Consistencia baixa: {consistency:.2f} < {self.min_consistency}")

        # ========== TEST 4: Sharpe Ratio ==========
        sharpe = self._calculate_sharpe(equity_curve)
        result.details['sharpe_ratio'] = sharpe

        if sharpe >= self.min_sharpe:
            result.passed_tests.append("sharpe_ratio")
            tests_passed += 1
        else:
            result.failed_tests.append("sharpe_ratio")
            result.warnings.append(f"Sharpe baixo: {sharpe:.2f} < {self.min_sharpe}")

        # ========== TEST 5: Max Drawdown ==========
        max_dd = self._calculate_max_drawdown(equity_curve)
        result.details['max_drawdown'] = max_dd

        if max_dd <= self.max_drawdown:
            result.passed_tests.append("max_drawdown")
            tests_passed += 1
        else:
            result.failed_tests.append("max_drawdown")
            result.warnings.append(f"Drawdown alto: {max_dd:.2f} > {self.max_drawdown}")

        # ========== TEST 6: Parameter Stability ==========
        param_stability = self._check_parameter_stability(params_history)
        result.details['param_stability'] = param_stability

        if param_stability >= 0.5:  # >= 50% estabilidade
            result.passed_tests.append("param_stability")
            tests_passed += 1
        else:
            result.failed_tests.append("param_stability")
            result.warnings.append(f"Parametros instaveis: {param_stability:.2f}")

        # ========== TEST 7: Monte Carlo Confidence ==========
        mc_confidence = self._monte_carlo_test(oos_results)
        result.details['monte_carlo_confidence'] = mc_confidence

        if mc_confidence >= 0.6:  # >= 60% confianca
            result.passed_tests.append("monte_carlo")
            tests_passed += 1
        else:
            result.failed_tests.append("monte_carlo")
            result.warnings.append(f"Confianca MC baixa: {mc_confidence:.2f}")

        # ========== TEST 8: Overfitting Detection ==========
        is_overfit = self._detect_overfitting(is_results, oos_results)
        result.details['overfitting_detected'] = is_overfit

        if not is_overfit:
            result.passed_tests.append("overfitting_check")
            tests_passed += 1
        else:
            result.failed_tests.append("overfitting_check")
            result.warnings.append("Possivel overfitting detectado")

        # ========== RESULTADO FINAL ==========
        result.score = tests_passed / total_tests
        result.is_valid = tests_passed >= 5  # Pelo menos 5/8 testes

        return result

    def _calculate_oos_is_ratio(
        self,
        is_results: List[Dict],
        oos_results: List[Dict]
    ) -> float:
        """Calcula ratio entre performance OOS e IS."""
        if not is_results or not oos_results:
            return 0.0

        is_returns = [r.get('total_return', 0) for r in is_results]
        oos_returns = [r.get('total_return', 0) for r in oos_results]

        avg_is = np.mean(is_returns) if is_returns else 0
        avg_oos = np.mean(oos_returns) if oos_returns else 0

        if abs(avg_is) < 0.001:
            return 1.0 if avg_oos >= 0 else 0.0

        return avg_oos / avg_is

    def _calculate_consistency(self, oos_results: List[Dict]) -> float:
        """Calcula % de janelas OOS lucrativas."""
        if not oos_results:
            return 0.0

        profitable = sum(1 for r in oos_results if r.get('total_return', 0) > 0)
        return profitable / len(oos_results)

    def _calculate_sharpe(self, equity_curve: np.ndarray) -> float:
        """Calcula Sharpe Ratio."""
        if len(equity_curve) < 2:
            return 0.0

        returns = np.diff(equity_curve) / equity_curve[:-1]
        returns = np.nan_to_num(returns, nan=0.0)

        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0

        # Anualizado (assumindo hourly)
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(8760)
        return np.clip(sharpe, -10, 10)

    def _calculate_max_drawdown(self, equity_curve: np.ndarray) -> float:
        """Calcula Max Drawdown."""
        if len(equity_curve) == 0:
            return 0.0

        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        return abs(np.min(drawdown))

    def _check_parameter_stability(self, params_history: List[Dict]) -> float:
        """Verifica estabilidade dos parametros entre janelas."""
        if not params_history or len(params_history) < 2:
            return 0.5  # Neutro se nao ha historico

        variations = []

        # Parametros para verificar
        param_keys = set()
        for p in params_history:
            param_keys.update(p.keys())

        for key in param_keys:
            values = [p.get(key) for p in params_history if key in p]
            values = [v for v in values if v is not None and isinstance(v, (int, float))]

            if len(values) > 1:
                mean_val = np.mean(values)
                if abs(mean_val) > 0.001:
                    cv = np.std(values) / abs(mean_val)
                    variations.append(cv)

        if not variations:
            return 0.5

        avg_cv = np.mean(variations)
        # Estabilidade = 1 - coeficiente de variacao (normalizado)
        stability = max(0, 1 - min(avg_cv, 2) / 2)
        return stability

    def _monte_carlo_test(self, oos_results: List[Dict], n_simulations: int = 1000) -> float:
        """
        Teste Monte Carlo de robustez.

        Simula distribuicao de retornos via bootstrap para estimar confianca.
        """
        if not oos_results or len(oos_results) < 3:
            return 0.5

        oos_returns = [r.get('total_return', 0) for r in oos_results]

        # Bootstrap
        simulated_means = []
        for _ in range(n_simulations):
            sample = np.random.choice(oos_returns, size=len(oos_returns), replace=True)
            simulated_means.append(np.mean(sample))

        # % simulacoes com retorno positivo
        positive_count = sum(1 for m in simulated_means if m > 0)
        return positive_count / n_simulations

    def _detect_overfitting(
        self,
        is_results: List[Dict],
        oos_results: List[Dict],
        threshold: float = 0.5
    ) -> bool:
        """
        Detecta overfitting comparando IS vs OOS.

        Overfitting indica se:
        - Performance OOS muito inferior a IS
        - Degradacao consistente entre janelas
        """
        if not is_results or not oos_results:
            return False

        is_returns = [r.get('total_return', 0) for r in is_results]
        oos_returns = [r.get('total_return', 0) for r in oos_results]

        avg_is = np.mean(is_returns)
        avg_oos = np.mean(oos_returns)

        # Degradacao muito grande?
        if abs(avg_is) > 0.01:
            degradation = (avg_is - avg_oos) / abs(avg_is)
            if degradation > threshold:
                return True

        # Sharpe OOS muito menor que IS?
        is_sharpes = [r.get('sharpe_ratio', 0) for r in is_results]
        oos_sharpes = [r.get('sharpe_ratio', 0) for r in oos_results]

        avg_is_sharpe = np.mean(is_sharpes)
        avg_oos_sharpe = np.mean(oos_sharpes)

        if avg_is_sharpe > 0.5 and avg_oos_sharpe < 0:
            return True

        return False


class StatisticalValidator:
    """Validador estatistico para significancia dos resultados."""

    @staticmethod
    def t_test(returns: np.ndarray, threshold: float = 0.0) -> Tuple[float, bool]:
        """
        Teste t para verificar se retornos sao significativamente > threshold.

        Returns:
            (p_value, is_significant)
        """
        if len(returns) < 10:
            return 1.0, False

        from scipy import stats
        t_stat, p_value = stats.ttest_1samp(returns, threshold)

        # One-sided test (retornos > threshold)
        p_one_sided = p_value / 2 if t_stat > 0 else 1 - p_value / 2

        return p_one_sided, p_one_sided < 0.05

    @staticmethod
    def bootstrap_confidence_interval(
        data: np.ndarray,
        statistic: callable = np.mean,
        n_bootstrap: int = 1000,
        confidence: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Calcula intervalo de confianca via bootstrap.

        Returns:
            (lower, mean, upper)
        """
        if len(data) < 5:
            return 0.0, 0.0, 0.0

        bootstrapped = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrapped.append(statistic(sample))

        lower = np.percentile(bootstrapped, (1 - confidence) / 2 * 100)
        upper = np.percentile(bootstrapped, (1 + confidence) / 2 * 100)
        mean = np.mean(bootstrapped)

        return lower, mean, upper

    @staticmethod
    def check_randomness(trades: List[Dict]) -> Tuple[float, bool]:
        """
        Verifica se sequencia de trades nao eh aleatoria.

        Usa runs test para verificar dependencia serial.

        Returns:
            (p_value, is_non_random)
        """
        if len(trades) < 20:
            return 0.5, False

        # Converte para sequencia de wins/losses
        results = [1 if t.get('pnl', 0) > 0 else 0 for t in trades]

        # Conta runs (sequencias de mesma direcao)
        runs = 1
        for i in range(1, len(results)):
            if results[i] != results[i-1]:
                runs += 1

        n1 = sum(results)
        n2 = len(results) - n1

        if n1 == 0 or n2 == 0:
            return 0.5, False

        # Calcula esperado e variancia
        expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
        var_runs = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2) ** 2 * (n1 + n2 - 1))

        if var_runs <= 0:
            return 0.5, False

        z = (runs - expected_runs) / np.sqrt(var_runs)

        # P-value (two-sided)
        from scipy import stats
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        # Non-random se p < 0.05 (sequencia tem padrao)
        return p_value, p_value < 0.05


class RegimeValidator:
    """Valida performance em diferentes regimes de mercado."""

    @staticmethod
    def detect_regimes(prices: np.ndarray, volatility_threshold: float = 0.02) -> np.ndarray:
        """
        Detecta regimes de mercado baseado em volatilidade.

        Returns:
            Array com regime por periodo (0=baixa vol, 1=alta vol)
        """
        if len(prices) < 20:
            return np.zeros(len(prices))

        # Calcula volatilidade rolling
        returns = np.diff(prices) / prices[:-1]
        vol = np.zeros(len(prices))

        window = 20
        for i in range(window, len(returns)):
            vol[i] = np.std(returns[i-window:i])

        # Classifica regimes
        regimes = (vol > volatility_threshold).astype(int)
        return regimes

    @staticmethod
    def validate_by_regime(
        trades: List[Dict],
        prices: np.ndarray,
        volatility_threshold: float = 0.02
    ) -> Dict:
        """
        Valida performance por regime de mercado.

        Returns:
            Estatisticas por regime
        """
        regimes = RegimeValidator.detect_regimes(prices, volatility_threshold)

        results = {
            'low_vol': {'trades': 0, 'wins': 0, 'total_pnl': 0},
            'high_vol': {'trades': 0, 'wins': 0, 'total_pnl': 0}
        }

        for trade in trades:
            # Determina regime do trade (simplificado)
            entry_idx = trade.get('entry_idx', 0)
            if entry_idx < len(regimes):
                regime = 'high_vol' if regimes[entry_idx] == 1 else 'low_vol'
            else:
                regime = 'low_vol'

            pnl = trade.get('pnl', 0)
            results[regime]['trades'] += 1
            results[regime]['total_pnl'] += pnl
            if pnl > 0:
                results[regime]['wins'] += 1

        # Calcula win rates
        for regime in results:
            if results[regime]['trades'] > 0:
                results[regime]['win_rate'] = (
                    results[regime]['wins'] / results[regime]['trades']
                )
                results[regime]['avg_pnl'] = (
                    results[regime]['total_pnl'] / results[regime]['trades']
                )
            else:
                results[regime]['win_rate'] = 0
                results[regime]['avg_pnl'] = 0

        # Verifica consistencia entre regimes
        lr_wr = results['low_vol']['win_rate']
        hr_wr = results['high_vol']['win_rate']

        results['regime_consistency'] = 1 - abs(lr_wr - hr_wr)
        results['is_consistent'] = results['regime_consistency'] >= 0.7

        return results


# ============================================================
# Funcoes auxiliares de conveniencia
# ============================================================

def validate_strategy(
    is_results: List[Dict],
    oos_results: List[Dict],
    trades: List[Dict],
    equity_curve: np.ndarray,
    params_history: List[Dict] = None,
    **kwargs
) -> ValidationResult:
    """
    Funcao de conveniencia para validacao de estrategia.

    Args:
        is_results: Resultados In-Sample
        oos_results: Resultados Out-of-Sample
        trades: Lista de trades
        equity_curve: Curva de equity
        params_history: Historico de parametros
        **kwargs: Parametros adicionais para o validador

    Returns:
        ValidationResult
    """
    validator = RobustnessValidator(**kwargs)
    return validator.validate(
        is_results=is_results,
        oos_results=oos_results,
        trades=trades,
        equity_curve=equity_curve,
        params_history=params_history
    )


def quick_validate(
    sharpe: float,
    profit_factor: float,
    win_rate: float,
    max_drawdown: float,
    total_trades: int
) -> Tuple[bool, List[str]]:
    """
    Validacao rapida de metricas basicas.

    Args:
        sharpe: Sharpe ratio
        profit_factor: Profit factor
        win_rate: Win rate (0-1)
        max_drawdown: Max drawdown (0-1)
        total_trades: Numero de trades

    Returns:
        (is_valid, list of warnings)
    """
    warnings_list = []
    checks_passed = 0
    total_checks = 5

    # Check sharpe
    if sharpe >= 0.5:
        checks_passed += 1
    else:
        warnings_list.append(f"Sharpe baixo: {sharpe:.2f}")

    # Check profit factor
    if profit_factor >= 1.2:
        checks_passed += 1
    else:
        warnings_list.append(f"Profit factor baixo: {profit_factor:.2f}")

    # Check win rate
    if win_rate >= 0.4:
        checks_passed += 1
    else:
        warnings_list.append(f"Win rate baixo: {win_rate:.2%}")

    # Check drawdown
    if max_drawdown <= 0.3:
        checks_passed += 1
    else:
        warnings_list.append(f"Drawdown alto: {max_drawdown:.2%}")

    # Check trades
    if total_trades >= 30:
        checks_passed += 1
    else:
        warnings_list.append(f"Poucos trades: {total_trades}")

    is_valid = checks_passed >= 3  # Pelo menos 3/5

    return is_valid, warnings_list
