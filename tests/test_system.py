"""
Sistema de Testes Completo
==========================
Testa todos os componentes do sistema de trading.

Uso:
    python tests/test_system.py           # Executa todos os testes
    python tests/test_system.py --quick   # Apenas testes rapidos
    python tests/test_system.py --full    # Testes completos com dados reais
"""
import sys
import os
import time
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

# Adicionar raiz ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestResult:
    """Resultado de um teste."""
    def __init__(self, name: str, passed: bool, message: str = "", duration: float = 0):
        self.name = name
        self.passed = passed
        self.message = message
        self.duration = duration


class SystemTester:
    """Testador completo do sistema."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: List[TestResult] = []

    def log(self, msg: str):
        if self.verbose:
            print(msg)

    def run_test(self, name: str, test_func) -> TestResult:
        """Executar um teste e capturar resultado."""
        self.log(f"\n  Testing: {name}...")
        start = time.perf_counter()
        try:
            result = test_func()
            duration = time.perf_counter() - start
            if result is True or result is None:
                self.log(f"    [OK] {name} ({duration:.3f}s)")
                return TestResult(name, True, "", duration)
            else:
                self.log(f"    [FAIL] {name}: {result}")
                return TestResult(name, False, str(result), duration)
        except Exception as e:
            duration = time.perf_counter() - start
            self.log(f"    [ERROR] {name}: {e}")
            return TestResult(name, False, str(e), duration)

    # ==================== TESTES DE INDICADORES ====================

    def test_indicators_import(self) -> bool:
        """Testar import dos indicadores."""
        from core.signals import (
            calculate_rsi, calculate_atr, calculate_ema,
            calculate_adx, calculate_bollinger, calculate_stochastic
        )
        return True

    def test_rsi_calculation(self) -> bool:
        """Testar calculo de RSI."""
        from core.signals import calculate_rsi

        # Dados de teste
        np.random.seed(42)
        close = pd.Series(np.cumsum(np.random.randn(100)) + 100)
        rsi = calculate_rsi(close, 14)

        # Verificacoes
        assert not rsi.isna().all(), "RSI todo NaN"
        assert rsi.dropna().min() >= 0, "RSI < 0"
        assert rsi.dropna().max() <= 100, "RSI > 100"
        return True

    def test_atr_calculation(self) -> bool:
        """Testar calculo de ATR."""
        from core.signals import calculate_atr

        np.random.seed(42)
        n = 100
        close = pd.Series(np.cumsum(np.random.randn(n)) + 100)
        high = close + abs(np.random.randn(n))
        low = close - abs(np.random.randn(n))

        atr = calculate_atr(high, low, close, 14)

        assert not atr.isna().all(), "ATR todo NaN"
        assert atr.dropna().min() >= 0, "ATR negativo"
        return True

    def test_bollinger_calculation(self) -> bool:
        """Testar calculo de Bollinger Bands."""
        from core.signals import calculate_bollinger

        np.random.seed(42)
        close = pd.Series(np.cumsum(np.random.randn(100)) + 100)
        upper, mid, lower = calculate_bollinger(close, 20, 2.0)

        assert (upper.dropna() >= mid.dropna()).all(), "Upper < Mid"
        assert (mid.dropna() >= lower.dropna()).all(), "Mid < Lower"
        return True

    def test_stochastic_calculation(self) -> bool:
        """Testar calculo de Stochastic."""
        from core.signals import calculate_stochastic

        np.random.seed(42)
        n = 100
        close = pd.Series(np.cumsum(np.random.randn(n)) + 100)
        high = close + abs(np.random.randn(n))
        low = close - abs(np.random.randn(n))

        k, d = calculate_stochastic(high, low, close, 14, 3, 3)

        assert k.dropna().min() >= 0, "Stoch K < 0"
        assert k.dropna().max() <= 100, "Stoch K > 100"
        return True

    def test_indicators_vs_pandas_ta(self) -> bool:
        """Testar precisao vs pandas_ta."""
        try:
            import pandas_ta as ta
        except ImportError:
            return "pandas_ta nao instalado (skip)"

        from core.signals import (
            calculate_rsi, calculate_atr, calculate_bollinger, calculate_stochastic
        )

        np.random.seed(42)
        n = 200
        close = pd.Series(np.cumsum(np.random.randn(n) * 2) + 100)
        high = close + abs(np.random.randn(n) * 2)
        low = close - abs(np.random.randn(n) * 2)

        # RSI
        our_rsi = calculate_rsi(close, 14)
        ta_rsi = ta.rsi(close, length=14)
        rsi_diff = (our_rsi - ta_rsi).dropna().abs().iloc[-50:].max()
        assert rsi_diff < 0.1, f"RSI diff {rsi_diff:.4f} > 0.1"

        # Bollinger
        our_upper, _, _ = calculate_bollinger(close, 20, 2.0)
        ta_bb = ta.bbands(close, length=20, std=2.0)
        bb_diff = (our_upper - ta_bb['BBU_20_2.0']).dropna().abs().iloc[-50:].max()
        assert bb_diff < 0.01, f"BB diff {bb_diff:.4f} > 0.01"

        # Stochastic
        our_k, _ = calculate_stochastic(high, low, close, 14, 3, 3)
        ta_stoch = ta.stoch(high, low, close, k=14, d=3, smooth_k=3)
        stoch_diff = (our_k - ta_stoch['STOCHk_14_3_3']).dropna().abs().iloc[-50:].max()
        assert stoch_diff < 0.1, f"Stoch diff {stoch_diff:.4f} > 0.1"

        return True

    # ==================== TESTES DE SINAIS ====================

    def test_signal_generator_import(self) -> bool:
        """Testar import do SignalGenerator."""
        from core.signals import SignalGenerator, Signal
        return True

    def test_signal_generation(self) -> bool:
        """Testar geracao de sinais."""
        from core.signals import SignalGenerator

        gen = SignalGenerator({'strategy': 'stoch_extreme'})

        # Criar dados de teste
        np.random.seed(42)
        n = 100
        close = np.cumsum(np.random.randn(n)) + 100
        df = pd.DataFrame({
            'open': close + np.random.randn(n) * 0.1,
            'high': close + abs(np.random.randn(n)),
            'low': close - abs(np.random.randn(n)),
            'close': close,
            'volume': np.random.randint(1000, 10000, n)
        })

        signal = gen.generate_signal(df)

        assert signal is not None, "Signal is None"
        assert signal.direction in ['long', 'short', 'none'], f"Invalid direction: {signal.direction}"
        assert 0 <= signal.strength <= 10, f"Invalid strength: {signal.strength}"
        return True

    def test_all_strategies(self) -> bool:
        """Testar todas as estrategias."""
        from core.signals import SignalGenerator

        strategies = ['rsi_extremes', 'stoch_extreme', 'trend_following', 'mean_reversion', 'momentum_burst']

        np.random.seed(42)
        n = 100
        close = np.cumsum(np.random.randn(n)) + 100
        df = pd.DataFrame({
            'open': close + np.random.randn(n) * 0.1,
            'high': close + abs(np.random.randn(n)),
            'low': close - abs(np.random.randn(n)),
            'close': close,
            'volume': np.random.randint(1000, 10000, n)
        })

        for strategy in strategies:
            gen = SignalGenerator({'strategy': strategy})
            signal = gen.generate_signal(df)
            assert signal is not None, f"{strategy}: Signal is None"

        return True

    # ==================== TESTES DE TAXAS BINANCE ====================

    def test_binance_fees_import(self) -> bool:
        """Testar import do modulo de taxas."""
        from core.binance_fees import BinanceFees, get_binance_fees
        return True

    def test_binance_fees_fetch(self) -> bool:
        """Testar busca de taxas da Binance."""
        from core.binance_fees import get_binance_fees

        fees_manager = get_binance_fees(use_testnet=False)
        fees = fees_manager.get_all_fees_for_symbol('BTCUSDT')

        assert 'maker_fee' in fees, "maker_fee ausente"
        assert 'taker_fee' in fees, "taker_fee ausente"
        assert 'funding_rate' in fees, "funding_rate ausente"
        assert fees['maker_fee'] > 0, "maker_fee = 0"
        assert fees['taker_fee'] > 0, "taker_fee = 0"
        return True

    def test_liquidation_price(self) -> bool:
        """Testar calculo de preco de liquidacao."""
        from core.binance_fees import get_binance_fees

        fees = get_binance_fees(use_testnet=False)

        # Long position
        liq_long = fees.calculate_liquidation_price(
            symbol='BTCUSDT',
            side='long',
            entry_price=50000,
            quantity=0.1,
            wallet_balance=1000,
            leverage=5
        )
        assert 0 < liq_long < 50000, f"Liq long invalido: {liq_long}"

        # Short position
        liq_short = fees.calculate_liquidation_price(
            symbol='BTCUSDT',
            side='short',
            entry_price=50000,
            quantity=0.1,
            wallet_balance=1000,
            leverage=5
        )
        assert liq_short > 50000, f"Liq short invalido: {liq_short}"

        return True

    # ==================== TESTES DE BACKTEST ====================

    def test_backtester_import(self) -> bool:
        """Testar import do backtester."""
        from portfolio_wfo import PortfolioBacktester, PortfolioBacktestResult
        return True

    def test_backtest_with_real_data(self) -> bool:
        """Testar backtest com dados reais da Binance."""
        from portfolio_wfo import PortfolioBacktester

        wfo = PortfolioBacktester()

        end = datetime.now()
        start = end - timedelta(days=7)

        data = wfo.fetch_data(['BTCUSDT'], '1h', start, end)
        assert len(data) > 0, "Nenhum dado retornado"
        assert len(data['BTCUSDT']) > 50, "Dados insuficientes"

        params = {'strategy': 'stoch_extreme', 'sl_atr_mult': 2.0, 'tp_atr_mult': 3.0}
        result = wfo.run_backtest(data, params)

        assert result is not None, "Resultado None"
        assert hasattr(result, 'total_return_pct'), "total_return_pct ausente"
        assert hasattr(result, 'equity_curve'), "equity_curve ausente"
        return True

    def test_backtest_consistency(self) -> bool:
        """Testar consistencia entre PnL e equity curve."""
        from portfolio_wfo import PortfolioBacktester

        wfo = PortfolioBacktester()

        end = datetime.now()
        start = end - timedelta(days=14)

        data = wfo.fetch_data(['BTCUSDT', 'ETHUSDT'], '1h', start, end)
        if not data:
            return "Sem dados da Binance"

        params = {'strategy': 'stoch_extreme'}
        result = wfo.run_backtest(data, params)

        if result.trades:
            initial = wfo.config['initial_capital']
            final_equity = result.equity_curve[-1] if result.equity_curve else initial
            equity_return = (final_equity - initial) / initial * 100
            reported_return = result.total_return_pct

            diff = abs(equity_return - reported_return)
            assert diff < 0.1, f"Inconsistencia: equity={equity_return:.2f}% vs reported={reported_return:.2f}%"

        return True

    # ==================== TESTES DE PERFORMANCE ====================

    def test_backtest_performance(self) -> bool:
        """Testar performance do backtest."""
        from portfolio_wfo import PortfolioBacktester

        wfo = PortfolioBacktester()

        end = datetime.now()
        start = end - timedelta(days=30)

        data = wfo.fetch_data(['BTCUSDT', 'ETHUSDT'], '1h', start, end)
        if not data:
            return "Sem dados"

        total_candles = sum(len(df) for df in data.values())
        params = {'strategy': 'stoch_extreme'}

        # Warmup
        wfo.run_backtest(data, params)

        # Benchmark
        start_time = time.perf_counter()
        result = wfo.run_backtest(data, params)
        elapsed = time.perf_counter() - start_time

        candles_per_sec = total_candles / elapsed
        assert candles_per_sec > 1000, f"Performance baixa: {candles_per_sec:.0f} candles/s"

        return True

    # ==================== TESTES DE MODULOS CORE ====================

    def test_scoring_system(self) -> bool:
        """Testar sistema de scoring."""
        from core.scoring import ScoringSystem

        scoring = ScoringSystem()
        score = scoring.calculate_score(
            symbol='BTCUSDT',
            side='long',
            strength=7.5,
            entry_price=50000,
            stop_loss=49000,
            take_profit=52000,
            atr=500,
            historical_winrate=0.55
        )

        assert score is not None, "Score None"
        assert hasattr(score, 'score'), "score attr ausente"
        assert 0 <= score.score <= 15, f"Score fora do range: {score.score}"
        return True

    def test_evolution_storage(self) -> bool:
        """Testar armazenamento de estrategias."""
        from core.evolution import StrategyStorage, get_storage

        storage = get_storage()
        assert storage is not None, "Storage None"
        return True

    def test_config_import(self) -> bool:
        """Testar import de configuracao."""
        from core.config import SYMBOLS, API_KEY, SECRET_KEY
        assert len(SYMBOLS) > 0, "SYMBOLS vazio"
        return True

    # ==================== EXECUTAR TODOS OS TESTES ====================

    def run_all_tests(self, quick: bool = False) -> Dict:
        """Executar todos os testes."""
        print("=" * 60)
        print("TESTES DO SISTEMA DE TRADING")
        print("=" * 60)

        # Testes de indicadores
        print("\n[INDICADORES]")
        self.results.append(self.run_test("Import indicadores", self.test_indicators_import))
        self.results.append(self.run_test("Calculo RSI", self.test_rsi_calculation))
        self.results.append(self.run_test("Calculo ATR", self.test_atr_calculation))
        self.results.append(self.run_test("Calculo Bollinger", self.test_bollinger_calculation))
        self.results.append(self.run_test("Calculo Stochastic", self.test_stochastic_calculation))
        self.results.append(self.run_test("Precisao vs pandas_ta", self.test_indicators_vs_pandas_ta))

        # Testes de sinais
        print("\n[SINAIS]")
        self.results.append(self.run_test("Import SignalGenerator", self.test_signal_generator_import))
        self.results.append(self.run_test("Geracao de sinais", self.test_signal_generation))
        self.results.append(self.run_test("Todas estrategias", self.test_all_strategies))

        # Testes de taxas
        print("\n[TAXAS BINANCE]")
        self.results.append(self.run_test("Import binance_fees", self.test_binance_fees_import))
        self.results.append(self.run_test("Busca taxas", self.test_binance_fees_fetch))
        self.results.append(self.run_test("Preco liquidacao", self.test_liquidation_price))

        # Testes de modulos core
        print("\n[MODULOS CORE]")
        self.results.append(self.run_test("Sistema scoring", self.test_scoring_system))
        self.results.append(self.run_test("Evolution storage", self.test_evolution_storage))
        self.results.append(self.run_test("Config import", self.test_config_import))

        if not quick:
            # Testes de backtest (mais lentos)
            print("\n[BACKTEST]")
            self.results.append(self.run_test("Import backtester", self.test_backtester_import))
            self.results.append(self.run_test("Backtest dados reais", self.test_backtest_with_real_data))
            self.results.append(self.run_test("Consistencia PnL", self.test_backtest_consistency))
            self.results.append(self.run_test("Performance", self.test_backtest_performance))

        # Resumo
        print("\n" + "=" * 60)
        print("RESUMO")
        print("=" * 60)

        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        total_time = sum(r.duration for r in self.results)

        print(f"\nTotal: {len(self.results)} testes")
        print(f"Passou: {passed}")
        print(f"Falhou: {failed}")
        print(f"Tempo total: {total_time:.2f}s")

        if failed > 0:
            print("\nFALHAS:")
            for r in self.results:
                if not r.passed:
                    print(f"  - {r.name}: {r.message}")

        print("\n" + ("=" * 60))
        if failed == 0:
            print("[OK] TODOS OS TESTES PASSARAM!")
        else:
            print(f"[!] {failed} TESTE(S) FALHARAM")
        print("=" * 60)

        return {
            'total': len(self.results),
            'passed': passed,
            'failed': failed,
            'duration': total_time,
            'results': self.results
        }


def main():
    parser = argparse.ArgumentParser(description='Testes do sistema de trading')
    parser.add_argument('--quick', action='store_true', help='Apenas testes rapidos')
    parser.add_argument('--full', action='store_true', help='Testes completos')
    parser.add_argument('-v', '--verbose', action='store_true', default=True)
    args = parser.parse_args()

    tester = SystemTester(verbose=args.verbose)
    results = tester.run_all_tests(quick=args.quick)

    # Exit code baseado em falhas
    sys.exit(0 if results['failed'] == 0 else 1)


if __name__ == "__main__":
    main()
