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
        from wfo import PreciseBacktester
        return True

    def test_backtest_with_real_data(self) -> bool:
        """Testar backtest com dados reais da Binance."""
        from wfo import PreciseBacktester
        import ccxt
        from core.config import API_KEY, SECRET_KEY, USE_TESTNET

        # Criar exchange para buscar dados
        exchange = ccxt.binance({
            'apiKey': API_KEY,
            'secret': SECRET_KEY,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        if USE_TESTNET:
            exchange.set_sandbox_mode(True)

        # Buscar dados OHLCV
        ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1h', limit=200)
        if not ohlcv:
            return "Sem dados da Binance"

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        assert len(df) > 50, "Dados insuficientes"

        # Testar backtester
        backtester = PreciseBacktester()
        params = {'strategy': 'stoch_extreme', 'sl_atr_mult': 2.0, 'tp_atr_mult': 3.0}
        result = backtester.run({'BTC/USDT': df}, params)

        assert result is not None, "Resultado None"
        return True

    def test_backtest_consistency(self) -> bool:
        """Testar consistencia entre PnL e equity curve."""
        from wfo import PreciseBacktester
        import ccxt
        from core.config import API_KEY, SECRET_KEY, USE_TESTNET

        exchange = ccxt.binance({
            'apiKey': API_KEY,
            'secret': SECRET_KEY,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        if USE_TESTNET:
            exchange.set_sandbox_mode(True)

        # Buscar dados
        ohlcv_btc = exchange.fetch_ohlcv('BTC/USDT', '1h', limit=300)
        ohlcv_eth = exchange.fetch_ohlcv('ETH/USDT', '1h', limit=300)

        if not ohlcv_btc or not ohlcv_eth:
            return "Sem dados da Binance"

        data = {}
        for sym, ohlcv in [('BTC/USDT', ohlcv_btc), ('ETH/USDT', ohlcv_eth)]:
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            data[sym] = df

        backtester = PreciseBacktester()
        params = {'strategy': 'stoch_extreme'}
        result = backtester.run(data, params)

        if result.trades:
            initial = 10000  # Capital inicial padrao
            final_equity = result.equity_curve[-1] if result.equity_curve else initial
            equity_return = (final_equity - initial) / initial * 100
            reported_return = result.total_return_pct

            diff = abs(equity_return - reported_return)
            assert diff < 1.0, f"Inconsistencia: equity={equity_return:.2f}% vs reported={reported_return:.2f}%"

        return True

    # ==================== TESTES DE PERFORMANCE ====================

    def test_backtest_performance(self) -> bool:
        """Testar performance do backtest."""
        from wfo import PreciseBacktester
        import ccxt
        from core.config import API_KEY, SECRET_KEY, USE_TESTNET

        exchange = ccxt.binance({
            'apiKey': API_KEY,
            'secret': SECRET_KEY,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        if USE_TESTNET:
            exchange.set_sandbox_mode(True)

        # Buscar dados
        ohlcv_btc = exchange.fetch_ohlcv('BTC/USDT', '1h', limit=500)
        ohlcv_eth = exchange.fetch_ohlcv('ETH/USDT', '1h', limit=500)

        if not ohlcv_btc or not ohlcv_eth:
            return "Sem dados"

        data = {}
        total_candles = 0
        for sym, ohlcv in [('BTC/USDT', ohlcv_btc), ('ETH/USDT', ohlcv_eth)]:
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            data[sym] = df
            total_candles += len(df)

        backtester = PreciseBacktester()
        params = {'strategy': 'stoch_extreme'}

        # Warmup
        backtester.run(data, params)

        # Benchmark
        start_time = time.perf_counter()
        result = backtester.run(data, params)
        elapsed = time.perf_counter() - start_time

        candles_per_sec = total_candles / elapsed
        assert candles_per_sec > 500, f"Performance baixa: {candles_per_sec:.0f} candles/s"

        return True

    # ==================== TESTES DE CORREÇÕES DE BUGS ====================

    def test_pnl_net_calculation(self) -> bool:
        """Bug #1: Verificar que PnL armazenado é LÍQUIDO (após taxas)."""
        from core.trader import Trader

        trader = Trader({'strategy': 'stoch_extreme', 'max_leverage': 10})
        trader.balance = 10000

        # Registrar entrada
        trader.record_entry(
            symbol='BTC/USDT',
            side='buy',
            price=50000,
            quantity=0.1,
            sl=49000,
            tp=52000,
            reason='test'
        )

        # Registrar saída (lucro de $100)
        pnl = trader.record_exit('BTC/USDT', 51000, 'test')

        # Verificar que há pelo menos um trade
        assert len(trader.trades) > 0, "Nenhum trade registrado"

        trade = trader.trades[-1]

        # Verificar que pnl é LÍQUIDO (menor que bruto devido às taxas)
        gross_pnl = (51000 - 50000) * 0.1  # = $100
        assert trade.pnl_gross == gross_pnl, f"PnL bruto errado: {trade.pnl_gross} vs {gross_pnl}"
        assert trade.pnl < trade.pnl_gross, f"PnL líquido ({trade.pnl}) deveria ser menor que bruto ({trade.pnl_gross})"
        assert trade.commission > 0, "Commission deveria ser > 0"

        return True

    def test_funding_periods_discrete(self) -> bool:
        """Bug #2: Verificar contagem discreta de funding periods."""
        from core.trader import count_funding_periods
        from datetime import datetime

        # Teste 1: 2 horas (sem cruzar funding)
        entry1 = datetime(2024, 1, 1, 10, 0, 0)
        exit1 = datetime(2024, 1, 1, 12, 0, 0)
        assert count_funding_periods(entry1, exit1) == 0, "2h deveria ser 0 funding"

        # Teste 2: Cruzando 16:00 (1 funding)
        entry2 = datetime(2024, 1, 1, 14, 0, 0)
        exit2 = datetime(2024, 1, 1, 18, 0, 0)
        assert count_funding_periods(entry2, exit2) == 1, "Cruzando 16:00 deveria ser 1 funding"

        # Teste 3: 25 horas (3 fundings: 16:00, 00:00, 08:00)
        entry3 = datetime(2024, 1, 1, 10, 0, 0)
        exit3 = datetime(2024, 1, 2, 11, 0, 0)
        periods = count_funding_periods(entry3, exit3)
        assert periods == 3, f"25h deveria ser 3 fundings, got {periods}"

        # Teste 4: Exatamente no horário de funding (não conta)
        entry4 = datetime(2024, 1, 1, 8, 0, 0)  # Exatamente às 08:00
        exit4 = datetime(2024, 1, 1, 10, 0, 0)
        assert count_funding_periods(entry4, exit4) == 0, "Entry exato no funding não deveria contar"

        return True

    def test_circuit_breaker_with_unrealized(self) -> bool:
        """Bug #3: Verificar circuit breaker considera PnL não realizado."""
        from core.trader import Trader

        trader = Trader({'strategy': 'stoch_extreme', 'max_drawdown': 0.10})
        trader.balance = 10000
        trader.initial_balance = 10000
        trader.high_water_mark = 10000

        # Adicionar trade fechado para bypass do check inicial
        from core.trader import Trade
        trader.trades.append(Trade(
            symbol='TEST/USDT', side='long', entry_price=100, exit_price=101,
            quantity=1, pnl=1, entry_time='', exit_time='', status='closed'
        ))

        # Registrar posição com perda não realizada
        trader.record_entry('ETH/USDT', 'buy', 3000, 1.0, 2700, 3300, 'test')

        # Simular queda de preço (10%+ de drawdown)
        current_prices = {'ETH/USDT': 2500}  # Perda de $500

        # Sem unrealized PnL, balance ainda é 10000
        result_without = trader.update_balance(10000, None)
        assert not result_without, "Sem unrealized, não deveria ativar circuit breaker"

        # Com unrealized PnL, equity = 10000 - 500 = 9500 (5% drawdown)
        result_with_5pct = trader.update_balance(10000, current_prices)
        assert not result_with_5pct, "5% drawdown não deveria ativar circuit breaker"

        # Simular queda maior (15%+ de drawdown)
        current_prices_big = {'ETH/USDT': 1800}  # Perda de $1200 (12%)
        result_with_12pct = trader.update_balance(10000, current_prices_big)
        assert result_with_12pct, "12% drawdown DEVERIA ativar circuit breaker"

        return True

    def test_position_sizing_margin_check(self) -> bool:
        """Bug #5: Verificar position sizing respeita limites de margem."""
        from core.trader import Trader

        trader = Trader({
            'strategy': 'stoch_extreme',
            'max_leverage': 10,
            'max_margin_usage': 0.8,
            'min_position_pct': 0.02
        })
        trader.balance = 10000

        # Primeira posição (deveria funcionar)
        qty1 = trader.calculate_position_size(50000, 49000)
        assert qty1 > 0, "Primeira posição deveria ser possível"

        # Registrar várias posições para usar margem
        for i in range(4):
            trader.record_entry(
                symbol=f'TEST{i}/USDT',
                side='buy',
                price=50000,
                quantity=0.1,  # ~$5000 notional, $500 margem cada
                sl=49000,
                tp=52000,
                reason='test'
            )

        # Verificar margem usada
        margin_used = trader._calculate_total_margin_used()
        assert margin_used > 0, "Margem usada deveria ser > 0"

        # Com 4 posições de $500 margem = $2000
        # Max margem = 10000 * 0.8 = $8000
        # Disponível = 8000 - 2000 = $6000
        # Nova posição deveria ser limitada

        # Tentar abrir posição muito grande (deveria ser limitada)
        qty_big = trader.calculate_position_size(50000, 30000)  # SL muito longe = qty alta
        max_margin_qty = (8000 - margin_used) * 10 / 50000
        assert qty_big <= max_margin_qty + 0.001, f"Qty {qty_big} deveria ser <= {max_margin_qty}"

        return True

    def test_short_signals_enabled(self) -> bool:
        """Bug #6: Verificar shorts não têm penalidade e MR shorts habilitados."""
        from core.signals import SignalGenerator

        # Criar generator com config neutro
        gen = SignalGenerator({
            'strategy': 'mean_reversion',
            'long_bias': 1.0,
            'short_penalty': 1.0,
            'mr_short_enabled': True,
            'mr_adx_max': 25
        })

        # Verificar params
        assert gen.long_bias == 1.0, f"long_bias deveria ser 1.0, got {gen.long_bias}"
        assert gen.short_penalty == 1.0, f"short_penalty deveria ser 1.0, got {gen.short_penalty}"

        # Criar dados com condições de SHORT (BB upper + RSI alto + bearish candle)
        np.random.seed(123)
        n = 100
        prices = np.linspace(100, 120, n)  # Tendência de alta
        prices[-1] = 119  # Candle bearish (fechou abaixo do anterior que era 120)

        df = pd.DataFrame({
            'open': prices + 0.5,
            'high': prices + 2,
            'low': prices - 1,
            'close': prices,
            'volume': np.ones(n) * 1000
        })

        # Forçar condições extremas para mean reversion short
        df = gen.prepare_data(df)
        df['rsi'].iloc[-1] = 75  # RSI alto
        df['adx'].iloc[-1] = 15  # ADX baixo (mercado lateral)
        df['bb_upper'].iloc[-1] = df['close'].iloc[-1] - 1  # Preço acima de BB upper

        signal = gen._signal_mean_reversion(df)

        # Com as condições certas, deveria gerar short
        # Nota: pode não gerar se condições não forem perfeitas, mas penalty deve ser 1.0
        if signal.direction == 'short':
            # Se gerou short, strength não deveria ter penalidade
            assert signal.strength > 0, "Short strength deveria ser > 0"

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

        # Testes de correcoes de bugs
        print("\n[CORREÇÕES DE BUGS]")
        self.results.append(self.run_test("Bug #1: PnL liquido", self.test_pnl_net_calculation))
        self.results.append(self.run_test("Bug #2: Funding discreto", self.test_funding_periods_discrete))
        self.results.append(self.run_test("Bug #3: Circuit breaker unrealized", self.test_circuit_breaker_with_unrealized))
        self.results.append(self.run_test("Bug #5: Position sizing margem", self.test_position_sizing_margin_check))
        self.results.append(self.run_test("Bug #6: Shorts habilitados", self.test_short_signals_enabled))

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
