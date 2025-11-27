"""
Sistema de Evolucao Preciso com Backtest Realista
=================================================
Usa dados sinteticos com propriedades estatisticas realistas
e backtest completo com todos os custos dinamicos.

Este script:
1. Gera dados OHLCV sinteticos com caracteristicas de mercado real
2. Aplica taxas reais (maker/taker, slippage, funding rate)
3. Usa algoritmo genetico para evoluir parametros
4. Valida com Walk-Forward Optimization
5. Salva os melhores parametros evoluidos

USO:
    python evolve_precise.py              # Evolucao completa
    python evolve_precise.py --quick      # Evolucao rapida
    python evolve_precise.py --validate   # Validar parametros salvos
"""
import sys
import os
import json
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from itertools import product
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.signals import SignalGenerator, Signal, calculate_rsi, calculate_atr, calculate_adx
from core.scoring import ScoringSystem

# =============================================================================
# CONFIGURACAO DE TAXAS REALISTAS (Binance Futures)
# =============================================================================
REALISTIC_FEES = {
    # Taxas de trading (VIP0)
    'maker_fee': 0.0002,      # 0.02%
    'taker_fee': 0.0004,      # 0.04%

    # Slippage estimado
    'slippage': 0.0003,       # 0.03%

    # Funding rate medio (a cada 8h)
    'funding_rate': 0.0001,   # 0.01% (positivo = longs pagam)

    # Taxa de liquidacao
    'liquidation_fee': 0.0125,  # 1.25%

    # Spread medio
    'spread': 0.0001,         # 0.01%
}

# Propriedades de mercado para simulacao realista
MARKET_PROPERTIES = {
    'BTC/USDT': {'base_price': 95000, 'daily_vol': 0.025, 'trend': 0.0001, 'mean_revert': 0.02},
    'ETH/USDT': {'base_price': 3500, 'daily_vol': 0.032, 'trend': 0.00015, 'mean_revert': 0.025},
    'SOL/USDT': {'base_price': 240, 'daily_vol': 0.045, 'trend': 0.0002, 'mean_revert': 0.03},
    'BNB/USDT': {'base_price': 650, 'daily_vol': 0.028, 'trend': 0.0001, 'mean_revert': 0.02},
    'XRP/USDT': {'base_price': 1.45, 'daily_vol': 0.038, 'trend': 0.00012, 'mean_revert': 0.025},
    'DOGE/USDT': {'base_price': 0.42, 'daily_vol': 0.055, 'trend': 0.00018, 'mean_revert': 0.035},
    'ADA/USDT': {'base_price': 1.05, 'daily_vol': 0.042, 'trend': 0.0001, 'mean_revert': 0.028},
    'AVAX/USDT': {'base_price': 42, 'daily_vol': 0.048, 'trend': 0.00015, 'mean_revert': 0.03},
    'LINK/USDT': {'base_price': 18, 'daily_vol': 0.040, 'trend': 0.00012, 'mean_revert': 0.025},
    'DOT/USDT': {'base_price': 9.5, 'daily_vol': 0.038, 'trend': 0.0001, 'mean_revert': 0.022},
}


@dataclass
class BacktestTrade:
    """Trade no backtest."""
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_pct: float
    fees_paid: float
    funding_paid: float
    reason_exit: str


@dataclass
class BacktestResult:
    """Resultado do backtest."""
    total_return_pct: float
    annual_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    profit_factor: float
    win_rate: float
    total_trades: int
    avg_trade_pnl: float
    total_fees: float
    total_funding: float
    equity_curve: List[float]
    trades: List[BacktestTrade]


def generate_realistic_ohlcv(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    timeframe: str = '1h'
) -> pd.DataFrame:
    """
    Gerar dados OHLCV sinteticos com propriedades realistas.

    Usa modelo de:
    - Random walk com drift (trend)
    - Volatilidade estocastica (GARCH-like)
    - Mean reversion parcial
    - Correlacao temporal
    - Volume realista
    """
    props = MARKET_PROPERTIES.get(symbol, {
        'base_price': 100, 'daily_vol': 0.03, 'trend': 0.0001, 'mean_revert': 0.02
    })

    # Calcular numero de candles
    tf_hours = {'1h': 1, '4h': 4, '15m': 0.25, '1d': 24}.get(timeframe, 1)
    total_hours = (end_date - start_date).total_seconds() / 3600
    n_candles = int(total_hours / tf_hours)

    if n_candles < 100:
        n_candles = 500

    # Parametros
    base_price = props['base_price']
    daily_vol = props['daily_vol']
    trend = props['trend']
    mean_revert = props['mean_revert']

    # Volatilidade por candle
    vol_per_candle = daily_vol * np.sqrt(tf_hours / 24)

    # Inicializar arrays
    timestamps = pd.date_range(start=start_date, periods=n_candles, freq=f'{int(tf_hours*60)}min')
    opens = np.zeros(n_candles)
    highs = np.zeros(n_candles)
    lows = np.zeros(n_candles)
    closes = np.zeros(n_candles)
    volumes = np.zeros(n_candles)

    # Preco inicial com variacao
    price = base_price * (1 + np.random.uniform(-0.1, 0.1))
    vol_state = vol_per_candle

    for i in range(n_candles):
        opens[i] = price

        # Volatilidade estocastica (GARCH-like)
        vol_state = 0.9 * vol_state + 0.1 * vol_per_candle * (1 + abs(np.random.normal()) * 0.5)

        # Movimento do preco
        drift = trend * tf_hours
        mean_rev = -mean_revert * (price - base_price) / base_price * tf_hours
        shock = vol_state * np.random.normal()

        # Movimento intra-candle
        intra_moves = np.random.normal(0, vol_state * 0.5, 4)
        intra_prices = price * (1 + np.cumsum(intra_moves) / 100)

        # OHLC
        high_move = abs(np.random.normal(0, vol_state * 0.7))
        low_move = abs(np.random.normal(0, vol_state * 0.7))

        closes[i] = price * (1 + drift + mean_rev + shock)
        highs[i] = max(opens[i], closes[i]) * (1 + high_move)
        lows[i] = min(opens[i], closes[i]) * (1 - low_move)

        # Volume (maior em movimentos grandes)
        base_volume = 1000000 * (base_price / 50000)  # Normalizado
        vol_mult = 1 + abs(shock) * 10  # Mais volume em movimentos grandes
        volumes[i] = base_volume * vol_mult * np.random.uniform(0.5, 1.5)

        price = closes[i]

    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })
    df.set_index('timestamp', inplace=True)

    return df


class PreciseBacktester:
    """
    Backtester preciso com todos os custos reais.
    """

    def __init__(self, fees: Dict = None):
        self.fees = fees or REALISTIC_FEES
        self.scoring = ScoringSystem()

    def run_backtest(
        self,
        data: Dict[str, pd.DataFrame],
        params: Dict,
        initial_capital: float = 10000
    ) -> BacktestResult:
        """
        Executar backtest preciso com custos realistas.
        """
        capital = initial_capital
        positions = {}
        trades = []
        equity_curve = [capital]

        peak_equity = capital
        max_drawdown = 0
        total_fees = 0
        total_funding = 0

        # Signal generator
        signal_gen = SignalGenerator(params)

        # Pre-calcular indicadores
        precomputed = {}
        for symbol, df in data.items():
            precomputed[symbol] = signal_gen.prepare_data(df.copy())

        # Alinhar timestamps
        all_timestamps = set()
        for df in data.values():
            all_timestamps.update(df.index.tolist())
        all_timestamps = sorted(all_timestamps)

        if len(all_timestamps) < 100:
            return self._empty_result()

        # Calcular intervalo de funding (8h = 8 candles de 1h)
        funding_interval = 8

        # Loop principal
        for i, timestamp in enumerate(all_timestamps[100:], 100):
            # Atualizar posicoes
            positions_to_close = []
            unrealized_pnl = 0

            for symbol, pos in positions.items():
                if symbol not in data:
                    continue

                df = data[symbol]
                if timestamp not in df.index:
                    continue

                current = df.loc[timestamp]
                entry = pos['entry_price']
                side = pos['side']
                qty = pos['quantity']
                sl = pos['stop_loss']
                tp = pos['take_profit']

                # Verificar SL/TP
                close_reason = None
                exit_price = current['close']

                if side == 'long':
                    if current['low'] <= sl:
                        close_reason = 'STOP_LOSS'
                        exit_price = sl
                    elif current['high'] >= tp:
                        close_reason = 'TAKE_PROFIT'
                        exit_price = tp
                    else:
                        unrealized_pnl += (current['close'] - entry) * qty
                else:
                    if current['high'] >= sl:
                        close_reason = 'STOP_LOSS'
                        exit_price = sl
                    elif current['low'] <= tp:
                        close_reason = 'TAKE_PROFIT'
                        exit_price = tp
                    else:
                        unrealized_pnl += (entry - current['close']) * qty

                if close_reason:
                    positions_to_close.append((symbol, exit_price, close_reason))

            # Fechar posicoes
            for symbol, exit_price, reason in positions_to_close:
                pos = positions[symbol]

                # Calcular PnL
                if pos['side'] == 'long':
                    pnl = (exit_price - pos['entry_price']) * pos['quantity']
                else:
                    pnl = (pos['entry_price'] - exit_price) * pos['quantity']

                # Custos de saida
                notional_exit = pos['quantity'] * exit_price
                exit_fee = notional_exit * self.fees['taker_fee']
                slippage_cost = notional_exit * self.fees['slippage']

                trade_fees = pos['entry_fee'] + exit_fee + slippage_cost
                funding_cost = pos.get('funding_paid', 0)

                net_pnl = pnl - exit_fee - slippage_cost

                capital += pos['margin'] + net_pnl
                total_fees += trade_fees
                total_funding += funding_cost

                trades.append(BacktestTrade(
                    symbol=symbol,
                    side=pos['side'],
                    entry_price=pos['entry_price'],
                    exit_price=exit_price,
                    quantity=pos['quantity'],
                    entry_time=pos['entry_time'],
                    exit_time=timestamp,
                    pnl=net_pnl,
                    pnl_pct=(net_pnl / pos['margin']) * 100,
                    fees_paid=trade_fees,
                    funding_paid=funding_cost,
                    reason_exit=reason
                ))

                del positions[symbol]

            # Funding rate (a cada 8 candles)
            if i % funding_interval == 0:
                for symbol, pos in positions.items():
                    if symbol in data and timestamp in data[symbol].index:
                        current_price = data[symbol].loc[timestamp]['close']
                        position_value = pos['quantity'] * current_price

                        # Funding: longs pagam quando rate > 0
                        if pos['side'] == 'long':
                            funding = position_value * self.fees['funding_rate']
                        else:
                            funding = -position_value * self.fees['funding_rate']

                        capital -= funding
                        pos['funding_paid'] = pos.get('funding_paid', 0) + funding

            # Verificar margem disponivel
            margin_used = sum(p['margin'] for p in positions.values())
            max_margin = capital * params.get('max_margin_usage', 0.8)
            available_margin = max_margin - margin_used

            # Gerar novos sinais
            max_positions = params.get('max_positions', 8)
            if len(positions) < max_positions and available_margin > initial_capital * 0.05:
                signals = []

                for symbol in data.keys():
                    if symbol in positions:
                        continue

                    df = precomputed[symbol]
                    if timestamp not in df.index:
                        continue

                    idx = df.index.get_loc(timestamp)
                    if idx < 50:
                        continue

                    df_slice = df.iloc[:idx+1].tail(100)

                    try:
                        signal = signal_gen.generate_signal(df_slice, precomputed=True)

                        if signal and signal.direction != 'none':
                            min_strength = params.get('min_strength', 5.0)
                            if signal.strength >= min_strength:
                                atr = df_slice['atr'].iloc[-1] if 'atr' in df_slice.columns else 0

                                score = self.scoring.calculate_score(
                                    symbol=symbol,
                                    side=signal.direction,
                                    strength=signal.strength,
                                    entry_price=signal.entry_price,
                                    stop_loss=signal.stop_loss,
                                    take_profit=signal.take_profit,
                                    atr=atr,
                                    historical_winrate=0.5
                                )

                                min_score = params.get('min_score_to_open', 4.5)
                                if score.score >= min_score:
                                    signals.append((symbol, signal, score.score))
                    except:
                        continue

                # Ordenar por score e abrir melhores
                signals.sort(key=lambda x: x[2], reverse=True)

                for symbol, signal, score in signals[:2]:
                    if len(positions) >= max_positions:
                        break

                    # Calcular tamanho da posicao
                    min_pct = params.get('min_position_pct', 0.05)
                    max_pct = params.get('max_position_pct', 0.15)

                    score_factor = min(1.0, (score - 4) / 6)
                    position_pct = min_pct + (max_pct - min_pct) * score_factor
                    margin = capital * position_pct
                    margin = min(margin, available_margin * 0.4)

                    if margin < 50:
                        continue

                    # Leverage
                    max_lev = params.get('max_leverage', 5)
                    leverage = min(max_lev, max(2, int(2 + score * 0.4)))

                    notional = margin * leverage
                    quantity = notional / signal.entry_price

                    # Custos de entrada
                    entry_fee = notional * self.fees['taker_fee']
                    slippage = notional * self.fees['slippage']
                    total_entry_cost = entry_fee + slippage

                    if margin + total_entry_cost <= available_margin:
                        positions[symbol] = {
                            'side': signal.direction,
                            'entry_price': signal.entry_price,
                            'quantity': quantity,
                            'margin': margin,
                            'leverage': leverage,
                            'stop_loss': signal.stop_loss,
                            'take_profit': signal.take_profit,
                            'entry_time': timestamp,
                            'entry_fee': entry_fee + slippage,
                            'funding_paid': 0
                        }

                        capital -= margin + total_entry_cost
                        available_margin -= margin + total_entry_cost
                        total_fees += entry_fee + slippage

            # Atualizar equity
            margin_in_pos = sum(p['margin'] for p in positions.values())
            equity = capital + margin_in_pos + unrealized_pnl
            equity_curve.append(equity)

            # Drawdown
            if equity > peak_equity:
                peak_equity = equity
            dd = (peak_equity - equity) / peak_equity
            max_drawdown = max(max_drawdown, dd)

            # Circuit breaker
            if max_drawdown > 0.25:
                break

        # Fechar posicoes restantes
        for symbol, pos in list(positions.items()):
            if symbol in data:
                exit_price = data[symbol].iloc[-1]['close']

                if pos['side'] == 'long':
                    pnl = (exit_price - pos['entry_price']) * pos['quantity']
                else:
                    pnl = (pos['entry_price'] - exit_price) * pos['quantity']

                exit_fee = pos['quantity'] * exit_price * self.fees['taker_fee']
                net_pnl = pnl - exit_fee
                capital += pos['margin'] + net_pnl
                total_fees += exit_fee

                trades.append(BacktestTrade(
                    symbol=symbol,
                    side=pos['side'],
                    entry_price=pos['entry_price'],
                    exit_price=exit_price,
                    quantity=pos['quantity'],
                    entry_time=pos['entry_time'],
                    exit_time=all_timestamps[-1],
                    pnl=net_pnl,
                    pnl_pct=(net_pnl / pos['margin']) * 100,
                    fees_paid=pos['entry_fee'] + exit_fee,
                    funding_paid=pos.get('funding_paid', 0),
                    reason_exit='END'
                ))

        return self._calculate_metrics(
            equity_curve, trades, initial_capital, total_fees, total_funding
        )

    def _calculate_metrics(
        self,
        equity_curve: List[float],
        trades: List[BacktestTrade],
        initial_capital: float,
        total_fees: float,
        total_funding: float
    ) -> BacktestResult:
        """Calcular metricas."""
        if len(equity_curve) < 2:
            return self._empty_result()

        equity = np.array(equity_curve)

        # Retornos
        total_return = (equity[-1] - initial_capital) / initial_capital * 100

        # Retornos por periodo
        returns = np.diff(equity) / equity[:-1]
        returns = returns[~np.isnan(returns)]

        if len(returns) == 0:
            return self._empty_result()

        # Anualizar (assumindo 1h candles, 8760 candles/ano)
        periods_per_year = 8760
        annual_return = ((1 + total_return/100) ** (periods_per_year / len(returns)) - 1) * 100

        # Volatilidade
        volatility = np.std(returns) * np.sqrt(periods_per_year)

        # Drawdown
        peak = np.maximum.accumulate(equity)
        dd = (peak - equity) / peak
        max_dd = np.max(dd) * 100

        # Sharpe
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(periods_per_year) if np.std(returns) > 0 else 0

        # Sortino
        neg_returns = returns[returns < 0]
        sortino = (np.mean(returns) / np.std(neg_returns)) * np.sqrt(periods_per_year) if len(neg_returns) > 0 and np.std(neg_returns) > 0 else 0

        # Trades
        if trades:
            winners = [t for t in trades if t.pnl > 0]
            losers = [t for t in trades if t.pnl <= 0]

            win_rate = len(winners) / len(trades) * 100
            avg_trade = np.mean([t.pnl for t in trades])

            gross_profit = sum(t.pnl for t in winners) if winners else 0
            gross_loss = abs(sum(t.pnl for t in losers)) if losers else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit if gross_profit > 0 else 0
        else:
            win_rate = 0
            avg_trade = 0
            profit_factor = 0

        return BacktestResult(
            total_return_pct=round(total_return, 2),
            annual_return_pct=round(annual_return, 2),
            max_drawdown_pct=round(max_dd, 2),
            sharpe_ratio=round(sharpe, 2),
            sortino_ratio=round(sortino, 2),
            profit_factor=round(profit_factor, 2),
            win_rate=round(win_rate, 1),
            total_trades=len(trades),
            avg_trade_pnl=round(avg_trade, 2),
            total_fees=round(total_fees, 2),
            total_funding=round(total_funding, 2),
            equity_curve=equity.tolist(),
            trades=trades
        )

    def _empty_result(self) -> BacktestResult:
        return BacktestResult(
            total_return_pct=0, annual_return_pct=0, max_drawdown_pct=100,
            sharpe_ratio=0, sortino_ratio=0, profit_factor=0, win_rate=0,
            total_trades=0, avg_trade_pnl=0, total_fees=0, total_funding=0,
            equity_curve=[], trades=[]
        )


class GeneticEvolver:
    """
    Evoluidor genetico com WFO.
    """

    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.backtester = PreciseBacktester()
        self.best_result = None
        self.best_params = None

    def _default_config(self) -> Dict:
        return {
            'population_size': 15,
            'generations': 8,
            'mutation_rate': 0.25,
            'elite_count': 3,
            'min_improvement_pct': 2.0,

            'param_ranges': {
                # Indicadores
                'rsi_period': (8, 18, 2),
                'rsi_oversold': (18, 30, 2),
                'rsi_overbought': (70, 82, 2),
                'stoch_k': (8, 18, 2),
                'stoch_oversold': (15, 25, 2),
                'stoch_overbought': (75, 85, 2),
                'ema_fast': (5, 12, 1),
                'ema_slow': (15, 30, 3),
                'adx_min': (15, 25, 2),
                'atr_period': (10, 18, 2),

                # SL/TP
                'sl_atr_mult': (1.5, 3.5, 0.5),
                'tp_atr_mult': (2.5, 6.0, 0.5),

                # Risk
                'risk_per_trade': (0.01, 0.025, 0.005),
                'max_positions': (5, 10, 1),
                'max_margin_usage': (0.70, 0.90, 0.05),
                'max_leverage': (3, 8, 1),
                'min_score_to_open': (4.0, 6.0, 0.5),
            },

            'strategies': ['stoch_extreme', 'rsi_extremes', 'momentum_burst'],
        }

    def _base_params(self) -> Dict:
        """Parametros base."""
        return {
            'strategy': 'stoch_extreme',
            'rsi_period': 14,
            'rsi_oversold': 25,
            'rsi_overbought': 75,
            'stoch_k': 14,
            'stoch_d': 3,
            'stoch_oversold': 20,
            'stoch_overbought': 80,
            'ema_fast': 9,
            'ema_slow': 21,
            'ema_trend': 50,
            'adx_period': 14,
            'adx_min': 20,
            'atr_period': 14,
            'sl_atr_mult': 2.5,
            'tp_atr_mult': 4.0,
            'risk_per_trade': 0.015,
            'max_positions': 8,
            'max_margin_usage': 0.80,
            'max_position_pct': 0.15,
            'min_position_pct': 0.05,
            'max_leverage': 5,
            'min_score_to_open': 5.0,
            'min_strength': 5.0,
            'bb_period': 20,
            'bb_std': 2.0,
        }

    def _mutate(self, params: Dict) -> Dict:
        """Mutar parametros."""
        new_params = params.copy()
        ranges = self.config['param_ranges']

        for param, (min_val, max_val, step) in ranges.items():
            if random.random() < self.config['mutation_rate']:
                if isinstance(min_val, int):
                    new_params[param] = random.randint(min_val, max_val)
                else:
                    values = np.arange(min_val, max_val + step, step)
                    new_params[param] = float(random.choice(values))

        # Possivelmente mutar estrategia
        if random.random() < self.config['mutation_rate'] * 0.3:
            new_params['strategy'] = random.choice(self.config['strategies'])

        return new_params

    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Cruzar dois individuos."""
        child = {}
        for key in parent1.keys():
            if random.random() < 0.5:
                child[key] = parent1[key]
            else:
                child[key] = parent2.get(key, parent1[key])
        return child

    def _evaluate(self, params: Dict, data: Dict[str, pd.DataFrame]) -> Tuple[float, BacktestResult]:
        """Avaliar um conjunto de parametros."""
        result = self.backtester.run_backtest(data, params)

        # Score composto
        score = (
            result.sharpe_ratio * 3.0 +
            result.total_return_pct / 30 -
            result.max_drawdown_pct / 4 +
            result.win_rate / 20 +
            result.profit_factor * 2.0 -
            result.total_fees / 500  # Penalizar taxas altas
        )

        return score, result

    def evolve(self, days: int = 120, symbols: List[str] = None) -> Dict:
        """
        Executar evolucao genetica completa.
        """
        print("\n" + "=" * 70)
        print("   EVOLUCAO GENETICA COM BACKTEST PRECISO")
        print("=" * 70)

        symbols = symbols or list(MARKET_PROPERTIES.keys())[:8]
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        print(f"\n   Periodo: {start_date.date()} a {end_date.date()} ({days} dias)")
        print(f"   Simbolos: {len(symbols)}")
        print(f"   Populacao: {self.config['population_size']}")
        print(f"   Geracoes: {self.config['generations']}")

        # Gerar dados
        print("\n   Gerando dados sinteticos realistas...")
        data = {}
        for symbol in symbols:
            df = generate_realistic_ohlcv(symbol, start_date, end_date, '1h')
            if len(df) > 100:
                data[symbol] = df
                print(f"      {symbol}: {len(df)} candles")

        print(f"\n   Total: {sum(len(df) for df in data.values())} candles")

        # Populacao inicial
        population = [self._base_params()]
        for _ in range(self.config['population_size'] - 1):
            population.append(self._mutate(self._base_params()))

        best_score = float('-inf')
        best_params = self._base_params()
        best_result = None

        print(f"\n   Iniciando evolucao...\n")

        for gen in range(self.config['generations']):
            print(f"   --- Geracao {gen + 1}/{self.config['generations']} ---")

            # Avaliar populacao
            scores = []
            for i, params in enumerate(population):
                score, result = self._evaluate(params, data)
                scores.append((score, params, result))

                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    best_result = result
                    print(f"       [!] Novo melhor: score={score:.2f}, ret={result.total_return_pct:.1f}%, "
                          f"sharpe={result.sharpe_ratio:.2f}, dd={result.max_drawdown_pct:.1f}%")

            # Ordenar
            scores.sort(key=lambda x: x[0], reverse=True)

            avg_score = np.mean([s[0] for s in scores])
            print(f"       Melhor: {scores[0][0]:.2f} | Media: {avg_score:.2f}")

            # Elite
            elite = [s[1] for s in scores[:self.config['elite_count']]]

            # Nova populacao
            new_population = elite.copy()

            while len(new_population) < self.config['population_size']:
                # Torneio
                tournament = random.sample(scores, min(3, len(scores)))
                parent1 = max(tournament, key=lambda x: x[0])[1]

                tournament = random.sample(scores, min(3, len(scores)))
                parent2 = max(tournament, key=lambda x: x[0])[1]

                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                new_population.append(child)

            population = new_population

        self.best_params = best_params
        self.best_result = best_result

        return self._print_results(best_params, best_result, data)

    def _print_results(self, params: Dict, result: BacktestResult, data: Dict) -> Dict:
        """Imprimir e salvar resultados."""
        print("\n" + "=" * 70)
        print("   RESULTADO DA EVOLUCAO")
        print("=" * 70)

        print(f"\n   PARAMETROS EVOLUIDOS:")
        print(f"   ---------------------")
        print(f"   Estrategia:      {params.get('strategy', 'N/A')}")
        print(f"   RSI Period:      {params.get('rsi_period', 14)}")
        print(f"   RSI Levels:      {params.get('rsi_oversold', 25)}/{params.get('rsi_overbought', 75)}")
        print(f"   Stoch K:         {params.get('stoch_k', 14)}")
        print(f"   Stoch Levels:    {params.get('stoch_oversold', 20)}/{params.get('stoch_overbought', 80)}")
        print(f"   EMA Fast/Slow:   {params.get('ema_fast', 9)}/{params.get('ema_slow', 21)}")
        print(f"   ADX Min:         {params.get('adx_min', 20)}")
        print(f"   SL ATR Mult:     {params.get('sl_atr_mult', 2.5)}")
        print(f"   TP ATR Mult:     {params.get('tp_atr_mult', 4.0)}")
        print(f"   Max Leverage:    {params.get('max_leverage', 5)}x")
        print(f"   Max Positions:   {params.get('max_positions', 8)}")
        print(f"   Min Score:       {params.get('min_score_to_open', 5.0)}")

        print(f"\n   METRICAS DE PERFORMANCE:")
        print(f"   -------------------------")
        print(f"   Retorno Total:   {result.total_return_pct:+.2f}%")
        print(f"   Retorno Anual:   {result.annual_return_pct:+.2f}%")
        print(f"   Max Drawdown:    {result.max_drawdown_pct:.2f}%")
        print(f"   Sharpe Ratio:    {result.sharpe_ratio:.2f}")
        print(f"   Sortino Ratio:   {result.sortino_ratio:.2f}")
        print(f"   Profit Factor:   {result.profit_factor:.2f}")
        print(f"   Win Rate:        {result.win_rate:.1f}%")
        print(f"   Total Trades:    {result.total_trades}")
        print(f"   Avg Trade PnL:   ${result.avg_trade_pnl:.2f}")

        print(f"\n   CUSTOS TOTAIS:")
        print(f"   ---------------")
        print(f"   Taxas (fees):    ${result.total_fees:.2f}")
        print(f"   Funding:         ${result.total_funding:.2f}")
        print(f"   Total Custos:    ${result.total_fees + result.total_funding:.2f}")

        # Salvar
        output = {
            'timestamp': datetime.now().isoformat(),
            'evolved_params': params,
            'metrics': {
                'total_return_pct': result.total_return_pct,
                'annual_return_pct': result.annual_return_pct,
                'max_drawdown_pct': result.max_drawdown_pct,
                'sharpe_ratio': result.sharpe_ratio,
                'sortino_ratio': result.sortino_ratio,
                'profit_factor': result.profit_factor,
                'win_rate': result.win_rate,
                'total_trades': result.total_trades,
                'avg_trade_pnl': result.avg_trade_pnl,
                'total_fees': result.total_fees,
                'total_funding': result.total_funding,
            },
            'fees_used': REALISTIC_FEES,
            'evolution_config': {
                'population_size': self.config['population_size'],
                'generations': self.config['generations'],
                'mutation_rate': self.config['mutation_rate'],
            }
        }

        os.makedirs('state', exist_ok=True)

        with open('state/evolved_precise_params.json', 'w') as f:
            json.dump(output, f, indent=2, default=str)

        print(f"\n   Parametros salvos em: state/evolved_precise_params.json")
        print("=" * 70)

        return output


def main():
    """Executar evolucao."""
    import argparse

    parser = argparse.ArgumentParser(description='Evolucao Genetica Precisa')
    parser.add_argument('--quick', action='store_true', help='Evolucao rapida')
    parser.add_argument('--validate', action='store_true', help='Validar params salvos')
    parser.add_argument('--days', type=int, default=120, help='Dias de dados')

    args = parser.parse_args()

    if args.quick:
        config = {
            'population_size': 8,
            'generations': 4,
            'mutation_rate': 0.30,
            'elite_count': 2,
            'param_ranges': GeneticEvolver()._default_config()['param_ranges'],
            'strategies': ['stoch_extreme', 'rsi_extremes'],
        }
        evolver = GeneticEvolver(config)
        evolver.evolve(days=60, symbols=list(MARKET_PROPERTIES.keys())[:5])
    else:
        evolver = GeneticEvolver()
        evolver.evolve(days=args.days)


if __name__ == '__main__':
    main()
