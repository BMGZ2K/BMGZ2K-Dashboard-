"""
Simulacao de Estrategia Evoluida com Dados Sinteticos
=====================================================
Demonstra o funcionamento da estrategia evoluida usando
dados de mercado simulados com caracteristicas realistas.

USO:
    python evolved_simulation.py
"""
import sys
import os
import json
import time
import random
import signal
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

# =============================================================================
# PARAMETROS EVOLUIDOS - Otimizados para maximo lucro
# =============================================================================
EVOLVED_PARAMS = {
    'strategy': 'evolved_aggressive_v2',
    'version': '2.0',

    # RSI mais agressivo
    'rsi_period': 10,
    'rsi_oversold': 22,
    'rsi_overbought': 78,

    # SL/TP otimizados
    'sl_pct': 0.018,  # 1.8% stop loss
    'tp_pct': 0.042,  # 4.2% take profit (2.33 R:R)

    # Gestao de risco
    'risk_per_trade': 0.02,
    'max_positions': 8,
    'max_margin_usage': 0.85,
    'max_position_pct': 0.18,
    'min_position_pct': 0.08,
    'max_leverage': 8,

    # Win rate esperado baseado em backtests
    'expected_win_rate': 0.48,

    # Bias
    'long_bias': 1.15,
    'short_bias': 0.85,
}

# Simbolos simulados
SIMULATED_SYMBOLS = [
    {'symbol': 'BTC/USDT', 'price': 95000, 'volatility': 0.025, 'trend': 0.0002},
    {'symbol': 'ETH/USDT', 'price': 3500, 'volatility': 0.030, 'trend': 0.0003},
    {'symbol': 'SOL/USDT', 'price': 240, 'volatility': 0.040, 'trend': 0.0004},
    {'symbol': 'BNB/USDT', 'price': 650, 'volatility': 0.025, 'trend': 0.0001},
    {'symbol': 'DOGE/USDT', 'price': 0.42, 'volatility': 0.045, 'trend': 0.0003},
    {'symbol': 'XRP/USDT', 'price': 1.45, 'volatility': 0.035, 'trend': 0.0002},
    {'symbol': 'ADA/USDT', 'price': 1.05, 'volatility': 0.038, 'trend': 0.0001},
    {'symbol': 'AVAX/USDT', 'price': 42, 'volatility': 0.042, 'trend': 0.0003},
    {'symbol': 'DOT/USDT', 'price': 9.5, 'volatility': 0.035, 'trend': 0.0001},
    {'symbol': 'LINK/USDT', 'price': 18, 'volatility': 0.038, 'trend': 0.0002},
]


@dataclass
class SimulatedPosition:
    symbol: str
    side: str
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    entry_time: datetime
    margin: float
    leverage: int
    current_price: float = 0
    pnl: float = 0
    pnl_pct: float = 0


@dataclass
class TradeRecord:
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    entry_time: datetime
    exit_time: datetime
    reason: str


class MarketSimulator:
    """Simulador de mercado realista."""

    def __init__(self):
        self.prices: Dict[str, float] = {}
        self.last_update = datetime.now()

        # Inicializar precos
        for sym_data in SIMULATED_SYMBOLS:
            self.prices[sym_data['symbol']] = sym_data['price']

    def update_prices(self, elapsed_seconds: float = 30):
        """Atualizar precos com movimento realista."""
        for sym_data in SIMULATED_SYMBOLS:
            symbol = sym_data['symbol']
            volatility = sym_data['volatility']
            trend = sym_data['trend']

            # Movimento browniano com drift
            dt = elapsed_seconds / 3600  # Converter para horas
            drift = trend * dt
            shock = volatility * np.sqrt(dt) * np.random.normal()

            # Aplicar mudanca
            change = drift + shock
            self.prices[symbol] *= (1 + change)

        self.last_update = datetime.now()

    def get_price(self, symbol: str) -> float:
        return self.prices.get(symbol, 0)

    def get_all_prices(self) -> Dict[str, float]:
        return self.prices.copy()


class SignalSimulator:
    """Simulador de sinais baseado em probabilidades."""

    def __init__(self, params: Dict):
        self.params = params
        self.last_signals: Dict[str, datetime] = {}

    def generate_signals(self, prices: Dict[str, float]) -> List[Dict]:
        """Gerar sinais simulados."""
        signals = []

        for sym_data in SIMULATED_SYMBOLS:
            symbol = sym_data['symbol']

            # Cooldown de 2 horas entre sinais do mesmo simbolo
            if symbol in self.last_signals:
                elapsed = (datetime.now() - self.last_signals[symbol]).seconds
                if elapsed < 7200:  # 2 horas
                    continue

            # Probabilidade base de sinal (5% por ciclo)
            if random.random() > 0.05:
                continue

            price = prices[symbol]

            # Decidir direcao (long tem bias)
            if random.random() < 0.55 * self.params['long_bias']:
                side = 'long'
                sl = price * (1 - self.params['sl_pct'])
                tp = price * (1 + self.params['tp_pct'])
            else:
                side = 'short'
                sl = price * (1 + self.params['sl_pct'])
                tp = price * (1 - self.params['tp_pct'])

            # Score baseado em volatilidade
            score = 5.0 + random.uniform(-1, 3)

            signals.append({
                'symbol': symbol,
                'side': side,
                'price': price,
                'stop_loss': sl,
                'take_profit': tp,
                'score': score,
            })

            self.last_signals[symbol] = datetime.now()

        return signals


class EvolvedSimulationTrader:
    """Trader de simulacao com estrategia evoluida."""

    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.params = EVOLVED_PARAMS.copy()

        # Simuladores
        self.market = MarketSimulator()
        self.signals = SignalSimulator(self.params)

        # Estado
        self.positions: Dict[str, SimulatedPosition] = {}
        self.trades: List[TradeRecord] = []
        self.equity_curve: List[float] = [initial_capital]
        self.timestamps: List[datetime] = [datetime.now()]

        # Metricas
        self.peak_equity = initial_capital
        self.max_drawdown = 0
        self.total_pnl = 0
        self.winning_trades = 0
        self.losing_trades = 0

        # Controle
        self.running = False
        self.iteration = 0

        self._print_header()

    def _print_header(self):
        print("\n" + "=" * 70)
        print("   EVOLVED STRATEGY SIMULATOR - Real-Time Profit Tracking")
        print("=" * 70)
        print(f"   Capital Inicial:  ${self.initial_capital:,.2f}")
        print(f"   Estrategia:       {self.params['strategy']}")
        print(f"   Max Posicoes:     {self.params['max_positions']}")
        print(f"   Leverage Max:     {self.params['max_leverage']}x")
        print(f"   Stop Loss:        {self.params['sl_pct']*100:.1f}%")
        print(f"   Take Profit:      {self.params['tp_pct']*100:.1f}%")
        print(f"   Risk/Reward:      1:{self.params['tp_pct']/self.params['sl_pct']:.2f}")
        print("=" * 70)

    def calculate_position_size(self, price: float, score: float) -> tuple:
        """Calcular tamanho da posicao."""
        base_pct = self.params['min_position_pct']
        max_pct = self.params['max_position_pct']

        # Score influencia (4-8 -> base a max)
        score_factor = min(1.0, max(0, (score - 4) / 4))
        position_pct = base_pct + (max_pct - base_pct) * score_factor

        margin = self.capital * position_pct

        # Verificar disponibilidade
        used_margin = sum(p.margin for p in self.positions.values())
        max_available = self.capital * self.params['max_margin_usage'] - used_margin
        margin = min(margin, max_available * 0.35)

        if margin < 50:
            return 0, 0, 0

        # Leverage
        leverage = min(self.params['max_leverage'], max(2, int(2 + score * 0.8)))
        notional = margin * leverage
        quantity = notional / price

        return margin, quantity, leverage

    def open_position(self, signal: Dict) -> bool:
        """Abrir posicao."""
        symbol = signal['symbol']
        if symbol in self.positions:
            return False

        margin, quantity, leverage = self.calculate_position_size(
            signal['price'], signal['score']
        )

        if margin < 50:
            return False

        # Custo de entrada (0.04% + slippage)
        entry_cost = margin * leverage * 0.0006

        if margin + entry_cost > self.capital - sum(p.margin for p in self.positions.values()):
            return False

        self.capital -= margin + entry_cost

        position = SimulatedPosition(
            symbol=symbol,
            side=signal['side'],
            entry_price=signal['price'],
            quantity=quantity,
            stop_loss=signal['stop_loss'],
            take_profit=signal['take_profit'],
            entry_time=datetime.now(),
            margin=margin,
            leverage=leverage,
            current_price=signal['price']
        )

        self.positions[symbol] = position

        print(f"\n  [OPEN] {signal['side'].upper():5} {symbol:12}")
        print(f"         Entry: ${signal['price']:,.4f} | Margin: ${margin:,.2f} | Lev: {leverage}x")
        print(f"         SL: ${signal['stop_loss']:,.4f} | TP: ${signal['take_profit']:,.4f}")

        return True

    def update_positions(self):
        """Atualizar posicoes."""
        to_close = []

        for symbol, pos in self.positions.items():
            current = self.market.get_price(symbol)
            pos.current_price = current

            # Calcular PnL
            if pos.side == 'long':
                pnl = (current - pos.entry_price) * pos.quantity
                hit_sl = current <= pos.stop_loss
                hit_tp = current >= pos.take_profit
            else:
                pnl = (pos.entry_price - current) * pos.quantity
                hit_sl = current >= pos.stop_loss
                hit_tp = current <= pos.take_profit

            pos.pnl = pnl
            pos.pnl_pct = (pnl / pos.margin) * 100

            if hit_sl:
                to_close.append((symbol, 'STOP_LOSS'))
            elif hit_tp:
                to_close.append((symbol, 'TAKE_PROFIT'))

        for symbol, reason in to_close:
            self.close_position(symbol, reason)

    def close_position(self, symbol: str, reason: str):
        """Fechar posicao."""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]

        # Exit price
        if reason == 'STOP_LOSS':
            exit_price = pos.stop_loss
        elif reason == 'TAKE_PROFIT':
            exit_price = pos.take_profit
        else:
            exit_price = pos.current_price

        # PnL
        if pos.side == 'long':
            pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            pnl = (pos.entry_price - exit_price) * pos.quantity

        # Custos
        exit_cost = pos.quantity * exit_price * 0.0006
        pnl -= exit_cost

        pnl_pct = (pnl / pos.margin) * 100

        # Atualizar estado
        self.capital += pos.margin + pnl
        self.total_pnl += pnl

        if pnl > 0:
            self.winning_trades += 1
            emoji = "+"
        else:
            self.losing_trades += 1
            emoji = ""

        # Registrar
        self.trades.append(TradeRecord(
            symbol=symbol,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=pos.quantity,
            pnl=pnl,
            pnl_pct=pnl_pct,
            entry_time=pos.entry_time,
            exit_time=datetime.now(),
            reason=reason
        ))

        print(f"\n  [CLOSE] {pos.side.upper():5} {symbol:12} - {reason}")
        print(f"          Entry: ${pos.entry_price:,.4f} -> Exit: ${exit_price:,.4f}")
        print(f"          PnL: {emoji}${pnl:,.2f} ({emoji}{pnl_pct:.1f}%)")

        del self.positions[symbol]

    def calculate_metrics(self) -> Dict:
        """Calcular metricas."""
        unrealized = sum(p.pnl for p in self.positions.values())
        margin_in = sum(p.margin for p in self.positions.values())
        equity = self.capital + margin_in + unrealized

        if equity > self.peak_equity:
            self.peak_equity = equity

        dd = (self.peak_equity - equity) / self.peak_equity * 100
        self.max_drawdown = max(self.max_drawdown, dd)

        total_trades = self.winning_trades + self.losing_trades
        win_rate = (self.winning_trades / total_trades * 100) if total_trades > 0 else 0
        total_return = (equity - self.initial_capital) / self.initial_capital * 100

        return {
            'equity': equity,
            'capital': self.capital,
            'unrealized_pnl': unrealized,
            'realized_pnl': self.total_pnl,
            'total_return_pct': total_return,
            'max_drawdown_pct': self.max_drawdown,
            'current_dd_pct': dd,
            'total_trades': total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'open_positions': len(self.positions),
        }

    def print_status(self):
        """Imprimir status."""
        m = self.calculate_metrics()

        # Calcular lucro/hora projetado
        elapsed = (datetime.now() - self.timestamps[0]).total_seconds() / 3600
        if elapsed > 0 and m['total_trades'] > 0:
            profit_per_hour = m['realized_pnl'] / elapsed
        else:
            profit_per_hour = 0

        print("\n" + "-" * 70)
        print(f"  STATUS | {datetime.now().strftime('%H:%M:%S')} | Iteracao {self.iteration}")
        print("-" * 70)
        print(f"  Equity:      ${m['equity']:,.2f}  |  Retorno: {m['total_return_pct']:+.2f}%")
        print(f"  Realizado:   ${m['realized_pnl']:+,.2f}  |  Nao-Realiz: ${m['unrealized_pnl']:+,.2f}")
        print(f"  Drawdown:    {m['current_dd_pct']:.2f}%  |  Max DD: {m['max_drawdown_pct']:.2f}%")
        print(f"  Trades:      {m['total_trades']} (W:{m['winning_trades']} L:{m['losing_trades']})  |  WinRate: {m['win_rate']:.1f}%")
        print(f"  Posicoes:    {m['open_positions']}/{self.params['max_positions']}")
        print(f"  Proj/Hora:   ${profit_per_hour:+,.2f}")

        if self.positions:
            print("\n  Posicoes Abertas:")
            for sym, pos in self.positions.items():
                sign = "+" if pos.pnl > 0 else ""
                print(f"    {pos.side.upper():5} {sym:12} @ ${pos.entry_price:,.4f} | "
                      f"Atual: ${pos.current_price:,.4f} | PnL: {sign}${pos.pnl:,.2f}")

        print("-" * 70)

        # Atualizar equity curve
        self.equity_curve.append(m['equity'])
        self.timestamps.append(datetime.now())

        # Salvar estado
        self._save_state(m)

        return m

    def _save_state(self, metrics: Dict):
        """Salvar estado."""
        state = {
            'timestamp': datetime.now().isoformat(),
            'params': self.params,
            'metrics': metrics,
            'positions': [
                {
                    'symbol': p.symbol,
                    'side': p.side,
                    'entry_price': p.entry_price,
                    'current_price': p.current_price,
                    'pnl': p.pnl,
                    'pnl_pct': p.pnl_pct,
                }
                for p in self.positions.values()
            ],
            'recent_trades': [
                {
                    'symbol': t.symbol,
                    'side': t.side,
                    'pnl': t.pnl,
                    'pnl_pct': t.pnl_pct,
                    'reason': t.reason,
                }
                for t in self.trades[-10:]
            ],
            'equity_curve': self.equity_curve[-100:],
        }

        try:
            os.makedirs('state', exist_ok=True)
            with open('state/evolved_simulation_state.json', 'w') as f:
                json.dump(state, f, indent=2, default=str)
        except:
            pass

    def run(self, iterations: int = 500, interval: float = 0.5):
        """Executar simulacao."""
        print(f"\n  Iniciando simulacao com {iterations} iteracoes...")
        print("  Pressione Ctrl+C para parar\n")

        self.running = True

        def signal_handler(sig, frame):
            print("\n\n  Parando simulacao...")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)

        for i in range(iterations):
            if not self.running:
                break

            self.iteration = i + 1

            try:
                # Atualizar mercado (simular 30 minutos por iteracao)
                self.market.update_prices(elapsed_seconds=1800)

                # Atualizar posicoes
                self.update_positions()

                # Verificar novos sinais
                if len(self.positions) < self.params['max_positions']:
                    prices = self.market.get_all_prices()
                    new_signals = self.signals.generate_signals(prices)

                    for sig in new_signals[:2]:
                        if len(self.positions) >= self.params['max_positions']:
                            break
                        self.open_position(sig)

                # Print status a cada 20 iteracoes
                if i % 20 == 0:
                    self.print_status()

                time.sleep(interval)

            except Exception as e:
                print(f"  Erro na iteracao {i}: {e}")

        self._print_summary()

    def _print_summary(self):
        """Resumo final."""
        m = self.calculate_metrics()

        print("\n")
        print("=" * 70)
        print("   RESUMO FINAL - ESTRATEGIA EVOLUIDA")
        print("=" * 70)
        print(f"\n   Capital Inicial:   ${self.initial_capital:,.2f}")
        print(f"   Capital Final:     ${m['equity']:,.2f}")
        print(f"   Lucro/Prejuizo:    ${m['equity'] - self.initial_capital:+,.2f}")
        print(f"   Retorno Total:     {m['total_return_pct']:+.2f}%")
        print(f"\n   Max Drawdown:      {m['max_drawdown_pct']:.2f}%")
        print(f"   Total Trades:      {m['total_trades']}")
        print(f"   Win Rate:          {m['win_rate']:.1f}%")

        if self.trades:
            pnls = [t.pnl for t in self.trades]
            wins = [t.pnl for t in self.trades if t.pnl > 0]
            losses = [t.pnl for t in self.trades if t.pnl < 0]

            print(f"\n   Maior Ganho:       ${max(pnls):+,.2f}")
            print(f"   Maior Perda:       ${min(pnls):+,.2f}")
            print(f"   Media/Trade:       ${np.mean(pnls):+,.2f}")

            if wins and losses:
                profit_factor = abs(sum(wins) / sum(losses))
                print(f"   Profit Factor:     {profit_factor:.2f}")

        print("\n" + "=" * 70)

        # Calcular metricas de tempo
        elapsed = (self.timestamps[-1] - self.timestamps[0]).total_seconds()
        hours = elapsed / 3600
        if hours > 0:
            print(f"\n   Tempo simulado:    {hours:.1f} horas ({hours*30:.0f} horas de mercado)")
            if m['total_trades'] > 0:
                print(f"   Lucro/Hora (sim):  ${m['realized_pnl'] / hours:+,.2f}")
                print(f"   Lucro/Hora (proj): ${m['realized_pnl'] / (hours * 30):+,.2f}")

        print("\n" + "=" * 70)


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("   INICIANDO SIMULACAO DE ESTRATEGIA EVOLUIDA")
    print("=" * 70)

    trader = EvolvedSimulationTrader(initial_capital=10000)
    trader.run(iterations=300, interval=0.3)
