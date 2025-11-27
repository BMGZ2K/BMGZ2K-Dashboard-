"""
Estrategia Evoluida com Monitoramento em Tempo Real
====================================================
Versao otimizada com parametros agressivos para maximizar lucro.
Executa em modo simulacao (paper trading) com precos reais.

USO:
    python evolved_realtime.py              # Simulacao em tempo real
    python evolved_realtime.py --backtest   # Backtest dos ultimos 30 dias
"""
import sys
import os
import json
import time
import signal
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import ccxt

from core.config import API_KEY, SECRET_KEY, SYMBOLS, get_validated_params
from core.signals import SignalGenerator, Signal

# =============================================================================
# PARAMETROS EVOLUIDOS - Otimizados para maximo lucro
# =============================================================================
EVOLVED_PARAMS = {
    # Identificacao
    'strategy': 'evolved_aggressive',
    'version': '2.0',

    # RSI mais agressivo
    'rsi_period': 10,
    'rsi_oversold': 22,
    'rsi_overbought': 78,

    # Stochastic mais rapido
    'stoch_k': 10,
    'stoch_d': 3,
    'stoch_oversold': 18,
    'stoch_overbought': 82,

    # EMAs mais responsivas
    'ema_fast': 7,
    'ema_slow': 18,
    'ema_trend': 40,

    # ADX mais sensivel
    'adx_period': 12,
    'adx_min': 18,
    'adx_strong': 22,
    'adx_very_strong': 30,

    # SL/TP otimizados (mais apertados para maior frequencia)
    'atr_period': 12,
    'sl_atr_mult': 2.0,
    'tp_atr_mult': 4.0,

    # Gestao de risco agressiva
    'risk_per_trade': 0.02,
    'max_positions': 8,
    'max_margin_usage': 0.85,
    'max_position_pct': 0.18,
    'min_position_pct': 0.08,
    'max_leverage': 8,
    'use_compounding': True,
    'min_score_to_open': 4.5,

    # Signal strength
    'min_strength': 4.5,
    'base_strength': 5.5,
    'min_signal_strength': 4,
    'max_signal_strength': 10,

    # Bollinger Bands
    'bb_period': 18,
    'bb_std': 1.8,

    # Bias para long (mercado cripto geralmente bullish)
    'long_bias': 1.15,
    'short_penalty': 0.85,
}


@dataclass
class SimulatedPosition:
    """Posicao simulada."""
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
    """Registro de trade."""
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


class EvolvedRealtimeTrader:
    """
    Trader evoluido com monitoramento em tempo real.
    Opera em modo simulacao com precos reais da Binance.
    """

    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.params = EVOLVED_PARAMS.copy()

        # Exchange (somente para dados)
        self.exchange = self._create_exchange()

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

        # Signal generator
        self.signal_gen = SignalGenerator(self.params)

        # Cache de dados
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self.last_update: Dict[str, datetime] = {}

        # Controle
        self.running = False
        self.update_interval = 30  # segundos

        print("=" * 60)
        print("EVOLVED REALTIME TRADER - v2.0")
        print("=" * 60)
        print(f"Capital Inicial: ${initial_capital:,.2f}")
        print(f"Estrategia: {self.params['strategy']}")
        print(f"Max Posicoes: {self.params['max_positions']}")
        print(f"Leverage Max: {self.params['max_leverage']}x")
        print(f"Risk/Trade: {self.params['risk_per_trade']*100:.1f}%")
        print("=" * 60)

    def _create_exchange(self):
        """Criar conexao com exchange."""
        exchange = ccxt.binance({
            'apiKey': API_KEY,
            'secret': SECRET_KEY,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        exchange.set_sandbox_mode(True)  # Sempre testnet para seguranca
        return exchange

    def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
        """Buscar dados OHLCV."""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            print(f"Erro buscando {symbol}: {e}")
            return pd.DataFrame()

    def get_current_price(self, symbol: str) -> float:
        """Obter preco atual."""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last']
        except:
            return 0

    def calculate_position_size(self, price: float, atr: float, score: float) -> tuple:
        """Calcular tamanho da posicao."""
        # Margem baseada no score
        base_margin = self.capital * self.params['min_position_pct']
        max_margin = self.capital * self.params['max_position_pct']

        # Score influencia tamanho (4-10 -> 0.5-1.0 do range)
        score_factor = min(1.0, (score - 4) / 6)
        margin = base_margin + (max_margin - base_margin) * score_factor

        # Verificar margem disponivel
        used_margin = sum(p.margin for p in self.positions.values())
        available = self.capital * self.params['max_margin_usage'] - used_margin
        margin = min(margin, available * 0.4)  # Max 40% do disponivel por trade

        if margin < 50:  # Minimo $50
            return 0, 0

        # Leverage baseado no score
        leverage = min(
            self.params['max_leverage'],
            max(2, int(3 + score * 0.5))
        )

        notional = margin * leverage
        quantity = notional / price

        return margin, quantity

    def check_signals(self) -> List[tuple]:
        """Verificar sinais em todos os simbolos."""
        signals = []

        for symbol in SYMBOLS[:20]:  # Top 20 para velocidade
            if symbol in self.positions:
                continue

            try:
                # Buscar dados
                df = self.fetch_ohlcv(symbol, '1h', 100)
                if len(df) < 50:
                    continue

                # Preparar indicadores
                df = self.signal_gen.prepare_data(df)

                # Gerar sinal
                signal = self.signal_gen.generate_signal(df, precomputed=True)

                if signal and signal.direction != 'none' and signal.strength >= 4.5:
                    atr = df['atr'].iloc[-1] if 'atr' in df.columns else 0

                    # Calcular score simples
                    score = signal.strength
                    if signal.direction == 'long':
                        score *= self.params['long_bias']
                    else:
                        score *= self.params['short_penalty']

                    signals.append((symbol, signal, atr, score))

            except Exception as e:
                continue

        # Ordenar por score
        signals.sort(key=lambda x: x[3], reverse=True)
        return signals[:3]  # Top 3

    def open_position(self, symbol: str, signal: Signal, atr: float, score: float):
        """Abrir posicao simulada."""
        price = signal.entry_price
        margin, quantity = self.calculate_position_size(price, atr, score)

        if margin < 50 or quantity <= 0:
            return False

        leverage = min(self.params['max_leverage'], max(2, int(3 + score * 0.5)))

        # Custos de entrada (0.04% taker + 0.02% slippage)
        entry_cost = margin * leverage * 0.0006

        if margin + entry_cost > self.capital - sum(p.margin for p in self.positions.values()):
            return False

        self.capital -= margin + entry_cost

        position = SimulatedPosition(
            symbol=symbol,
            side=signal.direction,
            entry_price=price,
            quantity=quantity,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            entry_time=datetime.now(),
            margin=margin,
            leverage=leverage,
            current_price=price
        )

        self.positions[symbol] = position

        print(f"\n[OPEN] {signal.direction.upper()} {symbol}")
        print(f"       Entry: ${price:,.4f} | Qty: {quantity:.6f}")
        print(f"       Margin: ${margin:,.2f} | Lev: {leverage}x")
        print(f"       SL: ${signal.stop_loss:,.4f} | TP: ${signal.take_profit:,.4f}")
        print(f"       Score: {score:.1f}")

        return True

    def update_positions(self):
        """Atualizar posicoes e verificar SL/TP."""
        to_close = []

        for symbol, pos in self.positions.items():
            try:
                current = self.get_current_price(symbol)
                if current <= 0:
                    continue

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

                # Verificar fechamento
                if hit_sl:
                    to_close.append((symbol, 'STOP_LOSS'))
                elif hit_tp:
                    to_close.append((symbol, 'TAKE_PROFIT'))

            except Exception as e:
                continue

        # Fechar posicoes
        for symbol, reason in to_close:
            self.close_position(symbol, reason)

    def close_position(self, symbol: str, reason: str):
        """Fechar posicao."""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]

        # Usar preco de SL/TP como exit
        if reason == 'STOP_LOSS':
            exit_price = pos.stop_loss
        elif reason == 'TAKE_PROFIT':
            exit_price = pos.take_profit
        else:
            exit_price = pos.current_price

        # Calcular PnL final
        if pos.side == 'long':
            pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            pnl = (pos.entry_price - exit_price) * pos.quantity

        # Custos de saida
        exit_cost = pos.quantity * exit_price * 0.0006
        pnl -= exit_cost

        pnl_pct = (pnl / pos.margin) * 100

        # Atualizar capital
        self.capital += pos.margin + pnl
        self.total_pnl += pnl

        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        # Registrar trade
        trade = TradeRecord(
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
        )
        self.trades.append(trade)

        # Print
        emoji = "+" if pnl > 0 else ""
        print(f"\n[CLOSE] {pos.side.upper()} {symbol} - {reason}")
        print(f"        Entry: ${pos.entry_price:,.4f} -> Exit: ${exit_price:,.4f}")
        print(f"        PnL: {emoji}${pnl:,.2f} ({emoji}{pnl_pct:.1f}%)")

        del self.positions[symbol]

    def calculate_metrics(self) -> Dict:
        """Calcular metricas atuais."""
        # Equity atual
        unrealized_pnl = sum(p.pnl for p in self.positions.values())
        margin_in_positions = sum(p.margin for p in self.positions.values())
        equity = self.capital + margin_in_positions + unrealized_pnl

        # Drawdown
        if equity > self.peak_equity:
            self.peak_equity = equity

        dd = (self.peak_equity - equity) / self.peak_equity * 100
        self.max_drawdown = max(self.max_drawdown, dd)

        # Win rate
        total_trades = self.winning_trades + self.losing_trades
        win_rate = (self.winning_trades / total_trades * 100) if total_trades > 0 else 0

        # Retorno
        total_return = (equity - self.initial_capital) / self.initial_capital * 100

        return {
            'equity': equity,
            'capital': self.capital,
            'unrealized_pnl': unrealized_pnl,
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
        """Imprimir status atual."""
        metrics = self.calculate_metrics()

        print("\n" + "=" * 60)
        print(f"STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        print(f"Equity:       ${metrics['equity']:,.2f}")
        print(f"Retorno:      {metrics['total_return_pct']:+.2f}%")
        print(f"PnL Realiz:   ${metrics['realized_pnl']:+,.2f}")
        print(f"PnL N-Realiz: ${metrics['unrealized_pnl']:+,.2f}")
        print(f"Drawdown:     {metrics['current_dd_pct']:.2f}% (Max: {metrics['max_drawdown_pct']:.2f}%)")
        print(f"Trades:       {metrics['total_trades']} (W: {metrics['winning_trades']} / L: {metrics['losing_trades']})")
        print(f"Win Rate:     {metrics['win_rate']:.1f}%")
        print(f"Posicoes:     {metrics['open_positions']}/{self.params['max_positions']}")

        if self.positions:
            print("\n--- Posicoes Abertas ---")
            for sym, pos in self.positions.items():
                emoji = "+" if pos.pnl > 0 else ""
                print(f"  {pos.side.upper():5} {sym:12} @ ${pos.entry_price:,.4f} | "
                      f"Atual: ${pos.current_price:,.4f} | PnL: {emoji}${pos.pnl:,.2f} ({emoji}{pos.pnl_pct:.1f}%)")

        print("=" * 60)

        # Salvar estado
        self.save_state()

        return metrics

    def save_state(self):
        """Salvar estado atual."""
        state = {
            'timestamp': datetime.now().isoformat(),
            'params': self.params,
            'metrics': self.calculate_metrics(),
            'positions': [
                {
                    'symbol': p.symbol,
                    'side': p.side,
                    'entry_price': p.entry_price,
                    'quantity': p.quantity,
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
                    'time': t.exit_time.isoformat(),
                }
                for t in self.trades[-20:]
            ]
        }

        try:
            with open('state/evolved_realtime_state.json', 'w') as f:
                json.dump(state, f, indent=2, default=str)
        except:
            pass

    def run(self, duration_hours: float = 24):
        """Executar simulacao em tempo real."""
        print(f"\nIniciando simulacao por {duration_hours} horas...")
        print("Pressione Ctrl+C para parar\n")

        self.running = True
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)

        # Handler para Ctrl+C
        def signal_handler(sig, frame):
            print("\n\nParando simulacao...")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)

        iteration = 0

        while self.running and datetime.now() < end_time:
            iteration += 1

            try:
                # Atualizar posicoes existentes
                self.update_positions()

                # Verificar novos sinais (a cada 2 iteracoes)
                if iteration % 2 == 0 and len(self.positions) < self.params['max_positions']:
                    signals = self.check_signals()

                    for symbol, sig, atr, score in signals:
                        if len(self.positions) >= self.params['max_positions']:
                            break
                        self.open_position(symbol, sig, atr, score)

                # Atualizar equity curve
                metrics = self.calculate_metrics()
                self.equity_curve.append(metrics['equity'])
                self.timestamps.append(datetime.now())

                # Print status a cada 5 iteracoes
                if iteration % 5 == 0:
                    self.print_status()

                # Aguardar
                time.sleep(self.update_interval)

            except Exception as e:
                print(f"Erro na iteracao {iteration}: {e}")
                time.sleep(5)

        # Resumo final
        self.print_final_summary()

    def print_final_summary(self):
        """Imprimir resumo final."""
        metrics = self.calculate_metrics()

        print("\n")
        print("=" * 60)
        print("RESUMO FINAL - ESTRATEGIA EVOLUIDA")
        print("=" * 60)
        print(f"\nCapital Inicial: ${self.initial_capital:,.2f}")
        print(f"Capital Final:   ${metrics['equity']:,.2f}")
        print(f"Lucro/Prejuizo:  ${metrics['equity'] - self.initial_capital:+,.2f}")
        print(f"Retorno Total:   {metrics['total_return_pct']:+.2f}%")
        print(f"\nMax Drawdown:    {metrics['max_drawdown_pct']:.2f}%")
        print(f"Total Trades:    {metrics['total_trades']}")
        print(f"Win Rate:        {metrics['win_rate']:.1f}%")

        if self.trades:
            pnls = [t.pnl for t in self.trades]
            print(f"\nMaior Ganho:     ${max(pnls):+,.2f}")
            print(f"Maior Perda:     ${min(pnls):+,.2f}")
            print(f"Media/Trade:     ${np.mean(pnls):+,.2f}")

        print("\n" + "=" * 60)


def run_backtest(days: int = 30) -> Dict:
    """Executar backtest com parametros evoluidos."""
    from portfolio_wfo import PortfolioBacktester

    print("\n" + "=" * 60)
    print("BACKTEST - ESTRATEGIA EVOLUIDA")
    print("=" * 60)

    backtester = PortfolioBacktester()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    print(f"\nPeriodo: {start_date.date()} a {end_date.date()}")
    print(f"Simbolos: {len(SYMBOLS[:15])}")

    # Buscar dados
    data = backtester.fetch_data(SYMBOLS[:15], '1h', start_date, end_date)
    print(f"Dados carregados: {len(data)} simbolos")

    # Executar backtest
    result = backtester.run_backtest(data, EVOLVED_PARAMS)

    print("\n" + "=" * 60)
    print("RESULTADOS DO BACKTEST")
    print("=" * 60)
    print(f"\nRetorno Total:    {result.total_return_pct:+.2f}%")
    print(f"Retorno Anual:    {result.annual_return_pct:+.2f}%")
    print(f"Sharpe Ratio:     {result.sharpe_ratio:.2f}")
    print(f"Sortino Ratio:    {result.sortino_ratio:.2f}")
    print(f"Max Drawdown:     {result.max_drawdown_pct:.2f}%")
    print(f"Profit Factor:    {result.profit_factor:.2f}")
    print(f"Win Rate:         {result.win_rate:.1f}%")
    print(f"Total Trades:     {result.total_trades}")
    print(f"Avg Trade PnL:    ${result.avg_trade_pnl:.2f}")
    print("=" * 60)

    # Salvar resultados
    output = {
        'timestamp': datetime.now().isoformat(),
        'params': EVOLVED_PARAMS,
        'period': {
            'start': start_date.isoformat(),
            'end': end_date.isoformat(),
            'days': days
        },
        'results': {
            'total_return_pct': result.total_return_pct,
            'annual_return_pct': result.annual_return_pct,
            'sharpe_ratio': result.sharpe_ratio,
            'sortino_ratio': result.sortino_ratio,
            'max_drawdown_pct': result.max_drawdown_pct,
            'profit_factor': result.profit_factor,
            'win_rate': result.win_rate,
            'total_trades': result.total_trades,
        }
    }

    with open('state/evolved_backtest_result.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResultados salvos em: state/evolved_backtest_result.json")

    return output


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Estrategia Evoluida - Trading em Tempo Real')
    parser.add_argument('--backtest', action='store_true', help='Executar backtest')
    parser.add_argument('--days', type=int, default=30, help='Dias para backtest')
    parser.add_argument('--capital', type=float, default=10000, help='Capital inicial')
    parser.add_argument('--hours', type=float, default=24, help='Horas de simulacao')

    args = parser.parse_args()

    if args.backtest:
        run_backtest(args.days)
    else:
        trader = EvolvedRealtimeTrader(initial_capital=args.capital)
        trader.run(duration_hours=args.hours)
