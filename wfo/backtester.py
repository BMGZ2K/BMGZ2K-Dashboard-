"""
Precise Backtester - Simulador de trading com custos reais
==========================================================
Inclui: taxas maker/taker, slippage, funding rate, liquidacao
Suporta taxas dinâmicas da API Binance
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import Config
from core.binance_fees import get_binance_fees, BinanceFees

log = logging.getLogger(__name__)


@dataclass
class Trade:
    """Representa um trade completo."""
    symbol: str
    side: str  # 'long' or 'short'
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    quantity: float = 0
    pnl: float = 0
    pnl_pct: float = 0
    fees: float = 0
    slippage_cost: float = 0
    funding_cost: float = 0
    exit_reason: str = ''
    duration_hours: float = 0
    max_drawdown: float = 0
    max_runup: float = 0


@dataclass
class BacktestResult:
    """Resultado completo do backtest."""
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    avg_trade_pnl: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_trade_duration: float
    total_fees: float
    total_slippage: float
    total_funding: float
    expectancy: float
    recovery_factor: float
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    drawdown_curve: List[float] = field(default_factory=list)


class PreciseBacktester:
    """
    Backtester preciso com custos realistas.

    Features:
    - Taxas maker/taker baseadas em volume
    - Slippage dinamico baseado em volatilidade
    - Funding rate a cada 8 horas
    - Liquidacao forçada
    - Margem e alavancagem realistas
    """

    def __init__(
        self,
        initial_capital: float = None,
        maker_fee: float = None,
        taker_fee: float = None,
        slippage_pct: float = None,
        funding_rate: float = None,
        max_leverage: int = None,
        liquidation_threshold: float = 0.8,  # 80% de perda = liquidacao
        use_dynamic_slippage: bool = True,
        commission_type: str = 'taker',  # 'maker' ou 'taker'
        use_dynamic_fees: bool = False,  # Buscar taxas da API Binance
        symbol: str = None  # Símbolo para buscar taxas dinâmicas
    ):
        # Load from Config with fallback to defaults
        self.initial_capital = initial_capital or Config.get('backtest.initial_capital', 10000)
        self.max_leverage = max_leverage or Config.get('risk.max_leverage', 10)
        self.liquidation_threshold = liquidation_threshold
        self.use_dynamic_slippage = use_dynamic_slippage
        self.commission_type = commission_type
        self.use_dynamic_fees = use_dynamic_fees
        self._binance_fees = None

        # Taxas: usar API dinâmica ou config
        if use_dynamic_fees and symbol:
            self._load_dynamic_fees(symbol)
        else:
            self.maker_fee = maker_fee if maker_fee is not None else Config.get('fees.maker_fee', 0.0002)
            self.taker_fee = taker_fee if taker_fee is not None else Config.get('fees.taker_fee', 0.0004)
            self.slippage_pct = slippage_pct if slippage_pct is not None else Config.get('fees.slippage', 0.0002)
            self.funding_rate = funding_rate if funding_rate is not None else Config.get('fees.funding_rate', 0.0001)

        # State
        self.capital = self.initial_capital
        self.positions: Dict[str, dict] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.high_water_mark = self.initial_capital
        self.max_drawdown = 0

        # WFO Config values
        self._min_data_points = Config.get('wfo.min_data_points', 100)
        self._warmup_period = Config.get('wfo.warmup_period', 50)
        self._window_lookback = Config.get('wfo.window_lookback', 100)
        self._annualization_factor = Config.get('wfo.annualization.periods_per_year_1h', 8760)

    def _load_dynamic_fees(self, symbol: str):
        """
        Carregar taxas dinâmicas da API Binance.

        Args:
            symbol: Símbolo para buscar taxas (ex: 'BTC/USDT' ou 'BTCUSDT')
        """
        try:
            self._binance_fees = get_binance_fees(use_testnet=Config.get('api.use_testnet', True))
            raw_symbol = symbol.replace('/', '')

            # Buscar taxas de comissão
            commission = self._binance_fees.get_commission_rates(raw_symbol)
            self.maker_fee = commission.maker_rate
            self.taker_fee = commission.taker_rate

            # Buscar funding rate atual
            self.funding_rate = self._binance_fees.get_funding_rate(raw_symbol)

            # Slippage ainda vem do config (não tem API)
            self.slippage_pct = Config.get('fees.slippage', 0.0001)

            log.info(f"Fees dinâmicas carregadas para {symbol}: maker={self.maker_fee*100:.4f}%, taker={self.taker_fee*100:.4f}%, funding={self.funding_rate*100:.4f}%")
        except Exception as e:
            log.warning(f"Erro ao carregar fees dinâmicas: {e}. Usando valores do config.")
            self.maker_fee = Config.get('fees.maker_fee', 0.0002)
            self.taker_fee = Config.get('fees.taker_fee', 0.0004)
            self.slippage_pct = Config.get('fees.slippage', 0.0001)
            self.funding_rate = Config.get('fees.funding_rate', 0.0001)

    def get_symbol_fees(self, symbol: str) -> Dict:
        """
        Obter taxas para um símbolo específico (útil para multi-symbol backtest).

        Args:
            symbol: Símbolo (ex: 'BTC/USDT')

        Returns:
            Dict com maker_fee, taker_fee, funding_rate
        """
        if self._binance_fees and self.use_dynamic_fees:
            try:
                raw_symbol = symbol.replace('/', '')
                commission = self._binance_fees.get_commission_rates(raw_symbol)
                funding = self._binance_fees.get_funding_rate(raw_symbol)
                return {
                    'maker_fee': commission.maker_rate,
                    'taker_fee': commission.taker_rate,
                    'funding_rate': funding
                }
            except Exception:
                pass
        return {
            'maker_fee': self.maker_fee,
            'taker_fee': self.taker_fee,
            'funding_rate': self.funding_rate
        }

    def reset(self):
        """Reset backtester state."""
        self.capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = [self.initial_capital]
        self.high_water_mark = self.initial_capital
        self.max_drawdown = 0

    def _get_fee(self) -> float:
        """Retorna taxa baseada no tipo de ordem."""
        return self.maker_fee if self.commission_type == 'maker' else self.taker_fee

    def _calculate_slippage(self, price: float, atr: float = None) -> float:
        """
        Calcula slippage dinamico.

        Em mercados volateis (ATR alto), slippage e maior.
        """
        if self.use_dynamic_slippage and atr is not None:
            # Slippage proporcional a volatilidade
            volatility_factor = min(2.0, atr / price * 100)  # Max 2x
            return price * self.slippage_pct * (1 + volatility_factor)
        return price * self.slippage_pct

    def _calculate_funding(self, position_value: float, hours_held: float) -> float:
        """
        Calcula custo de funding.

        Funding e cobrado a cada 8 horas.
        """
        funding_periods = hours_held / 8
        return position_value * self.funding_rate * funding_periods

    def _check_liquidation(self, position: dict, current_price: float) -> bool:
        """Verifica se posicao deve ser liquidada."""
        entry = position['entry_price']
        leverage = position.get('leverage', self.max_leverage)
        side = position['side']

        if side == 'long':
            pnl_pct = (current_price - entry) / entry * leverage
        else:
            pnl_pct = (entry - current_price) / entry * leverage

        return pnl_pct <= -self.liquidation_threshold

    def open_position(
        self,
        symbol: str,
        side: str,
        price: float,
        capital_pct: float,
        stop_loss: float,
        take_profit: float,
        timestamp: datetime,
        atr: float = None,
        leverage: int = None
    ) -> Optional[Trade]:
        """
        Abre uma nova posicao.

        Args:
            symbol: Par de trading
            side: 'long' ou 'short'
            price: Preco de entrada
            capital_pct: Percentual do capital a usar
            stop_loss: Preco de stop loss
            take_profit: Preco de take profit
            timestamp: Horario de entrada
            atr: ATR para slippage dinamico
            leverage: Alavancagem (default: max_leverage)
        """
        if symbol in self.positions:
            return None  # Ja tem posicao

        leverage = leverage or self.max_leverage

        # Calcula slippage na entrada
        slippage = self._calculate_slippage(price, atr)

        if side == 'long':
            entry_price = price + slippage  # Compra mais caro
        else:
            entry_price = price - slippage  # Vende mais barato

        # Calcula tamanho da posicao
        position_capital = self.capital * capital_pct
        position_value = position_capital * leverage
        quantity = position_value / entry_price

        # Calcula taxa de entrada
        fee = position_value * self._get_fee()

        # Deduz taxa do capital
        self.capital -= fee

        # Registra posicao
        self.positions[symbol] = {
            'side': side,
            'entry_price': entry_price,
            'quantity': quantity,
            'position_value': position_value,
            'margin': position_capital,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_time': timestamp,
            'leverage': leverage,
            'entry_fee': fee,
            'slippage_cost': slippage * quantity
        }

        return Trade(
            symbol=symbol,
            side=side,
            entry_time=timestamp,
            entry_price=entry_price,
            quantity=quantity
        )

    def close_position(
        self,
        symbol: str,
        price: float,
        timestamp: datetime,
        reason: str,
        atr: float = None
    ) -> Optional[Trade]:
        """
        Fecha uma posicao existente.

        Args:
            symbol: Par de trading
            price: Preco de saida
            timestamp: Horario de saida
            reason: Motivo do fechamento
            atr: ATR para slippage dinamico
        """
        if symbol not in self.positions:
            return None

        pos = self.positions[symbol]

        # Calcula slippage na saida
        slippage = self._calculate_slippage(price, atr)

        if pos['side'] == 'long':
            exit_price = price - slippage  # Vende mais barato
            raw_pnl = (exit_price - pos['entry_price']) * pos['quantity']
        else:
            exit_price = price + slippage  # Compra mais caro
            raw_pnl = (pos['entry_price'] - exit_price) * pos['quantity']

        # Calcula taxa de saida
        exit_value = exit_price * pos['quantity']
        exit_fee = exit_value * self._get_fee()

        # Calcula funding
        hours_held = (timestamp - pos['entry_time']).total_seconds() / 3600
        funding_cost = self._calculate_funding(pos['position_value'], hours_held)

        # PnL final
        total_fees = pos['entry_fee'] + exit_fee
        total_costs = total_fees + funding_cost
        net_pnl = raw_pnl - total_costs

        # Atualiza capital
        self.capital += pos['margin'] + net_pnl

        # Cria trade completo
        trade = Trade(
            symbol=symbol,
            side=pos['side'],
            entry_time=pos['entry_time'],
            entry_price=pos['entry_price'],
            exit_time=timestamp,
            exit_price=exit_price,
            quantity=pos['quantity'],
            pnl=net_pnl,
            pnl_pct=net_pnl / pos['margin'] * 100,
            fees=total_fees,
            slippage_cost=pos['slippage_cost'] + slippage * pos['quantity'],
            funding_cost=funding_cost,
            exit_reason=reason,
            duration_hours=hours_held
        )

        self.trades.append(trade)
        del self.positions[symbol]

        return trade

    def update_equity(self, current_prices: Dict[str, float]):
        """
        Atualiza curva de equity com precos atuais.

        Args:
            current_prices: Dict de symbol -> preco atual
        """
        unrealized_pnl = 0

        for symbol, pos in self.positions.items():
            if symbol in current_prices:
                price = current_prices[symbol]
                if pos['side'] == 'long':
                    unrealized_pnl += (price - pos['entry_price']) * pos['quantity']
                else:
                    unrealized_pnl += (pos['entry_price'] - price) * pos['quantity']

        current_equity = self.capital + unrealized_pnl
        self.equity_curve.append(current_equity)

        # Atualiza high water mark e drawdown
        if current_equity > self.high_water_mark:
            self.high_water_mark = current_equity

        drawdown = (self.high_water_mark - current_equity) / self.high_water_mark
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

    def run(
        self,
        data: pd.DataFrame,
        signals_func,
        params: dict,
        risk_per_trade: float = 0.02,
        max_positions: int = 5
    ) -> BacktestResult:
        """
        Executa backtest completo.

        Args:
            data: DataFrame com OHLCV
            signals_func: Funcao que gera sinais (df, params) -> Signal
            params: Parametros da estrategia
            risk_per_trade: Risco por trade (% do capital)
            max_positions: Maximo de posicoes simultaneas

        Returns:
            BacktestResult com todas as metricas
        """
        self.reset()

        if len(data) < self._min_data_points:
            return self._empty_result()

        # Pre-calcula indicadores
        from core.signals import SignalGenerator
        sg = SignalGenerator(params)
        data = sg.prepare_data(data)

        symbol = data.get('symbol', ['UNKNOWN'])[0] if 'symbol' in data.columns else 'UNKNOWN'

        for i in range(self._warmup_period, len(data)):
            row = data.iloc[i]
            timestamp = row.name if isinstance(row.name, datetime) else datetime.now()
            current_price = row['close']
            atr = row.get('atr', current_price * 0.02)

            # Verifica posicoes existentes
            for sym in list(self.positions.keys()):
                pos = self.positions[sym]

                # Check stop loss
                if pos['side'] == 'long':
                    if row['low'] <= pos['stop_loss']:
                        self.close_position(sym, pos['stop_loss'], timestamp, 'STOP_LOSS', atr)
                        continue
                    if row['high'] >= pos['take_profit']:
                        self.close_position(sym, pos['take_profit'], timestamp, 'TAKE_PROFIT', atr)
                        continue
                else:
                    if row['high'] >= pos['stop_loss']:
                        self.close_position(sym, pos['stop_loss'], timestamp, 'STOP_LOSS', atr)
                        continue
                    if row['low'] <= pos['take_profit']:
                        self.close_position(sym, pos['take_profit'], timestamp, 'TAKE_PROFIT', atr)
                        continue

                # Check liquidation
                if self._check_liquidation(pos, current_price):
                    self.close_position(sym, current_price, timestamp, 'LIQUIDATION', atr)

            # Gera sinal
            window = data.iloc[max(0, i-self._window_lookback):i+1]
            signal = sg.generate_signal(window, precomputed=True)

            # Abre nova posicao se tiver sinal
            if signal.direction != 'none' and signal.strength >= params.get('min_signal_strength', 5):
                if len(self.positions) < max_positions and symbol not in self.positions:
                    self.open_position(
                        symbol=symbol,
                        side=signal.direction,
                        price=current_price,
                        capital_pct=risk_per_trade,
                        stop_loss=signal.stop_loss,
                        take_profit=signal.take_profit,
                        timestamp=timestamp,
                        atr=atr
                    )

            # Atualiza equity
            self.update_equity({symbol: current_price})

        # Fecha posicoes abertas no final
        final_row = data.iloc[-1]
        final_price = final_row['close']
        final_time = final_row.name if isinstance(final_row.name, datetime) else datetime.now()

        for sym in list(self.positions.keys()):
            self.close_position(sym, final_price, final_time, 'END_OF_DATA')

        return self._calculate_results()

    def run_multi_symbol(
        self,
        data_dict: Dict[str, pd.DataFrame],
        signals_func,
        params: dict,
        risk_per_trade: float = 0.01,
        max_positions: int = 10
    ) -> BacktestResult:
        """
        Executa backtest com multiplos simbolos.

        Args:
            data_dict: Dict de symbol -> DataFrame
            signals_func: Funcao geradora de sinais
            params: Parametros da estrategia
            risk_per_trade: Risco por trade
            max_positions: Max posicoes totais
        """
        self.reset()

        # Prepara dados
        from core.signals import SignalGenerator
        sg = SignalGenerator(params)

        prepared_data = {}
        for symbol, df in data_dict.items():
            if len(df) >= self._min_data_points:
                prepared_data[symbol] = sg.prepare_data(df.copy())

        if not prepared_data:
            return self._empty_result()

        # Encontra range de datas comum
        all_dates = set()
        for df in prepared_data.values():
            all_dates.update(df.index.tolist())
        all_dates = sorted(all_dates)

        for timestamp in all_dates[self._warmup_period:]:
            current_prices = {}
            signals = []

            for symbol, df in prepared_data.items():
                if timestamp not in df.index:
                    continue

                idx = df.index.get_loc(timestamp)
                if idx < self._warmup_period:
                    continue

                row = df.iloc[idx]
                current_prices[symbol] = row['close']
                atr = row.get('atr', row['close'] * 0.02)

                # Verifica posicao existente
                if symbol in self.positions:
                    pos = self.positions[symbol]

                    if pos['side'] == 'long':
                        if row['low'] <= pos['stop_loss']:
                            self.close_position(symbol, pos['stop_loss'], timestamp, 'STOP_LOSS', atr)
                            continue
                        if row['high'] >= pos['take_profit']:
                            self.close_position(symbol, pos['take_profit'], timestamp, 'TAKE_PROFIT', atr)
                            continue
                    else:
                        if row['high'] >= pos['stop_loss']:
                            self.close_position(symbol, pos['stop_loss'], timestamp, 'STOP_LOSS', atr)
                            continue
                        if row['low'] <= pos['take_profit']:
                            self.close_position(symbol, pos['take_profit'], timestamp, 'TAKE_PROFIT', atr)
                            continue

                    if self._check_liquidation(pos, row['close']):
                        self.close_position(symbol, row['close'], timestamp, 'LIQUIDATION', atr)
                        continue

                # Gera sinal
                window = df.iloc[max(0, idx-self._window_lookback):idx+1]
                signal = sg.generate_signal(window, precomputed=True)

                if signal.direction != 'none' and signal.strength >= params.get('min_signal_strength', 5):
                    signals.append((symbol, signal, atr))

            # Ordena sinais por strength e abre posicoes
            signals.sort(key=lambda x: x[1].strength, reverse=True)

            for symbol, signal, atr in signals:
                if len(self.positions) >= max_positions:
                    break
                if symbol in self.positions:
                    continue

                self.open_position(
                    symbol=symbol,
                    side=signal.direction,
                    price=signal.entry_price,
                    capital_pct=risk_per_trade,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    timestamp=timestamp,
                    atr=atr
                )

            # Atualiza equity
            self.update_equity(current_prices)

        # Fecha posicoes abertas
        for symbol in list(self.positions.keys()):
            if symbol in current_prices:
                self.close_position(symbol, current_prices[symbol], all_dates[-1], 'END_OF_DATA')

        return self._calculate_results()

    def _calculate_results(self) -> BacktestResult:
        """Calcula metricas finais do backtest."""
        if not self.trades:
            return self._empty_result()

        # Metricas basicas
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]

        total_wins = sum(t.pnl for t in wins) if wins else 0
        total_losses = abs(sum(t.pnl for t in losses)) if losses else 0

        win_rate = len(wins) / len(self.trades) * 100 if self.trades else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        avg_win = total_wins / len(wins) if wins else 0
        avg_loss = total_losses / len(losses) if losses else 0

        # Sharpe Ratio
        returns = []
        for i in range(1, len(self.equity_curve)):
            ret = (self.equity_curve[i] - self.equity_curve[i-1]) / self.equity_curve[i-1]
            returns.append(ret)

        if returns:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe = (avg_return / std_return) * np.sqrt(self._annualization_factor) if std_return > 0 else 0
        else:
            sharpe = 0

        # Sortino Ratio (so considera retornos negativos)
        neg_returns = [r for r in returns if r < 0]
        if neg_returns:
            downside_std = np.std(neg_returns)
            sortino = (avg_return / downside_std) * np.sqrt(self._annualization_factor) if downside_std > 0 else 0
        else:
            sortino = sharpe

        # Calmar Ratio
        total_return = (self.capital - self.initial_capital) / self.initial_capital
        calmar = total_return / self.max_drawdown if self.max_drawdown > 0 else float('inf')

        # Expectancy
        expectancy = (win_rate/100 * avg_win) - ((1 - win_rate/100) * avg_loss)

        # Recovery Factor
        recovery_factor = total_return / self.max_drawdown if self.max_drawdown > 0 else float('inf')

        # Custos totais
        total_fees = sum(t.fees for t in self.trades)
        total_slippage = sum(t.slippage_cost for t in self.trades)
        total_funding = sum(t.funding_cost for t in self.trades)

        # Drawdown curve
        drawdown_curve = []
        hwm = self.equity_curve[0]
        for eq in self.equity_curve:
            if eq > hwm:
                hwm = eq
            dd = (hwm - eq) / hwm
            drawdown_curve.append(dd)

        return BacktestResult(
            initial_capital=self.initial_capital,
            final_capital=self.capital,
            total_return=self.capital - self.initial_capital,
            total_return_pct=total_return * 100,
            total_trades=len(self.trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=self.max_drawdown * 100,
            max_drawdown_pct=self.max_drawdown * 100,
            avg_trade_pnl=np.mean([t.pnl for t in self.trades]),
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=max(t.pnl for t in self.trades) if self.trades else 0,
            largest_loss=min(t.pnl for t in self.trades) if self.trades else 0,
            avg_trade_duration=np.mean([t.duration_hours for t in self.trades]),
            total_fees=total_fees,
            total_slippage=total_slippage,
            total_funding=total_funding,
            expectancy=expectancy,
            recovery_factor=recovery_factor,
            trades=self.trades,
            equity_curve=self.equity_curve,
            drawdown_curve=drawdown_curve
        )

    def _empty_result(self) -> BacktestResult:
        """Retorna resultado vazio."""
        return BacktestResult(
            initial_capital=self.initial_capital,
            final_capital=self.initial_capital,
            total_return=0,
            total_return_pct=0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0,
            profit_factor=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            calmar_ratio=0,
            max_drawdown=0,
            max_drawdown_pct=0,
            avg_trade_pnl=0,
            avg_win=0,
            avg_loss=0,
            largest_win=0,
            largest_loss=0,
            avg_trade_duration=0,
            total_fees=0,
            total_slippage=0,
            total_funding=0,
            expectancy=0,
            recovery_factor=0,
            trades=[],
            equity_curve=[self.initial_capital],
            drawdown_curve=[0]
        )
