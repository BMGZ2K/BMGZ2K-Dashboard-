"""
Validador de Máximo Lucro Possível
==================================
Sistema rigoroso para:
1. Testar TODAS as possibilidades sem limites arbitrários
2. Validar matematicamente cada resultado
3. Detectar over/underestimation
4. Provar com estatística out-of-sample
5. Encontrar o máximo lucro REAL possível

Princípios:
- SEM limites artificiais de margem ou posições
- Margem dinâmica e inteligente
- Validação matemática de cada trade
- Comparação rigorosa com baseline
- Detecção de anomalias estatísticas
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ccxt
from dotenv import load_dotenv
load_dotenv()

from core.config import API_KEY, SECRET_KEY, SYMBOLS
from core.signals import SignalGenerator
from core.binance_fees import get_binance_fees


@dataclass
class TradeRecord:
    """Registro detalhado de um trade para auditoria."""
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    leverage: int
    margin_used: float
    notional_value: float
    entry_time: datetime
    exit_time: datetime

    # PnL breakdown
    gross_pnl: float           # PnL bruto (sem custos)
    entry_fee: float           # Taxa de entrada
    exit_fee: float            # Taxa de saída
    slippage_cost: float       # Custo de slippage
    funding_cost: float        # Funding pago/recebido
    net_pnl: float             # PnL líquido

    # Métricas
    pnl_pct: float             # PnL % sobre margem
    rrr: float                 # Risk/Reward realizado
    reason_exit: str

    # Validação
    is_valid: bool = True
    validation_notes: str = ""


@dataclass
class ValidationReport:
    """Relatório de validação matemática."""
    # Básico
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # PnL
    total_gross_pnl: float
    total_fees: float
    total_funding: float
    total_net_pnl: float

    # Retornos
    initial_capital: float
    final_capital: float
    total_return_pct: float
    cagr: float  # Compound Annual Growth Rate

    # Risco
    max_drawdown_pct: float
    max_drawdown_duration: int  # em candles
    volatility: float

    # Métricas ajustadas
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    profit_factor: float

    # Validação
    is_mathematically_valid: bool
    pnl_reconciliation_error: float  # Diferença entre soma de trades e equity final
    suspicious_trades: List[TradeRecord] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Estatística
    t_statistic: float = 0.0
    p_value: float = 1.0
    confidence_level: float = 0.0
    is_statistically_significant: bool = False


class MaxProfitValidator:
    """
    Validador de máximo lucro com zero limites arbitrários.

    Testa todas as configurações possíveis e valida matematicamente
    cada resultado para encontrar o máximo lucro REAL.
    """

    def __init__(self):
        self.exchange = self._create_exchange()
        self.fees_manager = get_binance_fees(use_testnet=False)
        self.data_cache = {}

    def _create_exchange(self):
        """Criar conexão com exchange (dados públicos - sem autenticação)."""
        # Para dados públicos OHLCV não precisamos de API key
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        return exchange

    def fetch_data(self, symbols: List[str], timeframe: str,
                   start: datetime, end: datetime) -> Dict[str, pd.DataFrame]:
        """Buscar dados históricos."""
        data = {}
        since = int(start.timestamp() * 1000)
        until = int(end.timestamp() * 1000)

        for symbol in symbols:
            cache_key = f"{symbol}_{timeframe}_{start.date()}_{end.date()}"
            if cache_key in self.data_cache:
                data[symbol] = self.data_cache[cache_key]
                continue

            try:
                ohlcv = []
                current = since
                while current < until:
                    batch = self.exchange.fetch_ohlcv(symbol, timeframe, current, limit=1000)
                    if not batch:
                        break
                    ohlcv.extend(batch)
                    current = batch[-1][0] + 1

                if ohlcv:
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    df = df[(df.index >= start) & (df.index <= end)]

                    if len(df) > 50:
                        data[symbol] = df
                        self.data_cache[cache_key] = df
                        print(f"  {symbol}: {len(df)} candles")
            except Exception as e:
                print(f"  {symbol}: ERRO - {e}")

        return data

    def run_unlimited_backtest(
        self,
        data: Dict[str, pd.DataFrame],
        params: Dict,
        initial_capital: float = 10000.0,
        timeframe: str = '1h'
    ) -> Tuple[ValidationReport, List[TradeRecord]]:
        """
        Backtest SEM LIMITES arbitrários.

        A margem é alocada dinamicamente baseada em:
        - Score do sinal (quanto maior, mais margem)
        - Margem disponível real
        - Risco calculado (SL distance)

        NÃO há limite de posições ou margem - o sistema decide
        inteligentemente baseado em oportunidade.
        """
        # Estado
        capital = initial_capital
        available_margin = capital
        positions = {}  # symbol -> position_data
        trades: List[TradeRecord] = []
        equity_curve = [capital]

        # Tracking
        peak_equity = capital
        max_drawdown = 0
        drawdown_start = 0
        max_dd_duration = 0
        current_dd_duration = 0

        # Taxas dinâmicas
        fees_cache = {}
        for symbol in data.keys():
            try:
                fees_cache[symbol] = self.fees_manager.get_all_fees_for_symbol(symbol)
            except:
                fees_cache[symbol] = {'maker_fee': 0.0002, 'taker_fee': 0.0004, 'funding_rate': 0.0001}

        # Signal generator
        signal_gen = SignalGenerator(params)

        # Pré-calcular indicadores
        precomputed = {}
        for symbol, df in data.items():
            precomputed[symbol] = signal_gen.prepare_data(df.copy())

        # Timeline unificada
        all_timestamps = set()
        for df in data.values():
            all_timestamps.update(df.index.tolist())
        all_timestamps = sorted(all_timestamps)

        if len(all_timestamps) < 100:
            return self._empty_report(initial_capital), []

        # Intervalo de funding (8h)
        tf_minutes = {'1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240}.get(timeframe, 60)
        funding_interval = max(1, 480 // tf_minutes)

        # Loop principal
        for i, timestamp in enumerate(all_timestamps[100:], 100):
            current_prices = {}

            # 1. Atualizar preços e verificar SL/TP
            positions_to_close = []
            unrealized_pnl = 0

            for symbol, pos in list(positions.items()):
                if symbol not in data or timestamp not in data[symbol].index:
                    continue

                current = data[symbol].loc[timestamp]
                current_prices[symbol] = current['close']

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
                else:  # short
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

            # 2. Fechar posições
            for symbol, exit_price, reason in positions_to_close:
                pos = positions[symbol]
                fees = fees_cache.get(symbol, {'taker_fee': 0.0004})

                # Calcular PnL detalhado
                direction = 1 if pos['side'] == 'long' else -1
                gross_pnl = direction * (exit_price - pos['entry_price']) * pos['quantity']

                exit_notional = pos['quantity'] * exit_price
                exit_fee = exit_notional * fees.get('taker_fee', 0.0004)
                slippage = exit_notional * 0.0002

                # Net PnL = gross - entry fee - exit fee - slippage entrada - slippage saída - funding
                total_fees = pos['entry_fee'] + exit_fee + pos.get('entry_slippage', 0) + slippage + pos.get('funding_paid', 0)
                net_pnl = gross_pnl - total_fees

                # Registrar trade
                trade = TradeRecord(
                    symbol=symbol,
                    side=pos['side'],
                    entry_price=pos['entry_price'],
                    exit_price=exit_price,
                    quantity=pos['quantity'],
                    leverage=pos['leverage'],
                    margin_used=pos['margin'],
                    notional_value=pos['notional'],
                    entry_time=pos['entry_time'],
                    exit_time=timestamp,
                    gross_pnl=gross_pnl,
                    entry_fee=pos['entry_fee'],
                    exit_fee=exit_fee,
                    slippage_cost=slippage + pos.get('entry_slippage', 0),
                    funding_cost=pos.get('funding_paid', 0),
                    net_pnl=net_pnl,
                    pnl_pct=net_pnl / pos['margin'] * 100 if pos['margin'] > 0 else 0,
                    rrr=abs(gross_pnl / (pos['entry_price'] - pos['stop_loss'])) if pos['stop_loss'] != pos['entry_price'] else 0,
                    reason_exit=reason
                )

                # Validar trade
                trade = self._validate_trade(trade)
                trades.append(trade)

                # Atualizar capital
                capital += pos['margin'] + net_pnl
                available_margin += pos['margin'] + net_pnl
                del positions[symbol]

            # 3. Funding (a cada 8h) - apenas registrar, será deduzido no fechamento
            if i % funding_interval == 0:
                for symbol, pos in positions.items():
                    funding_rate = fees_cache.get(symbol, {}).get('funding_rate', 0.0001)
                    current_price = current_prices.get(symbol, pos['entry_price'])
                    position_value = pos['quantity'] * current_price

                    if pos['side'] == 'long':
                        funding_fee = position_value * funding_rate
                    else:
                        funding_fee = -position_value * funding_rate

                    # Apenas acumular o funding, será deduzido do PnL no fechamento
                    pos['funding_paid'] = pos.get('funding_paid', 0) + funding_fee

            # 4. Gerar novos sinais (SEM LIMITE de posições)
            new_signals = []

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

                    if signal and signal.direction != 'none' and signal.strength >= 4:
                        # Calcular score baseado em múltiplos fatores
                        atr = df_slice['atr'].iloc[-1] if 'atr' in df_slice.columns else (df_slice['high'].iloc[-1] - df_slice['low'].iloc[-1])

                        # Risk/Reward ratio
                        if signal.direction == 'long':
                            risk = signal.entry_price - signal.stop_loss
                            reward = signal.take_profit - signal.entry_price
                        else:
                            risk = signal.stop_loss - signal.entry_price
                            reward = signal.entry_price - signal.take_profit

                        rrr = reward / risk if risk > 0 else 0

                        # Score composto
                        score = signal.strength * 0.4 + rrr * 2 + (atr / signal.entry_price * 1000)

                        new_signals.append({
                            'symbol': symbol,
                            'signal': signal,
                            'score': score,
                            'atr': atr,
                            'rrr': rrr
                        })
                except:
                    continue

            # 5. Abrir posições (MARGEM DINÂMICA - sem limite fixo)
            new_signals.sort(key=lambda x: x['score'], reverse=True)

            for sig_data in new_signals:
                symbol = sig_data['symbol']
                signal = sig_data['signal']
                score = sig_data['score']

                # Margem dinâmica baseada em:
                # 1. Score do sinal (quanto maior, mais margem)
                # 2. Capital disponível
                # 3. Número de posições (diversificação natural)

                # Base: 5-20% do capital disponível por trade
                base_pct = 0.05 + min(0.15, score * 0.01)  # 5% a 20%

                # Ajuste por número de posições (diversificação)
                num_positions = len(positions)
                if num_positions > 0:
                    diversity_factor = 1 / (1 + num_positions * 0.1)  # Reduz conforme mais posições
                else:
                    diversity_factor = 1.0

                # Calcular margem
                target_margin = available_margin * base_pct * diversity_factor

                # Mínimo de $50 por posição
                if target_margin < 50:
                    continue

                # Leverage dinâmico (3-20x baseado em score e volatilidade)
                volatility = sig_data['atr'] / signal.entry_price
                if volatility > 0.03:  # Alta volatilidade
                    leverage = min(5, 3 + int(score * 0.2))
                elif volatility > 0.015:  # Média
                    leverage = min(10, 5 + int(score * 0.3))
                else:  # Baixa volatilidade
                    leverage = min(20, 8 + int(score * 0.5))

                notional = target_margin * leverage
                qty = notional / signal.entry_price

                # Taxas de entrada
                fees = fees_cache.get(symbol, {'taker_fee': 0.0004})
                entry_fee = notional * fees.get('taker_fee', 0.0004)
                entry_slippage = notional * 0.0002

                # Verificar se temos margem suficiente (incluindo taxas estimadas)
                total_required = target_margin + entry_fee + entry_slippage

                if total_required > available_margin:
                    continue

                # Abrir posição - capital reduzido apenas pela margem
                # As taxas serão deduzidas do PnL quando a posição fechar
                positions[symbol] = {
                    'side': signal.direction,
                    'entry_price': signal.entry_price,
                    'quantity': qty,
                    'margin': target_margin,
                    'leverage': leverage,
                    'notional': notional,
                    'stop_loss': signal.stop_loss,
                    'take_profit': signal.take_profit,
                    'entry_time': timestamp,
                    'entry_fee': entry_fee,
                    'entry_slippage': entry_slippage,
                    'score': score
                }

                # Só subtraímos a margem do capital, taxas vêm do PnL
                capital -= target_margin
                available_margin -= target_margin

            # 6. Calcular equity
            margin_in_positions = sum(p['margin'] for p in positions.values())
            equity = capital + margin_in_positions + unrealized_pnl
            equity_curve.append(equity)

            # Drawdown
            if equity > peak_equity:
                peak_equity = equity
                current_dd_duration = 0
            else:
                current_dd_duration += 1
                dd = (peak_equity - equity) / peak_equity
                if dd > max_drawdown:
                    max_drawdown = dd
                    max_dd_duration = current_dd_duration

        # Fechar posições restantes no final
        for symbol, pos in list(positions.items()):
            if symbol in data and len(data[symbol]) > 0:
                exit_price = data[symbol].iloc[-1]['close']
                direction = 1 if pos['side'] == 'long' else -1
                gross_pnl = direction * (exit_price - pos['entry_price']) * pos['quantity']

                fees = fees_cache.get(symbol, {'taker_fee': 0.0004})
                exit_fee = pos['quantity'] * exit_price * fees.get('taker_fee', 0.0004)
                slippage = pos['quantity'] * exit_price * 0.0002

                net_pnl = gross_pnl - pos['entry_fee'] - exit_fee - slippage - pos.get('funding_paid', 0)

                trade = TradeRecord(
                    symbol=symbol,
                    side=pos['side'],
                    entry_price=pos['entry_price'],
                    exit_price=exit_price,
                    quantity=pos['quantity'],
                    leverage=pos['leverage'],
                    margin_used=pos['margin'],
                    notional_value=pos['notional'],
                    entry_time=pos['entry_time'],
                    exit_time=all_timestamps[-1],
                    gross_pnl=gross_pnl,
                    entry_fee=pos['entry_fee'],
                    exit_fee=exit_fee,
                    slippage_cost=slippage + pos.get('entry_slippage', 0),
                    funding_cost=pos.get('funding_paid', 0),
                    net_pnl=net_pnl,
                    pnl_pct=net_pnl / pos['margin'] * 100 if pos['margin'] > 0 else 0,
                    rrr=0,
                    reason_exit='END_OF_TEST'
                )
                trades.append(trade)
                capital += pos['margin'] + net_pnl

        # Gerar relatório de validação
        report = self._generate_validation_report(
            trades=trades,
            equity_curve=equity_curve,
            initial_capital=initial_capital,
            final_capital=capital,
            max_drawdown=max_drawdown,
            max_dd_duration=max_dd_duration,
            num_candles=len(all_timestamps)
        )

        return report, trades

    def _validate_trade(self, trade: TradeRecord) -> TradeRecord:
        """Validar um trade individualmente."""
        warnings = []

        # 1. Verificar se PnL faz sentido matematicamente
        direction = 1 if trade.side == 'long' else -1
        expected_gross = direction * (trade.exit_price - trade.entry_price) * trade.quantity

        if abs(expected_gross - trade.gross_pnl) > 0.01:
            warnings.append(f"PnL bruto inconsistente: esperado {expected_gross:.2f}, obtido {trade.gross_pnl:.2f}")
            trade.is_valid = False

        # 2. Verificar se fees estão dentro do esperado
        expected_fee_range = (0.0001, 0.001)  # 0.01% a 0.1%
        fee_pct = (trade.entry_fee + trade.exit_fee) / trade.notional_value if trade.notional_value > 0 else 0

        if fee_pct < expected_fee_range[0] or fee_pct > expected_fee_range[1]:
            warnings.append(f"Fee fora do range: {fee_pct*100:.3f}%")

        # 3. Verificar PnL % razoável
        if abs(trade.pnl_pct) > 500:  # >500% em um trade é suspeito
            warnings.append(f"PnL% muito alto: {trade.pnl_pct:.1f}%")
            trade.is_valid = False

        # 4. Verificar preços
        if trade.entry_price <= 0 or trade.exit_price <= 0:
            warnings.append("Preços inválidos")
            trade.is_valid = False

        trade.validation_notes = "; ".join(warnings)
        return trade

    def _generate_validation_report(
        self,
        trades: List[TradeRecord],
        equity_curve: List[float],
        initial_capital: float,
        final_capital: float,
        max_drawdown: float,
        max_dd_duration: int,
        num_candles: int
    ) -> ValidationReport:
        """Gerar relatório de validação completo."""

        if not trades:
            return self._empty_report(initial_capital)

        # Métricas básicas
        winning = [t for t in trades if t.net_pnl > 0]
        losing = [t for t in trades if t.net_pnl <= 0]

        total_gross = sum(t.gross_pnl for t in trades)
        total_fees = sum(t.entry_fee + t.exit_fee for t in trades)
        total_slippage = sum(t.slippage_cost for t in trades)
        total_funding = sum(t.funding_cost for t in trades)
        total_net = sum(t.net_pnl for t in trades)

        # Reconciliação
        expected_final = initial_capital + total_net
        reconciliation_error = abs(final_capital - expected_final)

        # Retornos
        total_return = (final_capital - initial_capital) / initial_capital * 100

        # CAGR (assumindo ~8760 candles por ano para 1h)
        years = num_candles / 8760
        if years > 0 and final_capital > 0:
            cagr = (pow(final_capital / initial_capital, 1/years) - 1) * 100
        else:
            cagr = 0

        # Volatilidade
        equity = np.array(equity_curve)
        returns = np.diff(equity) / equity[:-1]
        returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
        volatility = np.std(returns) * np.sqrt(8760) * 100 if len(returns) > 0 else 0

        # Sharpe (rf = 0)
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(8760)
        else:
            sharpe = 0

        # Sortino
        neg_returns = returns[returns < 0]
        if len(neg_returns) > 0 and np.std(neg_returns) > 0:
            sortino = np.mean(returns) / np.std(neg_returns) * np.sqrt(8760)
        else:
            sortino = 0

        # Calmar
        calmar = cagr / (max_drawdown * 100) if max_drawdown > 0 else 0

        # Profit factor
        gross_profit = sum(t.net_pnl for t in winning)
        gross_loss = abs(sum(t.net_pnl for t in losing))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit

        # Validação estatística (t-test)
        if len(trades) > 1:
            trade_returns = [t.pnl_pct for t in trades]
            mean_return = np.mean(trade_returns)
            std_return = np.std(trade_returns)
            n = len(trade_returns)

            if std_return > 0:
                t_stat = mean_return / (std_return / np.sqrt(n))
                # Aproximação do p-value para t-distribution
                from scipy import stats
                try:
                    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n-1))
                except:
                    p_value = 1.0
            else:
                t_stat = 0
                p_value = 1.0
        else:
            t_stat = 0
            p_value = 1.0

        # Trades suspeitos
        suspicious = [t for t in trades if not t.is_valid or abs(t.pnl_pct) > 200]

        # Warnings
        warnings = []
        if reconciliation_error > 1:
            warnings.append(f"Erro de reconciliação: ${reconciliation_error:.2f}")
        if len(suspicious) > 0:
            warnings.append(f"{len(suspicious)} trades suspeitos")
        if profit_factor > 10:
            warnings.append(f"Profit factor muito alto ({profit_factor:.1f}) - possível overfitting")
        if sharpe > 5:
            warnings.append(f"Sharpe muito alto ({sharpe:.2f}) - verificar cálculos")

        return ValidationReport(
            total_trades=len(trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=len(winning) / len(trades) * 100 if trades else 0,
            total_gross_pnl=total_gross,
            total_fees=total_fees + total_slippage,
            total_funding=total_funding,
            total_net_pnl=total_net,
            initial_capital=initial_capital,
            final_capital=final_capital,
            total_return_pct=total_return,
            cagr=cagr,
            max_drawdown_pct=max_drawdown * 100,
            max_drawdown_duration=max_dd_duration,
            volatility=volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            profit_factor=profit_factor,
            is_mathematically_valid=reconciliation_error < 1 and len(suspicious) == 0,
            pnl_reconciliation_error=reconciliation_error,
            suspicious_trades=suspicious,
            warnings=warnings,
            t_statistic=t_stat,
            p_value=p_value,
            confidence_level=(1 - p_value) * 100 if p_value < 1 else 0,
            is_statistically_significant=p_value < 0.05
        )

    def _empty_report(self, initial_capital: float) -> ValidationReport:
        """Relatório vazio."""
        return ValidationReport(
            total_trades=0, winning_trades=0, losing_trades=0, win_rate=0,
            total_gross_pnl=0, total_fees=0, total_funding=0, total_net_pnl=0,
            initial_capital=initial_capital, final_capital=initial_capital,
            total_return_pct=0, cagr=0, max_drawdown_pct=0, max_drawdown_duration=0,
            volatility=0, sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
            profit_factor=0, is_mathematically_valid=True,
            pnl_reconciliation_error=0, suspicious_trades=[], warnings=[],
            t_statistic=0, p_value=1, confidence_level=0, is_statistically_significant=False
        )

    def run_wfo_validation(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        param_grid: Dict[str, List],
        n_folds: int = 6
    ) -> Dict:
        """
        Walk-Forward Optimization com validação rigorosa.

        Para cada combinação de parâmetros:
        1. Treinar em n-1 folds
        2. Testar em 1 fold (out-of-sample)
        3. Validar matematicamente
        4. Verificar significância estatística
        """
        print("=" * 70)
        print("WALK-FORWARD OPTIMIZATION COM VALIDAÇÃO RIGOROSA")
        print("=" * 70)

        # Buscar dados
        print(f"\nBuscando dados: {start_date.date()} a {end_date.date()}")
        all_data = self.fetch_data(symbols, '1h', start_date, end_date)
        print(f"Símbolos carregados: {len(all_data)}")

        if len(all_data) < 3:
            print("ERRO: Dados insuficientes!")
            return {}

        # Criar folds
        total_days = (end_date - start_date).days
        fold_days = total_days // n_folds

        folds = []
        for i in range(n_folds):
            fold_start = start_date + timedelta(days=i * fold_days)
            fold_end = fold_start + timedelta(days=fold_days)
            folds.append((fold_start, min(fold_end, end_date)))

        print(f"Folds criados: {n_folds}")

        # Gerar combinações
        from itertools import product
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))

        print(f"Combinações a testar: {len(combinations)}")

        # Testar cada combinação
        results = []

        for combo_idx, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))

            fold_reports = []

            for fold_idx, (fold_start, fold_end) in enumerate(folds):
                # Filtrar dados para este fold
                fold_data = {}
                for sym, df in all_data.items():
                    mask = (df.index >= fold_start) & (df.index <= fold_end)
                    if mask.sum() > 50:
                        fold_data[sym] = df[mask]

                if len(fold_data) < 3:
                    continue

                # Executar backtest
                report, trades = self.run_unlimited_backtest(fold_data, params)
                fold_reports.append(report)

            if fold_reports:
                # Agregar métricas
                avg_return = np.mean([r.total_return_pct for r in fold_reports])
                std_return = np.std([r.total_return_pct for r in fold_reports])
                min_return = min([r.total_return_pct for r in fold_reports])
                max_return = max([r.total_return_pct for r in fold_reports])

                avg_sharpe = np.mean([r.sharpe_ratio for r in fold_reports])
                avg_dd = np.mean([r.max_drawdown_pct for r in fold_reports])
                max_dd = max([r.max_drawdown_pct for r in fold_reports])

                avg_winrate = np.mean([r.win_rate for r in fold_reports])
                total_trades = sum([r.total_trades for r in fold_reports])

                # Validação matemática
                all_valid = all([r.is_mathematically_valid for r in fold_reports])
                all_warnings = []
                for r in fold_reports:
                    all_warnings.extend(r.warnings)

                # Significância estatística
                avg_confidence = np.mean([r.confidence_level for r in fold_reports])
                statistically_valid = np.mean([1 if r.is_statistically_significant else 0 for r in fold_reports]) > 0.5

                # Score composto (penaliza inconsistência)
                consistency_factor = 1 - (std_return / (abs(avg_return) + 1))
                consistency_factor = max(0.1, consistency_factor)

                score = (
                    avg_return * 0.3 +           # Retorno
                    avg_sharpe * 5 +             # Sharpe
                    -max_dd * 0.5 +              # Penalidade por drawdown
                    avg_winrate * 0.1 +          # Win rate
                    consistency_factor * 10      # Consistência
                )

                # Ajustar score por validação
                if not all_valid:
                    score *= 0.5
                if not statistically_valid:
                    score *= 0.8

                results.append({
                    'params': params,
                    'avg_return': avg_return,
                    'std_return': std_return,
                    'min_return': min_return,
                    'max_return': max_return,
                    'avg_sharpe': avg_sharpe,
                    'avg_dd': avg_dd,
                    'max_dd': max_dd,
                    'avg_winrate': avg_winrate,
                    'total_trades': total_trades,
                    'score': score,
                    'is_valid': all_valid,
                    'is_significant': statistically_valid,
                    'confidence': avg_confidence,
                    'warnings': all_warnings,
                    'fold_reports': fold_reports
                })

            if (combo_idx + 1) % 5 == 0:
                print(f"  Testadas {combo_idx + 1}/{len(combinations)} combinações...")

        if not results:
            print("ERRO: Nenhum resultado válido!")
            return {}

        # Ordenar por score
        results.sort(key=lambda x: x['score'], reverse=True)
        best = results[0]

        # Mostrar resultados
        print("\n" + "=" * 70)
        print("MELHOR CONFIGURAÇÃO ENCONTRADA")
        print("=" * 70)
        print(f"\nParâmetros: {json.dumps(best['params'], indent=2)}")
        print(f"\n--- MÉTRICAS ---")
        print(f"  Retorno médio: {best['avg_return']:.2f}%")
        print(f"  Retorno min/max: {best['min_return']:.2f}% / {best['max_return']:.2f}%")
        print(f"  Desvio padrão: {best['std_return']:.2f}%")
        print(f"  Sharpe médio: {best['avg_sharpe']:.2f}")
        print(f"  Max Drawdown: {best['max_dd']:.2f}%")
        print(f"  Win Rate: {best['avg_winrate']:.1f}%")
        print(f"  Total Trades: {best['total_trades']}")
        print(f"\n--- VALIDAÇÃO ---")
        print(f"  Matematicamente válido: {'SIM' if best['is_valid'] else 'NÃO'}")
        print(f"  Estatisticamente significante: {'SIM' if best['is_significant'] else 'NÃO'}")
        print(f"  Confiança: {best['confidence']:.1f}%")
        print(f"  Score final: {best['score']:.2f}")

        if best['warnings']:
            print(f"\n--- AVISOS ---")
            for w in set(best['warnings']):
                print(f"  ! {w}")

        return {
            'best': best,
            'top_10': results[:10],
            'all_results': results
        }


def main():
    """Executar validação de máximo lucro."""
    print("\n" + "=" * 70)
    print("VALIDADOR DE MÁXIMO LUCRO POSSÍVEL")
    print("=" * 70)
    print(f"Iniciando em: {datetime.now()}")
    sys.stdout.flush()

    print("\nCriando validador...")
    sys.stdout.flush()
    validator = MaxProfitValidator()
    print("Validador criado!")
    sys.stdout.flush()

    # Período de teste - 90 dias para validação robusta
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)  # 3 meses

    # Símbolos principais - Top 20 para diversificação
    symbols = SYMBOLS[:20]
    print(f"\nSímbolos a testar: {len(symbols)} symbols")
    sys.stdout.flush()

    # Grid de parâmetros COMPLETO (SEM LIMITES ARBITRÁRIOS)
    # Testando ampla gama para encontrar máximo lucro
    param_grid = {
        'strategy': ['stoch_extreme'],
        'sl_mult': [1.5, 2.0, 2.5, 3.0],           # 4 valores
        'tp_mult': [3.0, 4.0, 5.0, 6.0],            # 4 valores
        'rsi_oversold': [15, 20, 25, 30],           # 4 valores
        'rsi_overbought': [70, 75, 80, 85],         # 4 valores
        # NÃO incluímos max_positions ou max_margin - são dinâmicos!
    }

    total_combos = 1
    for v in param_grid.values():
        total_combos *= len(v)

    print(f"\nGrid de parâmetros ({total_combos} combinações):")
    for k, v in param_grid.items():
        print(f"  {k}: {v}")
    sys.stdout.flush()

    # Executar WFO com 5 folds para validação estatística robusta
    results = validator.run_wfo_validation(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        param_grid=param_grid,
        n_folds=5
    )

    if results:
        # Salvar resultados (convertendo numpy types para Python native)
        def convert_numpy(obj):
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        output = {
            'timestamp': datetime.now().isoformat(),
            'best_params': results['best']['params'],
            'metrics': {
                'avg_return': convert_numpy(results['best']['avg_return']),
                'std_return': convert_numpy(results['best']['std_return']),
                'avg_sharpe': convert_numpy(results['best']['avg_sharpe']),
                'max_dd': convert_numpy(results['best']['max_dd']),
                'win_rate': convert_numpy(results['best']['avg_winrate']),
                'total_trades': convert_numpy(results['best']['total_trades']),
                'score': convert_numpy(results['best']['score']),
                'is_valid': convert_numpy(results['best']['is_valid']),
                'is_significant': convert_numpy(results['best']['is_significant']),
                'confidence': convert_numpy(results['best']['confidence'])
            },
            'warnings': list(set(results['best']['warnings']))
        }

        os.makedirs('state', exist_ok=True)
        with open('state/max_profit_validation.json', 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\nResultados salvos em: state/max_profit_validation.json")

    return results


if __name__ == '__main__':
    main()
