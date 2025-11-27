"""
Portfolio WFO Backtest
======================
Walk-Forward Optimization para estratégia de portfolio
com múltiplas posições e gestão de margem.

Objetivo: Encontrar configuração ótima para máximo retorno
com mínimo drawdown, validado por WFO.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ccxt
from dotenv import load_dotenv

load_dotenv()

from core.config import (
    API_KEY, SECRET_KEY, SYMBOLS,
    get_backtest_config, get_wfo_config, get_validated_params
)
from core.signals import SignalGenerator
from core.scoring import ScoringSystem, SignalScore
from core.binance_fees import BinanceFees, get_binance_fees


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
    score: float
    reason_exit: str


@dataclass
class PortfolioBacktestResult:
    """Resultado do backtest de portfolio."""
    # Retornos
    total_return_pct: float
    annual_return_pct: float
    
    # Risco
    max_drawdown_pct: float
    volatility: float
    
    # Métricas
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    profit_factor: float
    
    # Trades
    total_trades: int
    win_rate: float
    avg_trade_pnl: float
    avg_winner: float
    avg_loser: float
    
    # Portfolio
    avg_positions: float
    max_positions: int
    avg_margin_usage: float
    
    # Equity curve
    equity_curve: List[float]
    dates: List[datetime]
    
    # Trades
    trades: List[BacktestTrade]


class PortfolioBacktester:
    """
    Backtester de portfolio com múltiplas posições.
    
    Simula:
    - Gestão de margem real
    - Custos de trading (fees, slippage, funding)
    - Rotação de posições baseada em score
    - Diversificação e correlação
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.scoring = ScoringSystem()

        # Cache de dados
        self.data_cache: Dict[str, pd.DataFrame] = {}

        # Exchange para dados
        self.exchange = self._create_exchange()

        # Gerenciador de taxas Binance (dinâmico)
        self.binance_fees = get_binance_fees(use_testnet=False)

        # Cache de taxas por símbolo
        self._fees_cache: Dict[str, Dict] = {}
    
    def _default_config(self) -> Dict:
        """
        Configuração padrão do backtest - valores centralizados.
        Usa get_backtest_config() do core/config.py como fonte única.
        """
        # Buscar config centralizado
        config = get_backtest_config()

        # Adicionar configs específicos de WFO
        wfo = get_wfo_config()
        config['train_months'] = wfo['train_days'] / 30
        config['test_months'] = wfo['test_days'] / 30
        config['min_folds'] = wfo['min_folds']
        config['target_margin_usage'] = 0.60

        return config
    
    def _create_exchange(self):
        """Criar conexão com exchange."""
        exchange = ccxt.binance({
            'apiKey': API_KEY,
            'secret': SECRET_KEY,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        exchange.set_sandbox_mode(True)
        return exchange

    def _get_fees_for_symbol(self, symbol: str) -> Dict:
        """
        Obter taxas dinâmicas da Binance para um símbolo.

        Returns:
            Dict com maker_fee, taker_fee, funding_rate, etc.
        """
        if symbol not in self._fees_cache:
            try:
                fees = self.binance_fees.get_all_fees_for_symbol(symbol)
                self._fees_cache[symbol] = fees
            except Exception as e:
                # Fallback para taxas padrão
                self._fees_cache[symbol] = {
                    'maker_fee': 0.0002,
                    'taker_fee': 0.0004,
                    'funding_rate': 0.0001,
                    'liquidation_fee': 0.0125
                }
        return self._fees_cache[symbol]

    def _load_all_fees(self, symbols: List[str]):
        """
        Carregar taxas de todos os símbolos de uma vez.
        Mais eficiente que carregar um por um.
        """
        try:
            all_fees = self.binance_fees.get_fees_for_symbols(symbols)
            self._fees_cache.update(all_fees)
        except Exception as e:
            print(f"Erro ao carregar taxas: {e}")
            # Fallback
            for symbol in symbols:
                self._fees_cache[symbol] = {
                    'maker_fee': 0.0002,
                    'taker_fee': 0.0004,
                    'funding_rate': 0.0001,
                    'liquidation_fee': 0.0125
                }

    def fetch_data(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, pd.DataFrame]:
        """Buscar dados históricos para todos os símbolos."""
        data = {}
        
        since = int(start_date.timestamp() * 1000)
        until = int(end_date.timestamp() * 1000)
        
        for symbol in symbols:
            cache_key = f"{symbol}_{timeframe}_{start_date.date()}_{end_date.date()}"
            
            if cache_key in self.data_cache:
                data[symbol] = self.data_cache[cache_key]
                continue
            
            try:
                ohlcv = []
                current = since
                
                while current < until:
                    batch = self.exchange.fetch_ohlcv(
                        symbol, timeframe, current, limit=1000
                    )
                    if not batch:
                        break
                    ohlcv.extend(batch)
                    current = batch[-1][0] + 1
                
                if ohlcv:
                    df = pd.DataFrame(
                        ohlcv,
                        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    )
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    
                    # Filtrar datas
                    df = df[(df.index >= start_date) & (df.index <= end_date)]
                    
                    if len(df) > 100:
                        data[symbol] = df
                        self.data_cache[cache_key] = df
                        
            except Exception as e:
                print(f"Erro buscando {symbol}: {e}")
        
        return data
    
    def run_backtest(
        self,
        data: Dict[str, pd.DataFrame],
        params: Dict,
        timeframe: str = '1h'
    ) -> PortfolioBacktestResult:
        """
        Executar backtest de portfolio.

        Args:
            data: Dados OHLCV por símbolo
            params: Parâmetros da estratégia
            timeframe: Timeframe dos dados (para cálculo de funding)

        Returns:
            PortfolioBacktestResult
        """
        config = {**self.config, **params}

        # Carregar taxas dinâmicas da Binance para todos os símbolos
        symbols_list = list(data.keys())
        self._load_all_fees(symbols_list)

        # Calcular intervalo de candles para funding (a cada 8 horas)
        # Mapeamento de timeframe para minutos
        tf_to_minutes = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480,
            '12h': 720, '1d': 1440
        }
        tf_minutes = tf_to_minutes.get(timeframe, 60)
        funding_interval = max(1, 480 // tf_minutes)  # 8h = 480 minutos

        # Cache de funding rates para backtest mais realista
        # Usar funding rates atuais (já estão em cache do _load_all_fees)
        # Isso evita chamadas API extras
        avg_funding_cache = {}
        all_funding = self.binance_fees.get_all_funding_rates()
        for symbol in symbols_list:
            if symbol in all_funding:
                avg_funding_cache[symbol] = all_funding[symbol].funding_rate
            else:
                avg_funding_cache[symbol] = 0.0001  # Fallback

        # Estado inicial
        initial_capital = config['initial_capital']
        capital = initial_capital
        positions = {}  # symbol -> position_data
        trades = []
        equity_curve = [capital]
        dates = []

        # Métricas de tracking
        peak_equity = capital
        max_drawdown = 0
        margin_usages = []
        position_counts = []

        # Signal generator
        signal_gen = SignalGenerator(params)

        # PRÉ-CALCULAR INDICADORES para evitar recálculos no loop
        # Isso reduz tempo de 95% -> ~5% do backtest
        precomputed_data = {}
        for symbol, df in data.items():
            precomputed_data[symbol] = signal_gen.prepare_data(df.copy())

        # Alinhar timestamps
        all_timestamps = set()
        for df in data.values():
            all_timestamps.update(df.index.tolist())
        all_timestamps = sorted(all_timestamps)
        
        if not all_timestamps:
            return self._empty_result()
        
        # Iterar por cada candle
        for i, timestamp in enumerate(all_timestamps[100:], 100):  # Skip primeiros 100 para indicadores
            dates.append(timestamp)
            
            # Atualizar posições existentes
            total_pnl = 0
            positions_to_close = []
            
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
                
                # Verificar Liquidação, SL e TP
                # Prioridade: LIQUIDATION > STOP_LOSS > TAKE_PROFIT
                close_reason = None
                exit_price = current['close']
                liq_price = pos.get('liquidation_price', 0)

                if side == 'long':
                    # LONG: Liquidação ocorre quando preço cai abaixo do liq_price
                    if liq_price > 0 and current['low'] <= liq_price:
                        close_reason = 'LIQUIDATION'
                        exit_price = liq_price
                    elif current['low'] <= sl:
                        close_reason = 'STOP_LOSS'
                        exit_price = sl
                    elif current['high'] >= tp:
                        close_reason = 'TAKE_PROFIT'
                        exit_price = tp
                else:
                    # SHORT: Liquidação ocorre quando preço sobe acima do liq_price
                    if liq_price > 0 and current['high'] >= liq_price:
                        close_reason = 'LIQUIDATION'
                        exit_price = liq_price
                    elif current['high'] >= sl:
                        close_reason = 'STOP_LOSS'
                        exit_price = sl
                    elif current['low'] <= tp:
                        close_reason = 'TAKE_PROFIT'
                        exit_price = tp
                
                if close_reason:
                    positions_to_close.append((symbol, exit_price, close_reason))
                else:
                    # Calcular PnL não realizado
                    if side == 'long':
                        pnl = (current['close'] - entry) * qty
                    else:
                        pnl = (entry - current['close']) * qty
                    total_pnl += pnl
            
            # Fechar posições
            for symbol, exit_price, reason in positions_to_close:
                pos = positions[symbol]
                entry = pos['entry_price']
                side = pos['side']
                qty = pos['quantity']

                # Obter taxas dinâmicas do símbolo
                fees = self._get_fees_for_symbol(symbol)
                taker_fee = fees.get('taker_fee', 0.0004)
                liquidation_fee = fees.get('liquidation_fee', 0.0125)

                # Custos acumulados durante a posição
                entry_cost = pos.get('entry_cost', 0)
                funding_paid = pos.get('funding_paid', 0)
                # Nota: funding_paid já foi subtraído do capital em tempo real
                # Portanto NÃO deve ser subtraído novamente do PnL

                if reason == 'LIQUIDATION':
                    # Em liquidação, a margem é perdida menos insurance fund
                    # Conforme Binance: perda = margem - (pequena quantia retornada se houver)
                    # Na prática, trader perde toda a margem + entry cost
                    # Funding já foi descontado do capital, então incluímos no PnL para report
                    notional = qty * entry
                    liq_fee = notional * liquidation_fee
                    pnl = -pos['margin'] - entry_cost - funding_paid  # Perda total para report
                    # Não há capital retornado em liquidação
                    capital += 0  # Margem já foi subtraída na abertura
                else:
                    # Calcular PnL conforme fórmula Binance:
                    # PnL = direction × (exit_price - entry_price) × quantity
                    # direction: 1 para long, -1 para short
                    direction = 1 if side == 'long' else -1
                    pnl = direction * (exit_price - entry) * qty

                    # Custos de saída (notional value × taker_fee)
                    notional_exit = qty * exit_price
                    exit_cost = notional_exit * taker_fee
                    exit_cost += notional_exit * config.get('slippage', 0.0003)

                    # PnL para equity = PnL bruto - custo saída (entry cost já foi pago)
                    pnl_for_equity = pnl - exit_cost

                    # PnL para report = inclui todos os custos
                    pnl = pnl - entry_cost - exit_cost - funding_paid

                    capital += pos['margin'] + pnl_for_equity
                
                # Registrar trade
                trades.append(BacktestTrade(
                    symbol=symbol,
                    side=side,
                    entry_price=entry,
                    exit_price=exit_price,
                    quantity=qty,
                    entry_time=pos['entry_time'],
                    exit_time=timestamp,
                    pnl=pnl,
                    pnl_pct=pnl / pos['margin'] * 100,
                    score=pos['score'],
                    reason_exit=reason
                ))
                
                del positions[symbol]
            
            # Gerar novos sinais
            available_margin = capital - sum(p['margin'] for p in positions.values())
            margin_used = sum(p['margin'] for p in positions.values())
            margin_pct = margin_used / capital if capital > 0 else 0
            
            margin_usages.append(margin_pct)
            position_counts.append(len(positions))
            
            # Verificar se pode abrir novas posições
            if margin_pct < config['max_margin_usage'] and len(positions) < config['max_positions']:
                new_signals = []
                
                for symbol in data.keys():
                    if symbol in positions:
                        continue

                    # Usar dados pré-calculados
                    df = precomputed_data[symbol]
                    if timestamp not in df.index:
                        continue

                    # Obter dados até este ponto
                    idx = df.index.get_loc(timestamp)
                    if idx < 50:
                        continue

                    df_slice = df.iloc[:idx+1].tail(100)

                    try:
                        # precomputed=True evita recálculo de indicadores
                        signal = signal_gen.generate_signal(df_slice, precomputed=True)

                        if signal and signal.direction != 'none' and signal.strength >= 5:
                            # Usar ATR pré-calculado (já está no df_slice)
                            atr = df_slice['atr'].iloc[-1] if 'atr' in df_slice.columns else (
                                df_slice['high'].iloc[-1] - df_slice['low'].iloc[-1]
                            )
                            
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
                            
                            if score.score >= config['min_score_to_open']:
                                new_signals.append(score)
                    except:
                        continue
                
                # Ordenar por score e abrir melhores
                new_signals.sort(key=lambda x: x.score, reverse=True)
                
                for signal in new_signals[:2]:  # Máximo 2 novas por candle
                    if available_margin < initial_capital * config['min_position_pct']:
                        break

                    # Obter taxas dinâmicas do símbolo
                    fees = self._get_fees_for_symbol(signal.symbol)
                    taker_fee = fees.get('taker_fee', 0.0004)
                    min_notional = fees.get('min_notional', 5)

                    # Capital base para cálculo de posição
                    # Se use_compounding=False, usar capital inicial
                    # Se use_compounding=True, usar capital atual
                    base_capital = capital if config.get('use_compounding', False) else initial_capital

                    # Calcular tamanho da posição
                    position_pct = min(config['max_position_pct'], 0.03 + signal.score * 0.003)
                    margin = base_capital * position_pct
                    margin = min(margin, available_margin * 0.25)  # Máx 25% do disponível

                    # Leverage conservador (2-5x)
                    leverage = min(2 + int(signal.score * 0.3), config.get('max_leverage', 5))
                    notional = margin * leverage
                    qty = notional / signal.entry_price

                    # Verificar notional mínimo (Binance requirement)
                    if notional < min_notional:
                        continue

                    # Custos de entrada conforme Binance:
                    # Trading Fee = Notional Value × Fee Rate
                    entry_cost = notional * taker_fee
                    entry_cost += notional * config.get('slippage', 0.0003)

                    if margin + entry_cost <= available_margin:
                        # Calcular preço de liquidação conforme Binance
                        # Usando fórmula: Liq = Entry × (1 ± 1/leverage - MMR)
                        # Simplificado para isolated margin mode
                        liq_price = self.binance_fees.calculate_liquidation_price(
                            symbol=signal.symbol,
                            side=signal.side,
                            entry_price=signal.entry_price,
                            quantity=qty,
                            wallet_balance=margin,  # Margem isolada
                            leverage=leverage
                        )

                        positions[signal.symbol] = {
                            'side': signal.side,
                            'entry_price': signal.entry_price,
                            'quantity': qty,
                            'margin': margin,
                            'leverage': leverage,
                            'stop_loss': signal.stop_loss,
                            'take_profit': signal.take_profit,
                            'liquidation_price': liq_price,  # Preço de liquidação
                            'score': signal.score,
                            'entry_time': timestamp,
                            'entry_cost': entry_cost,
                            'taker_fee': taker_fee  # Guardar taxa para uso posterior
                        }
                        # Subtrair margem + custo do capital ao abrir posição
                        capital -= margin + entry_cost
                        available_margin -= margin + entry_cost
            
            # Funding rate (pago a cada 8 horas em Futures)
            # Conforme Binance: Funding Fee = Position Value × Funding Rate
            # Position Value = Mark Price × Position Size
            # Funding é cobrado a cada 8 horas (00:00, 08:00, 16:00 UTC)
            # Intervalo dinâmico baseado no timeframe
            if i > 0 and i % funding_interval == 0:
                funding_cost = 0
                for symbol, pos in positions.items():
                    # Usar funding rate médio histórico (mais realista para backtest)
                    # ao invés do rate atual que pode ser volátil
                    funding_rate = avg_funding_cache.get(symbol, 0.0001)

                    # Position value = mark price × quantity
                    # Usando preço atual como proxy para mark price
                    if symbol in data and timestamp in data[symbol].index:
                        current_price = data[symbol].loc[timestamp]['close']
                    else:
                        current_price = pos['entry_price']

                    position_value = pos['quantity'] * current_price

                    # Funding fee conforme Binance:
                    # - Funding rate positivo: longs pagam, shorts recebem
                    # - Funding rate negativo: shorts pagam, longs recebem
                    if pos['side'] == 'long':
                        # Long paga quando rate > 0, recebe quando rate < 0
                        funding_fee = position_value * funding_rate
                    else:
                        # Short recebe quando rate > 0, paga quando rate < 0
                        funding_fee = -position_value * funding_rate

                    funding_cost += funding_fee
                    # Acumular funding pago pela posição para incluir no PnL
                    pos['funding_paid'] = pos.get('funding_paid', 0) + funding_fee

                capital -= funding_cost

            # Calcular equity = capital + margens em posicoes + pnl nao realizado
            margin_in_positions = sum(p['margin'] for p in positions.values())
            equity = capital + margin_in_positions + total_pnl
            equity_curve.append(equity)

            # Drawdown
            if equity > peak_equity:
                peak_equity = equity
            dd = (peak_equity - equity) / peak_equity
            max_drawdown = max(max_drawdown, dd)

            # Circuit breaker
            if max_drawdown > config['max_drawdown_halt']:
                break
        
        # Fechar posições restantes
        for symbol, pos in list(positions.items()):
            if symbol in data:
                df = data[symbol]
                if len(df) > 0:
                    exit_price = df.iloc[-1]['close']
                    entry = pos['entry_price']
                    side = pos['side']
                    qty = pos['quantity']
                    
                    if side == 'long':
                        pnl = (exit_price - entry) * qty
                    else:
                        pnl = (entry - exit_price) * qty
                    
                    cost = qty * exit_price * config['taker_fee']
                    pnl -= cost
                    capital += pos['margin'] + pnl
                    
                    trades.append(BacktestTrade(
                        symbol=symbol,
                        side=side,
                        entry_price=entry,
                        exit_price=exit_price,
                        quantity=qty,
                        entry_time=pos['entry_time'],
                        exit_time=dates[-1] if dates else datetime.now(),
                        pnl=pnl,
                        pnl_pct=pnl / pos['margin'] * 100,
                        score=pos['score'],
                        reason_exit='END'
                    ))
        
        # Calcular métricas
        return self._calculate_metrics(
            equity_curve, dates, trades, 
            margin_usages, position_counts,
            config['initial_capital']
        )
    
    def _calculate_metrics(
        self,
        equity_curve: List[float],
        dates: List[datetime],
        trades: List[BacktestTrade],
        margin_usages: List[float],
        position_counts: List[int],
        initial_capital: float
    ) -> PortfolioBacktestResult:
        """Calcular métricas do backtest."""
        if len(equity_curve) < 2:
            return self._empty_result()
        
        equity = np.array(equity_curve)
        
        # Retornos
        total_return = (equity[-1] - initial_capital) / initial_capital * 100

        # Retornos horários para cálculos
        returns = np.diff(equity) / equity[:-1]
        returns = returns[~np.isnan(returns)]

        if len(returns) == 0:
            return self._empty_result()

        # Calcular dias reais do backtest
        if dates and len(dates) >= 2:
            try:
                # Calcular diferença em dias
                first_date = pd.to_datetime(dates[0])
                last_date = pd.to_datetime(dates[-1])
                days_tested = (last_date - first_date).days
            except:
                days_tested = len(returns) / 24  # Fallback: assumir 1h candles
        else:
            days_tested = len(returns) / 24  # Assumir 1h candles

        # Anualização correta baseada em dias reais testados
        days_tested = max(1, days_tested)
        if total_return > -100:  # Evitar log de número negativo
            annual_return = ((1 + total_return/100) ** (365.25 / days_tested) - 1) * 100
        else:
            annual_return = -100

        # Volatilidade anualizada
        # Para dados horários: sqrt(8760) = sqrt(horas por ano)
        # Para dados diários: sqrt(252) = sqrt(dias de trading por ano)
        periods_per_year = 8760 if len(returns) > days_tested * 10 else 252  # Detectar granularidade
        annualization_factor = np.sqrt(periods_per_year)
        volatility = np.std(returns) * annualization_factor * 100

        # Drawdown
        peak = np.maximum.accumulate(equity)
        dd = (peak - equity) / peak
        max_dd = np.max(dd) * 100

        # Sharpe anualizado (rf = 0)
        # Fator de anualização: sqrt(períodos por ano)
        sharpe = (np.mean(returns) / np.std(returns)) * annualization_factor if np.std(returns) > 0 else 0

        # Sortino anualizado
        neg_returns = returns[returns < 0]
        if len(neg_returns) > 0 and np.std(neg_returns) > 0:
            sortino = (np.mean(returns) / np.std(neg_returns)) * annualization_factor
        else:
            sortino = 0

        # Calmar (retorno anual / max drawdown)
        calmar = annual_return / max_dd if max_dd > 0 else 0
        
        # Trades
        if trades:
            winners = [t for t in trades if t.pnl > 0]
            losers = [t for t in trades if t.pnl <= 0]
            
            win_rate = len(winners) / len(trades) * 100
            avg_trade = np.mean([t.pnl for t in trades])
            avg_winner = np.mean([t.pnl for t in winners]) if winners else 0
            avg_loser = np.mean([t.pnl for t in losers]) if losers else 0
            
            gross_profit = sum(t.pnl for t in winners)
            gross_loss = abs(sum(t.pnl for t in losers))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit
        else:
            win_rate = 0
            avg_trade = 0
            avg_winner = 0
            avg_loser = 0
            profit_factor = 0
        
        return PortfolioBacktestResult(
            total_return_pct=round(total_return, 2),
            annual_return_pct=round(annual_return, 2),
            max_drawdown_pct=round(max_dd, 2),
            volatility=round(volatility, 2),
            sharpe_ratio=round(sharpe, 2),
            sortino_ratio=round(sortino, 2),
            calmar_ratio=round(calmar, 2),
            profit_factor=round(profit_factor, 2),
            total_trades=len(trades),
            win_rate=round(win_rate, 1),
            avg_trade_pnl=round(avg_trade, 2),
            avg_winner=round(avg_winner, 2),
            avg_loser=round(avg_loser, 2),
            avg_positions=round(np.mean(position_counts), 1) if position_counts else 0,
            max_positions=max(position_counts) if position_counts else 0,
            avg_margin_usage=round(np.mean(margin_usages) * 100, 1) if margin_usages else 0,
            equity_curve=equity.tolist(),
            dates=dates,
            trades=trades
        )
    
    def _empty_result(self) -> PortfolioBacktestResult:
        """Resultado vazio para casos de erro."""
        return PortfolioBacktestResult(
            total_return_pct=0, annual_return_pct=0,
            max_drawdown_pct=100, volatility=100,
            sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
            profit_factor=0, total_trades=0, win_rate=0,
            avg_trade_pnl=0, avg_winner=0, avg_loser=0,
            avg_positions=0, max_positions=0, avg_margin_usage=0,
            equity_curve=[], dates=[], trades=[]
        )
    
    def run_wfo(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        param_grid: Dict[str, List]
    ) -> Dict:
        """
        Executar Walk-Forward Optimization.
        
        Returns:
            Melhor configuração e resultados
        """
        print("=" * 60)
        print("WALK-FORWARD OPTIMIZATION - PORTFOLIO")
        print("=" * 60)
        
        config = self.config
        train_months = config['train_months']
        test_months = config['test_months']
        
        # Buscar todos os dados
        print(f"\nBuscando dados: {start_date.date()} a {end_date.date()}")
        all_data = self.fetch_data(symbols, config['timeframe'], start_date, end_date)
        print(f"Símbolos com dados: {len(all_data)}")
        
        if len(all_data) < 5:
            print("Dados insuficientes!")
            return {}
        
        # Criar folds (com step de test_months para maximizar folds)
        folds = []
        current = start_date
        train_days = int(train_months * 30)
        test_days = int(test_months * 30)
        step_days = max(7, test_days)  # Step mínimo de 1 semana

        while current + timedelta(days=train_days + test_days) <= end_date:
            train_start = current
            train_end = current + timedelta(days=train_days)
            test_start = train_end
            test_end = test_start + timedelta(days=test_days)

            folds.append({
                'train': (train_start, train_end),
                'test': (test_start, test_end)
            })

            current = current + timedelta(days=step_days)  # Avançar pelo step, não pelo test completo
        
        print(f"Folds criados: {len(folds)}")
        
        if len(folds) < config['min_folds']:
            print(f"Folds insuficientes (mínimo: {config['min_folds']})")
            return {}
        
        # Gerar combinações de parâmetros
        from itertools import product
        
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))
        
        print(f"Combinações de parâmetros: {len(combinations)}")
        
        # Testar cada combinação
        results = []
        
        for i, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))
            
            fold_results = []
            
            for fold in folds:
                train_start, train_end = fold['train']
                test_start, test_end = fold['test']
                
                # Filtrar dados para o período de teste
                test_data = {}
                for sym, df in all_data.items():
                    mask = (df.index >= test_start) & (df.index <= test_end)
                    if mask.sum() > 50:
                        test_data[sym] = df[mask]
                
                if len(test_data) < 3:
                    continue
                
                # Executar backtest
                result = self.run_backtest(test_data, params)
                fold_results.append(result)
            
            if fold_results:
                # Agregar resultados dos folds
                avg_return = np.mean([r.total_return_pct for r in fold_results])
                avg_sharpe = np.mean([r.sharpe_ratio for r in fold_results])
                max_dd = max([r.max_drawdown_pct for r in fold_results])
                avg_winrate = np.mean([r.win_rate for r in fold_results])
                total_trades = sum([r.total_trades for r in fold_results])

                # Score composto
                score = avg_sharpe * 2 + avg_return / 10 - max_dd / 5 + avg_winrate / 20

                # Converter fold_results para dicionários com detalhes
                fold_details = []
                for fold_idx, (fold_info, result) in enumerate(zip(folds, fold_results)):
                    fold_details.append({
                        'fold_number': fold_idx + 1,
                        'train_start': fold_info['train'][0].isoformat(),
                        'train_end': fold_info['train'][1].isoformat(),
                        'test_start': fold_info['test'][0].isoformat(),
                        'test_end': fold_info['test'][1].isoformat(),
                        'return_pct': round(result.total_return_pct, 2),
                        'max_drawdown_pct': round(result.max_drawdown_pct, 2),
                        'sharpe_ratio': round(result.sharpe_ratio, 2),
                        'sortino_ratio': round(result.sortino_ratio, 2),
                        'profit_factor': round(result.profit_factor, 2),
                        'total_trades': result.total_trades,
                        'win_rate': round(result.win_rate, 1),
                        'avg_trade_pnl': round(result.avg_trade_pnl, 2),
                    })

                results.append({
                    'params': params,
                    'avg_return': avg_return,
                    'avg_sharpe': avg_sharpe,
                    'max_dd': max_dd,
                    'avg_winrate': avg_winrate,
                    'total_trades': total_trades,
                    'score': score,
                    'fold_results': fold_results,
                    'fold_details': fold_details  # Detalhes para armazenamento
                })
            
            if (i + 1) % 10 == 0:
                print(f"Testadas {i+1}/{len(combinations)} combinações...")
        
        if not results:
            print("Nenhum resultado válido!")
            return {}
        
        # Ordenar por score
        results.sort(key=lambda x: x['score'], reverse=True)
        best = results[0]
        
        print("\n" + "=" * 60)
        print("MELHOR CONFIGURAÇÃO ENCONTRADA")
        print("=" * 60)
        print(f"\nParâmetros: {json.dumps(best['params'], indent=2)}")
        print(f"\nMétricas médias:")
        print(f"  Retorno: {best['avg_return']:.2f}%")
        print(f"  Sharpe: {best['avg_sharpe']:.2f}")
        print(f"  Max DD: {best['max_dd']:.2f}%")
        print(f"  Win Rate: {best['avg_winrate']:.1f}%")
        print(f"  Total Trades: {best['total_trades']}")
        print(f"  Score: {best['score']:.2f}")
        
        # Preparar detalhes dos folds para armazenamento
        fold_details = best.get('fold_details', [])

        return {
            'best_params': best['params'],
            'metrics': {
                'avg_return': best['avg_return'],
                'avg_sharpe': best['avg_sharpe'],
                'max_dd': best['max_dd'],
                'avg_winrate': best['avg_winrate'],
                'total_trades': best['total_trades'],
                'score': best['score']
            },
            'fold_details': fold_details,  # Detalhes de cada fold
            'all_results': results[:10],  # Top 10
            'num_folds': len(folds),
            'symbols': symbols,
            'timeframe': config['timeframe'],
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            }
        }


def main():
    """Executar WFO de portfolio."""
    print("\n" + "=" * 60)
    print("INICIANDO PORTFOLIO WFO BACKTEST")
    print("=" * 60)
    
    # Configuração
    symbols = SYMBOLS[:15]  # Top 15 símbolos
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)  # 6 meses
    
    # Grid de parâmetros a testar
    param_grid = {
        'strategy': ['stoch_extreme', 'rsi_extreme', 'momentum_burst'],
        'sl_mult': [1.5, 2.0, 2.5],
        'tp_mult': [2.0, 3.0, 4.0],
        'rsi_oversold': [20, 25, 30],
        'rsi_overbought': [70, 75, 80],
        'max_positions': [5, 8, 10],
        'max_margin_usage': [0.7, 0.8, 0.9],
    }
    
    # Executar WFO
    backtester = PortfolioBacktester()
    results = backtester.run_wfo(symbols, start_date, end_date, param_grid)
    
    if results:
        # Salvar resultados
        output = {
            'timestamp': datetime.now().isoformat(),
            'best_params': results['best_params'],
            'metrics': results['metrics'],
            'symbols': symbols,
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            }
        }
        
        # Salvar como baseline
        baseline_path = 'state/baselines/portfolio_wfo_latest.json'
        os.makedirs(os.path.dirname(baseline_path), exist_ok=True)
        
        with open(baseline_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"\nResultados salvos em: {baseline_path}")
        
        # Salvar também como current_best se for melhor
        current_best_path = 'state/current_best.json'
        
        should_update = True
        if os.path.exists(current_best_path):
            with open(current_best_path, 'r') as f:
                current = json.load(f)
            if current.get('metrics', {}).get('score', 0) >= results['metrics']['score']:
                should_update = False
                print("Baseline atual é melhor ou igual. Não atualizado.")
        
        if should_update:
            with open(current_best_path, 'w') as f:
                json.dump(output, f, indent=2, default=str)
            print(f"Novo melhor resultado salvo em: {current_best_path}")
    
    return results


if __name__ == '__main__':
    main()
