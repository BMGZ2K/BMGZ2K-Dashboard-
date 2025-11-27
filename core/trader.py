"""
Trader Module - Gerenciador de trades simplificado
Combina signals, risk e execution
"""
import pandas as pd
from typing import Dict, Optional, List
from datetime import datetime
from dataclasses import dataclass
import json
import os
import logging

from .signals import SignalGenerator, check_exit_signal
from .config import get_validated_params, get_backtest_config
from .utils import save_json_atomic, load_json_safe
from .binance_fees import get_binance_fees

log = logging.getLogger(__name__)


@dataclass
class Trade:
    """Registro de trade."""
    symbol: str
    side: str
    entry_price: float
    exit_price: float = 0.0
    quantity: float = 0.0
    pnl: float = 0.0  # PnL líquido (após taxas)
    pnl_gross: float = 0.0  # PnL bruto (antes das taxas)
    commission: float = 0.0  # Taxa de trading (entrada + saída)
    funding_cost: float = 0.0  # Custo de funding
    entry_time: str = ""
    exit_time: str = ""
    reason_entry: str = ""
    reason_exit: str = ""
    status: str = "open"  # open, closed


@dataclass
class Position:
    """Posicao ativa."""
    symbol: str
    side: str
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    entry_time: str
    max_price: float = 0.0
    min_price: float = float('inf')
    reason_entry: str = ""
    strategy: str = "unknown"
    order_id: str = ""  # ID único da Binance para evitar duplicatas


class Trader:
    """
    Gerenciador de trading simplificado.
    """
    
    def __init__(self, params: Dict = None):
        self.params = params or self._default_params()
        self.signal_generator = SignalGenerator(self.params)

        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []

        # Balance inicial do config (será sobrescrito por load_state ou update_balance)
        backtest_config = get_backtest_config()
        self.balance = float(backtest_config.get('initial_capital', 10000))
        self.initial_balance = self.balance

        # Risk params (do config centralizado)
        self.risk_per_trade = self.params.get('risk_per_trade', 0.01)
        self.max_positions = self.params.get('max_positions', 10)
        self.max_drawdown = self.params.get('max_drawdown', backtest_config.get('max_drawdown_halt', 0.20))
        
        # State
        self.high_water_mark = self.balance
        self.is_halted = False
    
    def _default_params(self) -> Dict:
        # Usa parâmetros centralizados do config
        return get_validated_params()
    
    def update_balance(self, new_balance: float):
        """Atualizar balance e verificar circuit breaker."""
        if new_balance <= 0:
            return False
        
        # Se nao temos trades, aceitar o balance como inicial
        # Isso evita falsos positivos de drawdown ao iniciar
        if len(self.trades) == 0:
            self.balance = new_balance
            self.initial_balance = new_balance
            self.high_water_mark = new_balance
            self.is_halted = False
            return False
            
        self.balance = new_balance
        
        # Atualizar high water mark se subiu
        if new_balance > self.high_water_mark:
            self.high_water_mark = new_balance
        
        # Circuit breaker - so ativar se tivermos historico real de trades
        drawdown = (self.high_water_mark - self.balance) / self.high_water_mark
        if drawdown > self.max_drawdown:
            self.is_halted = True
            return True
        
        # Desativar circuit breaker se drawdown esta ok
        self.is_halted = False
        return False
    
    def calculate_position_size(self, price: float, stop_loss: float) -> float:
        """Calcular tamanho da posicao baseado em risco."""
        # Parâmetros do config
        max_leverage = self.params.get('max_leverage', 10)
        max_position_pct = self.params.get('max_position_pct', 0.15)  # 15% max por posição
        min_position_pct = self.params.get('min_position_pct', 0.05)  # 5% min por posição
        min_notional = self.params.get('min_notional', 6.0)

        risk_amount = self.balance * self.risk_per_trade
        stop_distance = abs(price - stop_loss)

        if stop_distance == 0:
            return 0.0

        # Quantidade baseada no risco
        quantity = risk_amount / stop_distance

        # Limitar pelo leverage máximo
        max_quantity_leverage = (self.balance * max_leverage) / price
        quantity = min(quantity, max_quantity_leverage)

        # Limitar pelo % máximo da posição
        max_quantity_pct = (self.balance * max_position_pct * max_leverage) / price
        quantity = min(quantity, max_quantity_pct)

        # Garantir quantidade mínima (% mínimo do balance)
        min_quantity = (self.balance * min_position_pct) / price
        if quantity < min_quantity:
            quantity = min_quantity

        # Verificar notional mínimo
        notional = quantity * price
        if notional < min_notional:
            return 0.0

        return quantity
    
    def process_candle(self, symbol: str, df: pd.DataFrame) -> Optional[Dict]:
        """
        Processar uma nova candle.
        
        Returns:
            Dict com acao a tomar ou None
        """
        if self.is_halted:
            return {'action': 'halted', 'reason': 'Circuit breaker ativo'}
        
        if len(df) < 50:
            return None
        
        current = df.iloc[-1]
        price = current['close']
        high = current['high']
        low = current['low']
        
        # Verificar posicao existente
        if symbol in self.positions:
            return self._manage_position(symbol, df, price, high, low)
        
        # Verificar nova entrada
        if len(self.positions) >= self.max_positions:
            return None
        
        return self._check_entry(symbol, df)
    
    def _manage_position(self, symbol: str, df: pd.DataFrame, price: float, high: float, low: float) -> Optional[Dict]:
        """Gerenciar posicao existente."""
        pos = self.positions[symbol]
        
        # Atualizar max/min
        pos.max_price = max(pos.max_price, high)
        pos.min_price = min(pos.min_price, low)
        
        # Calcular RSI e ATR usando params (não hardcoded)
        from .signals import calculate_rsi, calculate_atr
        import numpy as np
        rsi_period = self.params.get('rsi_period', 14)
        atr_period = self.params.get('atr_period', 14)
        rsi = calculate_rsi(df['close'], rsi_period).iloc[-1]
        atr = calculate_atr(df['high'], df['low'], df['close'], atr_period).iloc[-1]

        # CORRIGIDO: Fallback se ATR for NaN ou 0
        if np.isnan(atr) or atr == 0:
            # Usar volatilidade simples como fallback
            atr = df['close'].pct_change().std() * df['close'].iloc[-1] * np.sqrt(atr_period)
            if np.isnan(atr) or atr == 0:
                atr = df['close'].iloc[-1] * 0.02  # 2% fallback final
        
        # Verificar saida
        exit_reason = check_exit_signal(
            position={'side': pos.side, 'entry': pos.entry_price, 'sl': pos.stop_loss, 'tp': pos.take_profit},
            current_price=price,
            current_high=high,
            current_low=low,
            atr=atr,
            rsi=rsi,
            rsi_exit_long=self.params.get('rsi_exit_long', 70),
            rsi_exit_short=self.params.get('rsi_exit_short', 30)
        )
        
        if exit_reason:
            return {
                'action': 'close',
                'symbol': symbol,
                'side': 'sell' if pos.side == 'long' else 'buy',
                'quantity': pos.quantity,
                'reason': exit_reason
            }
        
        return None
    
    def _check_entry(self, symbol: str, df: pd.DataFrame) -> Optional[Dict]:
        """Verificar sinal de entrada."""
        signal = self.signal_generator.generate_signal(df)
        
        if signal.direction == 'none' or signal.strength < 5:
            return None
        
        quantity = self.calculate_position_size(signal.entry_price, signal.stop_loss)
        
        if quantity <= 0:
            return None
        
        return {
            'action': 'open',
            'symbol': symbol,
            'side': 'buy' if signal.direction == 'long' else 'sell',
            'quantity': quantity,
            'price': signal.entry_price,
            'stop_loss': signal.stop_loss,
            'take_profit': signal.take_profit,
            'reason': signal.reason,
            'strength': signal.strength
        }
    
    def record_entry(self, symbol: str, side: str, price: float, quantity: float, sl: float, tp: float, reason: str, strategy: str = None, order_id: str = None):
        """Registrar entrada."""
        # Criar order_id único se não fornecido
        if not order_id:
            order_id = f"{symbol.replace('/', '')}_{price}_{quantity}_{datetime.now().timestamp()}"

        self.positions[symbol] = Position(
            symbol=symbol,
            side='long' if side == 'buy' else 'short',
            entry_price=price,
            quantity=quantity,
            stop_loss=sl,
            take_profit=tp,
            entry_time=datetime.now().isoformat(),
            max_price=price,
            min_price=price,
            reason_entry=reason,
            strategy=strategy or self.params.get('strategy', 'unknown'),
            order_id=order_id
        )
    
    def record_exit(self, symbol: str, exit_price: float, reason: str) -> float:
        """Registrar saida e calcular PnL."""
        if symbol not in self.positions:
            # CORRIGIDO: Evitar duplicação - posição já foi fechada
            log.debug(f"record_exit ignorado para {symbol}: posição não existe (já fechada)")
            return 0.0

        pos = self.positions[symbol]

        # CORRIGIDO: Verificar se posição já está sendo fechada (flag de lock)
        if getattr(pos, '_closing', False):
            log.debug(f"record_exit ignorado para {symbol}: fechamento já em andamento")
            return 0.0
        pos._closing = True  # Marcar que está fechando

        # Obter taxas dinâmicas da Binance
        try:
            binance_fees = get_binance_fees(use_testnet=True)
            raw_symbol = symbol.replace('/', '')
            fees = binance_fees.get_all_fees_for_symbol(raw_symbol)
            taker_fee = fees.get('taker_fee', 0.0004)
            funding_rate = fees.get('funding_rate', 0.0001)
        except Exception:
            # Fallback para taxas padrão
            taker_fee = 0.0004
            funding_rate = 0.0001

        # PnL bruto
        if pos.side == 'long':
            pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            pnl = (pos.entry_price - exit_price) * pos.quantity

        # Custos de trading (entrada + saída)
        notional_entry = pos.entry_price * pos.quantity
        notional_exit = exit_price * pos.quantity
        commission = (notional_entry + notional_exit) * taker_fee

        # Calcular funding pago durante a posição
        # Funding é cobrado a cada 8h. Estimamos baseado no tempo da posição
        try:
            entry_time = datetime.fromisoformat(pos.entry_time)
            exit_time = datetime.now()
            hours_held = (exit_time - entry_time).total_seconds() / 3600
            funding_periods = hours_held / 8  # Número de períodos de funding
            # Funding = notional * rate * períodos
            # Long paga quando rate positivo, short recebe
            avg_notional = (notional_entry + notional_exit) / 2
            if pos.side == 'long':
                funding_cost = avg_notional * funding_rate * funding_periods
            else:
                funding_cost = -avg_notional * funding_rate * funding_periods  # Short recebe
        except Exception:
            funding_cost = 0

        # PnL líquido = bruto - comissão - funding
        net_pnl = pnl - commission - funding_cost

        log.debug(f"PnL {symbol}: bruto={pnl:.2f}, commission={commission:.2f}, funding={funding_cost:.2f}, net={net_pnl:.2f}")
        
        # Registrar trade com detalhes
        trade = Trade(
            symbol=symbol,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=pos.quantity,
            pnl=net_pnl,
            pnl_gross=pnl,
            commission=commission,
            funding_cost=funding_cost,
            entry_time=pos.entry_time,
            exit_time=datetime.now().isoformat(),
            reason_entry=getattr(pos, 'reason_entry', ''),
            reason_exit=reason,
            status="closed"
        )
        # Adicionar atributos extras
        trade.strategy = getattr(pos, 'strategy', self.params.get('strategy', 'unknown'))
        trade.pnl_pct = (net_pnl / notional_entry) * 100 if notional_entry > 0 else 0
        trade.order_id = getattr(pos, 'order_id', '')  # ID único da Binance
        self.trades.append(trade)
        
        # Salvar historico em arquivo
        self._save_trade_history(trade)
        self._save_trade_csv(trade)
        
        # Remover posicao
        del self.positions[symbol]
        
        # Atualizar balance
        self.balance += net_pnl
        self.update_balance(self.balance)
        
        return net_pnl
    
    def get_stats(self) -> Dict:
        """Obter estatisticas."""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_pnl': 0,
                'balance': self.balance
            }
        
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl < 0]
        
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        
        return {
            'total_trades': len(self.trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / len(self.trades) * 100 if self.trades else 0,
            'profit_factor': gross_profit / gross_loss if gross_loss > 0 else 0,
            'total_pnl': sum(t.pnl for t in self.trades),
            'avg_win': gross_profit / len(wins) if wins else 0,
            'avg_loss': gross_loss / len(losses) if losses else 0,
            'balance': self.balance,
            'return_pct': (self.balance - self.initial_balance) / self.initial_balance * 100
        }
    
    def _save_trade_history(self, trade: Trade):
        """Salvar trade no historico JSON."""
        history_file = 'state/trade_history.json'

        # Carregar historico existente
        history = []
        try:
            if os.path.exists(history_file):
                history = load_json_safe(history_file, default=[])
        except Exception as e:
            log.warning(f"Erro carregando trade history: {e}")

        # CORRIGIDO: Gerar ID baseado no maior ID existente + 1 (não len())
        max_id = max([t.get('id', 0) for t in history], default=0) if history else 0

        # Verificar duplicatas pelo order_id (mais confiável)
        trade_order_id = getattr(trade, 'order_id', '')
        if trade_order_id:
            for existing in history[-100:]:
                if existing.get('order_id') == trade_order_id:
                    log.warning(f"Trade duplicado detectado (order_id={trade_order_id}), ignorando: {trade.symbol}")
                    return

        # Fallback: verificar por symbol + entry_price + quantity (para trades sem order_id)
        for existing in history[-50:]:
            is_same_symbol = existing.get('symbol') == trade.symbol
            is_same_side = existing.get('side') == trade.side
            is_same_entry_price = abs(existing.get('entry_price', 0) - trade.entry_price) < 0.01
            is_same_quantity = abs(existing.get('quantity', 0) - trade.quantity) < 0.0001

            if is_same_symbol and is_same_side and is_same_entry_price and is_same_quantity:
                log.warning(f"Trade duplicado detectado (price+qty), ignorando: {trade.symbol}")
                return

        # Adicionar novo trade
        trade_data = {
            'id': max_id + 1,
            'order_id': trade_order_id,
            'symbol': trade.symbol,
            'side': trade.side,
            'strategy': getattr(trade, 'strategy', 'unknown'),
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'quantity': trade.quantity,
            'pnl': trade.pnl,
            'pnl_gross': getattr(trade, 'pnl_gross', trade.pnl),
            'commission': getattr(trade, 'commission', 0),
            'funding_cost': getattr(trade, 'funding_cost', 0),
            'pnl_pct': getattr(trade, 'pnl_pct', 0),
            'entry_time': trade.entry_time,
            'exit_time': trade.exit_time,
            'reason_entry': trade.reason_entry,
            'reason_exit': trade.reason_exit,
        }
        history.append(trade_data)
        
        # Manter apenas ultimos 500 trades
        history = history[-500:]
        
        # Salvar
        save_json_atomic(history_file, history)

    def _save_trade_csv(self, trade: Trade):
        """Salvar trade em CSV (append)."""
        import csv
        log_file = 'logs/trades.csv'
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_exists = os.path.exists(log_file)
        
        try:
            with open(log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['id', 'symbol', 'side', 'entry_price', 'exit_price', 'quantity', 'pnl', 'pnl_pct', 'entry_time', 'exit_time', 'strategy', 'reason_entry', 'reason_exit'])
                
                writer.writerow([
                    getattr(trade, 'id', ''),
                    trade.symbol,
                    trade.side,
                    trade.entry_price,
                    trade.exit_price,
                    trade.quantity,
                    trade.pnl,
                    getattr(trade, 'pnl_pct', 0),
                    trade.entry_time,
                    trade.exit_time,
                    getattr(trade, 'strategy', 'unknown'),
                    trade.reason_entry,
                    trade.reason_exit
                ])
        except Exception as e:
            log.error(f"Erro salvando CSV: {e}")
    
    def save_state(self, filepath: str):
        """Salvar estado."""
        # Sempre usar params do config centralizado (fonte única de verdade)
        current_params = get_validated_params()
        # Mesclar com params atuais (config pode ter sido atualizado)
        merged_params = {**self.params, **current_params}

        state = {
            'balance': self.balance,
            'initial_balance': self.initial_balance,
            'high_water_mark': self.high_water_mark,
            'is_halted': self.is_halted,
            'positions': {
                k: {
                    'symbol': v.symbol,
                    'side': v.side,
                    'entry_price': v.entry_price,
                    'quantity': v.quantity,
                    'stop_loss': v.stop_loss,
                    'take_profit': v.take_profit,
                    'entry_time': v.entry_time,
                    'reason_entry': getattr(v, 'reason_entry', ''),
                    'strategy': getattr(v, 'strategy', merged_params.get('strategy', 'unknown'))
                }
                for k, v in self.positions.items()
            },
            'stats': self.get_stats(),
            'params': merged_params,  # Usar params mesclados
            'timestamp': datetime.now().isoformat()
        }

        save_json_atomic(filepath, state)
    
    def load_state(self, filepath: str):
        """Carregar estado."""
        if not os.path.exists(filepath):
            # Estado inicial limpo
            self.is_halted = False
            return

        try:
            state = load_json_safe(filepath)

            self.balance = state.get('balance', 10000)
            self.initial_balance = state.get('initial_balance', 10000)
            self.high_water_mark = state.get('high_water_mark', self.balance)
            # NAO carregar is_halted - deixar o update_balance decidir
            self.is_halted = False

            # Restaurar posicoes
            for k, v in state.get('positions', {}).items():
                self.positions[k] = Position(**v)

            # CORRIGIDO: Carregar histórico de trades para stats
            self._load_trade_history()
        except Exception as e:
            # Em caso de erro, estado limpo
            self.is_halted = False

    def _load_trade_history(self):
        """Carregar histórico de trades do arquivo."""
        history_file = 'state/trade_history.json'
        try:
            if os.path.exists(history_file):
                history = load_json_safe(history_file, default=[])

                # Converter para objetos Trade
                self.trades = []
                for t in history:
                    trade = Trade(
                        symbol=t.get('symbol', ''),
                        side=t.get('side', ''),
                        entry_price=t.get('entry_price', 0),
                        exit_price=t.get('exit_price', 0),
                        quantity=t.get('quantity', 0),
                        pnl=t.get('pnl', 0),
                        entry_time=t.get('entry_time', ''),
                        exit_time=t.get('exit_time', ''),
                        reason_entry=t.get('reason_entry', ''),
                        reason_exit=t.get('reason_exit', ''),
                        status='closed'
                    )
                    self.trades.append(trade)

                log.info(f"Historico carregado: {len(self.trades)} trades")
        except Exception as e:
            log.warning(f"Erro carregando historico: {e}")
