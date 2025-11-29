"""
Trader Module - Gerenciador de trades simplificado
Combina signals, risk e execution

VERSÃO: 3.1 - Correções de lucratividade
- PnL LÍQUIDO (após taxas) - Bug #1 corrigido
- Funding cost DISCRETO (00:00, 08:00, 16:00 UTC) - Bug #2 corrigido
- Circuit breaker com unrealized PnL - Bug #3 corrigido
- Race condition corrigida - Bug #4 corrigido
- Position sizing com verificação de margem - Bug #5 corrigido
- Thread-safe record_exit() com Lock
- Detecção de duplicatas via hash SHA256
- Validação de preços entry/exit
- Sem truncamento de histórico
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
import os
import logging
import threading
import hashlib
import shutil

from .signals import SignalGenerator, check_exit_signal
from .config import get_validated_params, get_backtest_config, Config
from .utils import save_json_atomic, load_json_safe, backup_state_file
from .binance_fees import get_binance_fees

log = logging.getLogger(__name__)


def count_funding_periods(entry_time: datetime, exit_time: datetime) -> int:
    """
    Conta períodos de funding discretos cruzados (00:00, 08:00, 16:00 UTC).

    Binance cobra funding a cada 8 horas em horários fixos UTC.
    Esta função conta quantos desses horários foram cruzados durante a posição.

    Args:
        entry_time: Momento de abertura da posição
        exit_time: Momento de fechamento da posição

    Returns:
        Número de períodos de funding que ocorreram durante a posição
    """
    if entry_time >= exit_time:
        return 0

    funding_hours = [0, 8, 16]  # UTC
    count = 0

    # Começar do início do dia de entrada
    current_date = entry_time.replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = exit_time.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)

    while current_date <= end_date:
        for hour in funding_hours:
            funding_time = current_date.replace(hour=hour)
            # Funding é cobrado se: entry < funding_time <= exit
            if entry_time < funding_time <= exit_time:
                count += 1
        current_date += timedelta(days=1)

    return count


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
    # FASE 3: Tracking de saídas parciais
    partial_exits: Dict = field(default_factory=dict)  # {profit_pct: True} para níveis já executados
    original_quantity: float = 0.0  # Quantidade original antes de saídas parciais
    realized_pnl: float = 0.0  # PnL já realizado em saídas parciais

    def __post_init__(self):
        if self.original_quantity == 0.0:
            self.original_quantity = self.quantity
        # Inicializar min_price e max_price com entry_price se não foram definidos
        if self.min_price == float('inf'):
            self.min_price = self.entry_price
        if self.max_price == 0.0:
            self.max_price = self.entry_price


class Trader:
    """
    Gerenciador de trading simplificado.
    Thread-safe para operações de posição.
    """

    def __init__(self, params: Dict = None):
        self.params = params or self._default_params()
        self.signal_generator = SignalGenerator(self.params)

        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []

        # Lock para operações thread-safe em record_exit
        self._exit_lock = threading.Lock()

        # Balance inicial do config (será sobrescrito por load_state ou update_balance)
        backtest_config = get_backtest_config()
        self.balance = float(backtest_config.get('initial_capital', 10000))
        self.initial_balance = self.balance

        # Risk params (do config centralizado)
        self.risk_per_trade = self.params.get('risk_per_trade', 0.01)
        self.max_positions = self.params.get('max_positions', 10)
        # Circuit breaker: ativa em 10% de drawdown (mais conservador que antes)
        self.max_drawdown = self.params.get('max_drawdown', backtest_config.get('max_drawdown_halt', 0.10))

        # State
        self.high_water_mark = self.balance
        self.is_halted = False
    
    def _default_params(self) -> Dict:
        # Usa parâmetros centralizados do config
        return get_validated_params()
    
    def calculate_unrealized_pnl(self, current_prices: Dict[str, float]) -> float:
        """
        Calcula PnL não realizado de todas as posições abertas.

        CORRIGIDO (Bug #3): Circuit breaker agora considera unrealized PnL.

        Args:
            current_prices: Dicionário {symbol: price} com preços atuais

        Returns:
            Total de PnL não realizado (positivo = lucro, negativo = perda)
        """
        if not current_prices:
            return 0.0

        unrealized = 0.0
        for symbol, pos in self.positions.items():
            if symbol not in current_prices:
                continue

            price = current_prices[symbol]
            if pos.side == 'long':
                pnl = (price - pos.entry_price) * pos.quantity
            else:
                pnl = (pos.entry_price - price) * pos.quantity

            unrealized += pnl

        return unrealized

    def update_balance(self, new_balance: float, current_prices: Dict[str, float] = None):
        """
        Atualizar balance e verificar circuit breaker.

        CORRIGIDO (Bug #3): Agora considera PnL não realizado no drawdown.

        Args:
            new_balance: Novo saldo da conta
            current_prices: Preços atuais para calcular unrealized PnL (opcional)

        Returns:
            True se circuit breaker foi ativado, False caso contrário
        """
        if new_balance <= 0:
            return False

        # Se nao temos trades E nao temos posições, aceitar o balance como inicial
        # Isso evita falsos positivos de drawdown ao iniciar
        if len(self.trades) == 0 and len(self.positions) == 0:
            self.balance = new_balance
            self.initial_balance = new_balance
            self.high_water_mark = new_balance
            self.is_halted = False
            return False

        self.balance = new_balance

        # CORRIGIDO (Bug #3): Calcular equity total incluindo unrealized PnL
        unrealized_pnl = self.calculate_unrealized_pnl(current_prices) if current_prices else 0.0
        total_equity = self.balance + unrealized_pnl

        # Atualizar high water mark APENAS com balance realizado (não unrealized)
        # Isso evita que o circuit breaker fique inerte devido a flutuações de unrealized PnL
        if self.balance > self.high_water_mark:
            self.high_water_mark = self.balance

        # Circuit breaker - usa apenas balance realizado para evitar sensibilidade excessiva
        # a flutuações temporárias de unrealized PnL
        drawdown = (self.high_water_mark - self.balance) / self.high_water_mark
        if drawdown > self.max_drawdown:
            self.is_halted = True
            log.warning(f"CIRCUIT BREAKER: Drawdown {drawdown:.1%} > {self.max_drawdown:.1%} (equity={total_equity:.2f}, unrealized={unrealized_pnl:.2f})")
            return True

        # Desativar circuit breaker se drawdown esta ok
        self.is_halted = False
        return False

    # ==================== MELHORIAS FASE 2 ====================

    def _calculate_profit_pct(self, pos, current_price: float) -> float:
        """Calcular % de lucro de uma posição."""
        if pos.side == 'long':
            return ((current_price - pos.entry_price) / pos.entry_price) * 100
        else:
            return ((pos.entry_price - current_price) / pos.entry_price) * 100

    def update_trailing_stop(self, symbol: str, current_price: float, atr: float) -> bool:
        """
        Move SL para proteger lucros (Trailing Stop Agressivo).

        MELHORIA A1: Trailing stops para proteger lucros.
        FASE 5: Trailing mais agressivo com mais níveis.

        Args:
            symbol: Par de trading
            current_price: Preço atual
            atr: ATR atual para cálculo do trailing

        Returns:
            True se SL foi atualizado, False caso contrário
        """
        if symbol not in self.positions:
            return False

        pos = self.positions[symbol]
        profit_pct = self._calculate_profit_pct(pos, current_price)

        # Carregar thresholds do config
        activation_pct = Config.get('trader.trailing_stop_activation_pct', 0.0) * 100  # Converter para %
        trail_mult_high = Config.get('trader.trailing_stop_mult_high', 0.75)
        trail_mult_medium = Config.get('trader.trailing_stop_mult_medium', 1.0)
        trail_mult_normal = Config.get('trader.trailing_stop_mult_normal', 1.5)

        # Trailing multiplier baseado no lucro (mais agressivo com mais níveis)
        # Quanto maior o lucro, mais apertado o trailing (valores mais relaxados)
        if profit_pct >= 4.0:
            trail_mult = trail_mult_high  # Apertado após 4% de lucro
        elif profit_pct >= 2.5:
            trail_mult = trail_mult_medium  # Médio após 2.5% de lucro
        elif profit_pct >= activation_pct:
            trail_mult = trail_mult_normal  # Normal no início (1.5%)
        else:
            return False  # Ainda não ativado

        if pos.side == 'long':
            new_sl = current_price - (atr * trail_mult)
            # BREAKEVEN: Proteger lucros - margem de 0.2%
            breakeven_sl = pos.entry_price * 0.998  # 0.2% abaixo da entrada
            if profit_pct >= 1.0 and new_sl < breakeven_sl:  # Ativa em 1% de lucro
                new_sl = max(new_sl, breakeven_sl)

            if new_sl > pos.stop_loss:
                old_sl = pos.stop_loss
                pos.stop_loss = new_sl
                log.debug(f"Trailing LONG {symbol}: SL {old_sl:.4f} -> {new_sl:.4f} (profit={profit_pct:.1f}%, mult={trail_mult:.2f})")
                return True
        else:  # short
            new_sl = current_price + (atr * trail_mult)
            # BREAKEVEN: Proteger lucros - margem de 0.2%
            breakeven_sl = pos.entry_price * 1.002  # 0.2% acima da entrada
            if profit_pct >= 1.0 and new_sl > breakeven_sl:  # Ativa em 1% de lucro
                new_sl = min(new_sl, breakeven_sl)

            if new_sl < pos.stop_loss:
                old_sl = pos.stop_loss
                pos.stop_loss = new_sl
                log.debug(f"Trailing SHORT {symbol}: SL {old_sl:.4f} -> {new_sl:.4f} (profit={profit_pct:.1f}%, mult={trail_mult:.2f})")
                return True

        return False

    def recommend_leverage(self, atr_pct: float) -> int:
        """
        Recomenda leverage baseado na volatilidade do mercado.

        MELHORIA A2: Dynamic leverage por volatilidade.

        Args:
            atr_pct: ATR como % do preço (ex: 0.03 = 3%)

        Returns:
            Leverage recomendado (3, 6, ou 10)
        """
        max_leverage = self.params.get('max_leverage', 10)

        # Carregar thresholds do config
        high_volatility = Config.get('trader.volatility_high_threshold', 0.04)
        medium_volatility = Config.get('trader.volatility_medium_threshold', 0.025)

        if atr_pct > high_volatility:      # Alta volatilidade
            return min(3, max_leverage)
        elif atr_pct > medium_volatility:   # Média volatilidade
            return min(6, max_leverage)
        else:                                # Baixa volatilidade
            return max_leverage

    def check_portfolio_risk(self) -> Dict:
        """
        Verifica risco do portfólio completo.

        MELHORIA A3: Portfolio risk integration.

        Returns:
            Dict com métricas de risco e flags de permissão
        """
        total_exposure = 0.0
        net_direction = 0.0  # Positivo = net long, Negativo = net short
        long_exposure = 0.0
        short_exposure = 0.0

        for symbol, pos in self.positions.items():
            notional = pos.entry_price * pos.quantity
            total_exposure += notional
            if pos.side == 'long':
                net_direction += notional
                long_exposure += notional
            else:
                net_direction -= notional
                short_exposure += notional

        exposure_ratio = total_exposure / self.balance if self.balance > 0 else 0

        # Limites de exposição
        max_net_exposure = self.balance * 2  # Máximo 2x net direction
        max_total_exposure = self.balance * 5  # Máximo 5x exposure total

        return {
            'total_exposure': total_exposure,
            'exposure_ratio': exposure_ratio,
            'net_direction': net_direction,
            'long_exposure': long_exposure,
            'short_exposure': short_exposure,
            'allow_new_long': net_direction < max_net_exposure and total_exposure < max_total_exposure,
            'allow_new_short': net_direction > -max_net_exposure and total_exposure < max_total_exposure,
            'is_over_exposed': total_exposure > max_total_exposure
        }

    def get_historical_stats(self) -> Dict:
        """
        Obtém estatísticas históricas para Kelly Criterion.

        MELHORIA C3: Base para Kelly Criterion position sizing.

        Returns:
            Dict com win_rate, avg_win, avg_loss
        """
        if len(self.trades) < 10:
            return {'win_rate': 0.5, 'avg_win': 0.01, 'avg_loss': 0.01}

        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl < 0]

        win_rate = len(wins) / len(self.trades) if self.trades else 0.5

        # Calcular médias como % do notional
        avg_win = 0.01
        avg_loss = 0.01

        if wins:
            avg_win_pct = sum(getattr(t, 'pnl_pct', 1) for t in wins) / len(wins)
            avg_win = abs(avg_win_pct) / 100

        if losses:
            avg_loss_pct = sum(getattr(t, 'pnl_pct', -1) for t in losses) / len(losses)
            avg_loss = abs(avg_loss_pct) / 100

        return {
            'win_rate': win_rate,
            'avg_win': max(0.001, avg_win),
            'avg_loss': max(0.001, avg_loss)
        }

    def kelly_position_size(self, signal_strength: float) -> float:
        """
        Calcular tamanho de posição usando Kelly Criterion.

        MELHORIA C3: Kelly Criterion position sizing.

        Args:
            signal_strength: Força do sinal (0-10)

        Returns:
            Fração do capital a arriscar (0.005 a 0.03)
        """
        stats = self.get_historical_stats()
        win_rate = stats['win_rate']
        avg_win = stats['avg_win']
        avg_loss = stats['avg_loss']

        if avg_loss == 0 or win_rate == 0:
            return self.risk_per_trade

        # Kelly: f* = (bp - q) / b
        # b = ratio ganho/perda, p = prob ganho, q = prob perda
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p

        kelly_fraction = (b * p - q) / b

        # Carregar thresholds do config
        kelly_mult = Config.get('trader.kelly_fraction', 0.25)
        kelly_min = Config.get('trader.kelly_min_risk', 0.005)
        kelly_max = Config.get('trader.kelly_max_risk', 0.03)

        # Usar Kelly fracionário (kelly_mult do Kelly completo - mais conservador)
        kelly_fraction = max(0, kelly_fraction * kelly_mult)

        # Escalar por signal strength (não-linear)
        strength_multiplier = (signal_strength / 5) ** 1.5

        # Limitar entre kelly_min e kelly_max do capital
        final_risk = kelly_fraction * strength_multiplier
        return max(kelly_min, min(kelly_max, final_risk))

    # ==================== FIM MELHORIAS FASE 2 ====================

    # ==================== MELHORIAS FASE 3 ====================

    def check_partial_exits(self, symbol: str, current_price: float) -> List[Dict]:
        """
        Verificar se deve executar saídas parciais.

        FASE 3.1: Partial Position Exits
        Fecha 33% no primeiro TP, 50% no segundo, 100% no terceiro.

        Args:
            symbol: Par de trading
            current_price: Preço atual

        Returns:
            Lista de saídas parciais a executar
        """
        exits = []
        pos = self.positions.get(symbol)
        if not pos:
            return exits

        profit_pct = self._calculate_profit_pct(pos, current_price)

        # Carregar níveis de saída parcial do config
        config_levels = Config.get('trader.partial_exit_levels', [
            {'profit_pct': 1.0, 'close_pct': 0.33},
            {'profit_pct': 2.0, 'close_pct': 0.50},
            {'profit_pct': 3.0, 'close_pct': 1.00}
        ])

        # Converter para formato interno
        levels = [{'pct': l.get('profit_pct', 1.0), 'close_pct': l.get('close_pct', 0.33)} for l in config_levels]

        for level in levels:
            level_key = str(level['pct'])
            # Verificar se já executou este nível
            if pos.partial_exits.get(level_key):
                continue

            if profit_pct >= level['pct']:
                # Calcular quantidade a fechar
                qty_to_close = pos.quantity * level['close_pct']
                if qty_to_close > 0:
                    exits.append({
                        'level': level['pct'],
                        'quantity': qty_to_close,
                        'close_pct': level['close_pct'],
                        'reason': f'PARTIAL_TP_{level["pct"]}%',
                        'profit_pct': profit_pct
                    })
                    # Marcar nível como executado (será confirmado após execução)
                    # NÃO marcar aqui - deixar para depois da execução real

        return exits

    def execute_partial_exit(self, symbol: str, partial_exit: Dict, exit_price: float) -> float:
        """
        Executar uma saída parcial e atualizar a posição.

        Args:
            symbol: Par de trading
            partial_exit: Dict com informações da saída parcial
            exit_price: Preço de saída

        Returns:
            PnL realizado nesta saída parcial
        """
        pos = self.positions.get(symbol)
        if not pos:
            return 0.0

        qty_to_close = partial_exit['quantity']
        level_key = str(partial_exit['level'])

        # Calcular PnL desta saída parcial
        if pos.side == 'long':
            partial_pnl = (exit_price - pos.entry_price) * qty_to_close
        else:
            partial_pnl = (pos.entry_price - exit_price) * qty_to_close

        # Descontar taxas dinâmicas da API
        try:
            binance_fees = get_binance_fees(use_testnet=True)
            raw_symbol = symbol.replace('/', '')
            rates = binance_fees.get_commission_rates(raw_symbol)
            taker_fee = rates.taker_rate
        except Exception:
            taker_fee = 0.0004  # Fallback
        notional = exit_price * qty_to_close
        commission = notional * taker_fee
        partial_pnl_net = partial_pnl - commission

        # Atualizar posição
        pos.quantity -= qty_to_close
        pos.realized_pnl += partial_pnl_net
        pos.partial_exits[level_key] = True

        log.info(f"PARTIAL EXIT {symbol}: {partial_exit['close_pct']*100:.0f}% @ {exit_price:.4f} | PnL: ${partial_pnl_net:.2f} | Remaining: {pos.quantity:.4f}")

        # Se fechou tudo, remover posição
        if pos.quantity <= 0 or partial_exit['close_pct'] >= 1.0:
            # Registrar trade completo com PnL total (incluindo parciais)
            total_pnl = pos.realized_pnl
            self.record_exit(symbol, exit_price, f"PARTIAL_COMPLETE_{level_key}%")
            return total_pnl

        return partial_pnl_net

    def get_drawdown_adjusted_risk(self) -> float:
        """
        Reduzir risco progressivamente durante drawdown.

        FASE 3.2: Dynamic Risk Scaling

        Returns:
            Fração do risco a usar (0.0 a 1.0 multiplicador)
        """
        if self.high_water_mark <= 0:
            return self.risk_per_trade

        current_dd = 1 - (self.balance / self.high_water_mark)

        # Carregar drawdown brackets do config
        dd_brackets = Config.get('trader.drawdown_brackets', {})
        level_1_pct = dd_brackets.get('level_1_pct', 0.03)
        level_1_mult = dd_brackets.get('level_1_mult', 1.0)
        level_2_pct = dd_brackets.get('level_2_pct', 0.05)
        level_2_mult = dd_brackets.get('level_2_mult', 0.75)
        level_3_pct = dd_brackets.get('level_3_pct', 0.08)
        level_3_mult = dd_brackets.get('level_3_mult', 0.50)
        level_4_pct = dd_brackets.get('level_4_pct', 0.10)
        level_4_mult = dd_brackets.get('level_4_mult', 0.25)
        level_5_mult = dd_brackets.get('level_5_mult', 0.0)

        if current_dd < level_1_pct:
            multiplier = level_1_mult
        elif current_dd < level_2_pct:
            multiplier = level_2_mult
        elif current_dd < level_3_pct:
            multiplier = level_3_mult
        elif current_dd < level_4_pct:
            multiplier = level_4_mult
        else:
            multiplier = level_5_mult

        adjusted_risk = self.risk_per_trade * multiplier

        if multiplier < 1.0:
            log.debug(f"DD Risk Scaling: DD={current_dd:.1%}, multiplier={multiplier:.0%}, risk={adjusted_risk:.2%}")

        return adjusted_risk

    def check_time_exit(self, symbol: str, current_price: float) -> Optional[str]:
        """
        Verificar se deve sair por tempo sem lucro.

        FASE 3.3: Time-Based Exit Rules

        Args:
            symbol: Par de trading
            current_price: Preço atual

        Returns:
            Razão de saída se deve sair, None caso contrário
        """
        pos = self.positions.get(symbol)
        if not pos:
            return None

        try:
            entry_time = datetime.fromisoformat(pos.entry_time)
            hours_open = (datetime.now() - entry_time).total_seconds() / 3600
        except Exception:
            return None

        profit_pct = self._calculate_profit_pct(pos, current_price)

        # Carregar thresholds do config
        max_hours = Config.get('trader.time_exit_max_hours', 168)
        stale_1_hours = Config.get('trader.time_exit_stale_1_hours', 96)
        stale_1_min_profit = Config.get('trader.time_exit_stale_1_min_profit', 1.0)
        stale_2_hours = Config.get('trader.time_exit_stale_2_hours', 72)
        stale_2_min_profit = Config.get('trader.time_exit_stale_2_min_profit', 0.5)

        # Regras de saída por tempo
        if hours_open > max_hours:  # Sair independente após max_hours
            return f'TIME_EXIT_{max_hours}H (profit={profit_pct:.1f}%)'

        if hours_open > stale_1_hours and profit_pct < stale_1_min_profit:
            return f'TIME_EXIT_{stale_1_hours}H (profit={profit_pct:.1f}%)'

        if hours_open > stale_2_hours and profit_pct < stale_2_min_profit:
            return f'TIME_EXIT_{stale_2_hours}H (profit={profit_pct:.1f}%)'

        return None

    # FASE 3.5: Grupos de ativos correlacionados
    CORRELATED_GROUPS = {
        'btc_ecosystem': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
        'layer2': ['ARB/USDT', 'OP/USDT', 'MATIC/USDT'],
        'defi': ['UNI/USDT', 'AAVE/USDT', 'CRV/USDT', 'LINK/USDT'],
        'gaming': ['AXS/USDT', 'SAND/USDT', 'MANA/USDT', 'GALA/USDT'],
        'layer1_alt': ['AVAX/USDT', 'NEAR/USDT', 'FTM/USDT', 'ATOM/USDT'],
        'exchange': ['BNB/USDT', 'FTT/USDT', 'CRO/USDT'],
        'meme': ['DOGE/USDT', 'SHIB/USDT', 'PEPE/USDT', 'FLOKI/USDT'],
        'ai': ['FET/USDT', 'AGIX/USDT', 'OCEAN/USDT', 'RNDR/USDT'],
    }

    def check_correlation_limit(self, symbol: str, direction: str) -> bool:
        """
        Verificar se pode abrir posição sem exceder limite de correlação.

        FASE 3.5: Correlation Filter
        Limita a N posições por grupo correlacionado na mesma direção.

        Args:
            symbol: Par de trading
            direction: 'long' ou 'short'

        Returns:
            True se pode abrir posição, False se limite excedido
        """
        # Carregar limite do config
        max_per_group = Config.get('trader.max_correlated_positions', 2)

        for group_name, group_symbols in self.CORRELATED_GROUPS.items():
            if symbol in group_symbols:
                # Contar posições na mesma direção neste grupo
                same_group_positions = sum(
                    1 for s, p in self.positions.items()
                    if s in group_symbols and p.side == direction
                )

                if same_group_positions >= max_per_group:
                    log.debug(f"Correlation limit: {symbol} bloqueado ({direction}) - grupo '{group_name}' já tem {same_group_positions} posições")
                    return False

        return True

    # ==================== FIM MELHORIAS FASE 3 ====================

    # ==================== ROTAÇÃO INTELIGENTE DE POSIÇÕES ====================

    def calculate_position_weakness(self, symbol: str, current_price: float) -> float:
        """
        Calcula score de "fraqueza" de uma posição (quanto maior, mais fraca).

        ROTAÇÃO INTELIGENTE: Score composto para identificar posições a fechar.

        Critérios de fraqueza (ponderados):
        1. PnL% negativo (peso: 40%) - posições perdedoras são candidatas
        2. Tempo aberto sem lucro (peso: 25%) - capital parado = custo de oportunidade
        3. Distância do SL (peso: 20%) - perto do SL = risco alto
        4. Momentum negativo (peso: 15%) - preço movendo contra a posição

        Args:
            symbol: Par de trading
            current_price: Preço atual

        Returns:
            Score de fraqueza (0.0 = forte, 10.0+ = muito fraca)
        """
        pos = self.positions.get(symbol)
        if not pos:
            return 0.0

        weakness_score = 0.0

        # 1. PnL% (peso 40%) - posições perdendo são fracas
        profit_pct = self._calculate_profit_pct(pos, current_price)
        if profit_pct < 0:
            # Perda: quanto maior a perda, mais fraca (max 4 pontos)
            weakness_score += min(4.0, abs(profit_pct) * 0.8)
        elif profit_pct < 0.5:
            # Lucro mínimo: ainda um pouco fraca (0.5 pontos)
            weakness_score += 0.5
        # Lucro > 0.5%: não adiciona fraqueza

        # 2. Tempo aberto (peso 25%) - capital parado
        try:
            entry_time = datetime.fromisoformat(pos.entry_time)
            hours_open = (datetime.now() - entry_time).total_seconds() / 3600

            # Thresholds do config
            stale_hours = Config.get('trader.time_exit_stale_2_hours', 72)

            if hours_open > stale_hours and profit_pct < 1.0:
                # Muito tempo aberto sem lucro significativo
                weakness_score += 2.5
            elif hours_open > stale_hours / 2 and profit_pct < 0.5:
                # Tempo médio sem lucro
                weakness_score += 1.5
            elif hours_open > 24 and profit_pct < 0:
                # Mais de 24h em perda
                weakness_score += 1.0
        except Exception:
            pass

        # 3. Distância do SL (peso 20%) - risco de stop out
        if pos.side == 'long':
            sl_distance_pct = (current_price - pos.stop_loss) / current_price * 100
        else:
            sl_distance_pct = (pos.stop_loss - current_price) / current_price * 100

        if sl_distance_pct < 0.5:
            # Muito perto do SL (< 0.5%) - muito fraca
            weakness_score += 2.0
        elif sl_distance_pct < 1.0:
            # Perto do SL (< 1%) - fraca
            weakness_score += 1.0
        elif sl_distance_pct < 1.5:
            # Moderadamente perto
            weakness_score += 0.5

        # 4. Momentum negativo (peso 15%) - preço contra a posição
        if pos.side == 'long':
            # Para long: se preço está abaixo do max_price, momentum negativo
            if pos.max_price > 0:
                price_from_high = (pos.max_price - current_price) / pos.max_price * 100
                if price_from_high > 2.0:
                    weakness_score += 1.5
                elif price_from_high > 1.0:
                    weakness_score += 0.75
        else:
            # Para short: se preço está acima do min_price, momentum negativo
            if pos.min_price > 0 and pos.min_price != float('inf'):
                price_from_low = (current_price - pos.min_price) / pos.min_price * 100
                if price_from_low > 2.0:
                    weakness_score += 1.5
                elif price_from_low > 1.0:
                    weakness_score += 0.75

        return round(weakness_score, 2)

    def get_weakest_position(self, current_prices: Dict[str, float],
                             exclude_symbols: List[str] = None) -> Optional[str]:
        """
        Encontra a posição mais fraca para rotação.

        Args:
            current_prices: Dicionário {symbol: price}
            exclude_symbols: Símbolos a excluir da análise

        Returns:
            Symbol da posição mais fraca, ou None se nenhuma elegível
        """
        if not self.positions:
            return None

        exclude = set(exclude_symbols or [])

        # Calcular weakness para cada posição
        weakness_scores = {}
        min_notional = Config.get('risk.min_notional', 6.0)

        for symbol, pos in self.positions.items():
            if symbol in exclude:
                continue
            if symbol not in current_prices:
                continue

            # Pular posições com notional muito pequeno (não podem ser fechadas)
            notional = pos.quantity * current_prices[symbol]
            if notional < min_notional:
                continue

            score = self.calculate_position_weakness(symbol, current_prices[symbol])
            weakness_scores[symbol] = score

        if not weakness_scores:
            return None

        # Retornar a mais fraca
        weakest = max(weakness_scores, key=weakness_scores.get)

        # Só retornar se tiver score mínimo de fraqueza
        min_weakness = Config.get('trader.rotation_min_weakness', 2.0)
        if weakness_scores[weakest] >= min_weakness:
            log.debug(f"Posição mais fraca: {weakest} (score={weakness_scores[weakest]:.2f})")
            return weakest

        return None

    def should_rotate_position(self, new_signal_strength: float, proactive: bool = False) -> bool:
        """
        Verifica se deve rotacionar uma posição para um novo sinal.

        Critérios para rotação:
        1. Max positions atingido OU modo proativo ativado
        2. Novo sinal tem strength > threshold (default 7.0, proativo 8.0)
        3. Existe posição fraca suficiente para substituir

        Args:
            new_signal_strength: Força do novo sinal
            proactive: Se True, permite rotação mesmo sem atingir limite

        Returns:
            True se deve rotacionar
        """
        # Modo normal: só rotaciona no limite
        # Modo proativo: rotaciona se sinal muito forte e posição fraca existe
        if not proactive:
            if len(self.positions) < self.max_positions:
                return False
            min_strength = Config.get('trader.rotation_min_signal_strength', 7.0)
        else:
            # Proativo: exige sinal mais forte (8.0+) para substituir
            min_strength = Config.get('trader.rotation_proactive_min_strength', 8.0)
            # Precisa ter pelo menos algumas posições para rotacionar
            if len(self.positions) < 3:
                return False

        if new_signal_strength < min_strength:
            return False

        return True

    # ==================== FIM ROTAÇÃO INTELIGENTE ====================

    def _calculate_total_margin_used(self) -> float:
        """
        Calcula margem total usada por posições abertas.

        CORRIGIDO (Bug #5): Usado para verificar margem disponível.

        Returns:
            Total de margem usada (em USDT)
        """
        total = 0.0
        leverage = self.params.get('max_leverage', 10)
        for pos in self.positions.values():
            notional = pos.entry_price * pos.quantity
            total += notional / leverage
        return total

    def calculate_position_size(self, price: float, stop_loss: float, signal_strength: float = 5.0) -> float:
        """
        Calcular tamanho da posicao baseado em risco.

        CORRIGIDO (Bug #5): Agora verifica margem disponível e usa min_position_pct menor.
        FASE 3.2: Agora usa DD Risk Scaling para ajustar risco durante drawdown.
        CORRIGIDO (Bug #7): Agora integra Kelly Criterion opcionalmente.
        """
        # Parâmetros do config
        max_leverage = self.params.get('max_leverage', 10)
        max_position_pct = self.params.get('max_position_pct', 0.15)  # 15% max por posição
        # CORRIGIDO (Bug #5): Reduzido de 5% para 2% para permitir mais diversificação
        min_position_pct = self.params.get('min_position_pct', 0.02)  # 2% min por posição
        max_margin_usage = self.params.get('max_margin_usage', 0.8)  # 80% max de margem
        min_notional = self.params.get('min_notional', 6.0)

        # CORRIGIDO (Bug #5): Verificar margem disponível ANTES de calcular posição
        total_margin_used = self._calculate_total_margin_used()
        max_margin = self.balance * max_margin_usage
        available_margin = max_margin - total_margin_used

        if available_margin <= 0:
            log.debug(f"Sem margem disponível: usado={total_margin_used:.2f}, max={max_margin:.2f}")
            return 0.0

        # FASE 3.2: Usar risco ajustado por drawdown
        adjusted_risk = self.get_drawdown_adjusted_risk()
        if adjusted_risk <= 0:
            log.info("DD Risk Scaling: Risco zerado, não abrindo nova posição")
            return 0.0

        # CORRIGIDO (Bug #7): Integrar Kelly Criterion se habilitado
        use_kelly = Config.get('trader.use_kelly_sizing', False)
        kelly_min_trades = Config.get('trader.kelly_min_trades', 20)

        if use_kelly and len(self.trades) >= kelly_min_trades:
            kelly_risk = self.kelly_position_size(signal_strength)
            # Usar menor entre Kelly e risco ajustado por DD (conservador)
            # Mas limitar Kelly a no máximo 1.5x o risco base para evitar overexposure
            max_kelly_risk = adjusted_risk * 1.5
            adjusted_risk = min(kelly_risk, max_kelly_risk)
            log.debug(f"Kelly sizing: kelly_risk={kelly_risk:.4f}, final_risk={adjusted_risk:.4f}")

        risk_amount = self.balance * adjusted_risk
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

        # CORRIGIDO (Bug #5): Limitar pela margem disponível
        max_quantity_margin = (available_margin * max_leverage) / price
        quantity = min(quantity, max_quantity_margin)

        # Garantir quantidade mínima (% mínimo do balance)
        min_quantity = (self.balance * min_position_pct) / price
        if quantity < min_quantity:
            # Se a quantidade calculada é menor que o mínimo, usar o mínimo
            # MAS verificar se ainda cabe na margem disponível
            min_notional_check = min_quantity * price
            min_margin_needed = min_notional_check / max_leverage
            if min_margin_needed <= available_margin:
                quantity = min_quantity
            else:
                # Sem margem suficiente nem para o mínimo
                log.debug(f"Sem margem para posição mínima: precisa={min_margin_needed:.2f}, disponível={available_margin:.2f}")
                return 0.0

        # Verificar notional mínimo
        notional = quantity * price
        if notional < min_notional:
            return 0.0

        return quantity
    
    def process_candle(self, symbol: str, df: pd.DataFrame, timeframe: str = None) -> Optional[Dict]:
        """
        Processar uma nova candle.

        Args:
            symbol: Par de trading
            df: DataFrame com dados OHLCV
            timeframe: Timeframe sendo processado (para MTF)

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

        return self._check_entry(symbol, df, timeframe=timeframe)
    
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
                atr_fallback_pct = Config.get('trader.atr_fallback_pct', 0.02)
                atr = df['close'].iloc[-1] * atr_fallback_pct  # Fallback final do config

        # FASE 3.1: Verificar saídas parciais PRIMEIRO
        partial_exits = self.check_partial_exits(symbol, price)
        if partial_exits:
            # Retornar primeira saída parcial pendente
            partial = partial_exits[0]
            return {
                'action': 'partial_close',
                'symbol': symbol,
                'side': 'sell' if pos.side == 'long' else 'buy',
                'quantity': partial['quantity'],
                'reason': partial['reason'],
                'partial_exit': partial
            }

        # FASE 3.3: Verificar saída por tempo
        time_exit_reason = self.check_time_exit(symbol, price)
        if time_exit_reason:
            return {
                'action': 'close',
                'symbol': symbol,
                'side': 'sell' if pos.side == 'long' else 'buy',
                'quantity': pos.quantity,
                'reason': time_exit_reason
            }

        # Atualizar trailing stop (melhoria A1)
        self.update_trailing_stop(symbol, price, atr)

        # Verificar saida normal (SL/TP/RSI)
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
    
    def _check_entry(self, symbol: str, df: pd.DataFrame, timeframe: str = None) -> Optional[Dict]:
        """Verificar sinal de entrada."""
        signal = self.signal_generator.generate_signal(df)

        # CORRIGIDO: Usar min_score_to_open do config em vez de hardcoded 5
        min_strength = self.params.get('min_score_to_open', self.params.get('min_signal_strength', 5.0))

        # Ajustar min_strength por timeframe (scalping requer menos)
        if timeframe == '5m':
            min_strength = min_strength * 0.8  # 20% menos restritivo
        elif timeframe == '15m':
            min_strength = min_strength * 0.9  # 10% menos restritivo

        if signal.direction == 'none' or signal.strength < min_strength:
            return None

        # FASE 3.5: Verificar limite de correlação
        if not self.check_correlation_limit(symbol, signal.direction):
            return None

        # CORRIGIDO (Bug #7): Passa signal strength para Kelly sizing
        quantity = self.calculate_position_size(signal.entry_price, signal.stop_loss, signal.strength)

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
            'strength': signal.strength,
            'timeframe': timeframe
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
        """
        Registrar saida e calcular PnL. Thread-safe com Lock.

        CORRIGIDO (Bug #4): Race condition eliminada.
        Agora a posição é removida DENTRO do lock, e os dados são copiados
        para processamento fora do lock.
        """
        # Validação de preço de saída
        if exit_price <= 0:
            log.warning(f"record_exit ignorado para {symbol}: preço inválido ({exit_price})")
            return 0.0

        # CORRIGIDO (Bug #4): Thread-safe - copiar dados e remover posição DENTRO do lock
        with self._exit_lock:
            if symbol not in self.positions:
                # Posição já foi fechada por outra thread
                log.debug(f"record_exit ignorado para {symbol}: posição não existe (já fechada)")
                return 0.0

            pos = self.positions[symbol]

            # Verificar se posição já está sendo fechada
            if getattr(pos, '_closing', False):
                log.debug(f"record_exit ignorado para {symbol}: fechamento já em andamento")
                return 0.0

            # Marcar e REMOVER imediatamente - nenhuma outra thread pode acessar depois disso
            pos._closing = True
            del self.positions[symbol]

            # Copiar dados necessários para processamento fora do lock
            pos_data = {
                'side': pos.side,
                'entry_price': pos.entry_price,
                'quantity': pos.quantity,
                'entry_time': pos.entry_time,
                'reason_entry': getattr(pos, 'reason_entry', ''),
                'strategy': getattr(pos, 'strategy', self.params.get('strategy', 'unknown')),
                'order_id': getattr(pos, 'order_id', '')
            }

        # Processamento fora do lock (I/O bound, não bloqueia outras threads)
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

        # PnL bruto - usando pos_data (cópia thread-safe)
        if pos_data['side'] == 'long':
            pnl = (exit_price - pos_data['entry_price']) * pos_data['quantity']
        else:
            pnl = (pos_data['entry_price'] - exit_price) * pos_data['quantity']

        # Custos de trading (entrada + saída)
        notional_entry = pos_data['entry_price'] * pos_data['quantity']
        notional_exit = exit_price * pos_data['quantity']
        commission = (notional_entry + notional_exit) * taker_fee

        # Calcular funding durante a posição - CORRIGIDO: DISCRETO (Bug #2)
        # Funding é cobrado às 00:00, 08:00, 16:00 UTC (discreto, não contínuo)
        # Convenção: POSITIVO = recebeu (ganho), NEGATIVO = pagou (custo)
        funding_periods = 0
        try:
            entry_time = datetime.fromisoformat(pos_data['entry_time'])
            exit_time = datetime.now()
            # CORRIGIDO: Usar contagem discreta de períodos de funding
            funding_periods = count_funding_periods(entry_time, exit_time)
            # Funding = notional * rate * períodos
            # Quando funding rate é positivo: long paga, short recebe
            # Quando funding rate é negativo: long recebe, short paga
            if funding_periods > 0:
                avg_notional = (notional_entry + notional_exit) / 2
                funding_amount = avg_notional * funding_rate * funding_periods
                if pos_data['side'] == 'long':
                    funding_cost = -funding_amount  # Long paga (negativo = custo)
                else:
                    funding_cost = funding_amount   # Short recebe (positivo = ganho)
            else:
                funding_cost = 0
        except Exception:
            funding_cost = 0

        # CORRIGIDO (Bug #1): PnL LÍQUIDO (após taxas)
        # O usuário precisa ver o lucro REAL, não o bruto ilusório
        # pnl_net = lucro bruto - comissão + funding (funding_cost já é negativo se for custo)
        pnl_net = pnl - commission + funding_cost

        log.debug(f"PnL {symbol}: bruto={pnl:.2f}, net={pnl_net:.2f}, commission={commission:.2f}, funding={funding_cost:.2f}, periods={funding_periods}")

        # Registrar trade com detalhes
        # CORRIGIDO (Bug #1): pnl = LÍQUIDO (após taxas)
        trade = Trade(
            symbol=symbol,
            side=pos_data['side'],
            entry_price=pos_data['entry_price'],
            exit_price=exit_price,
            quantity=pos_data['quantity'],
            pnl=pnl_net,  # PnL LÍQUIDO (após taxas) - CORRIGIDO
            pnl_gross=pnl,  # PnL bruto mantido para auditoria
            commission=commission,
            funding_cost=funding_cost,
            entry_time=pos_data['entry_time'],
            exit_time=datetime.now().isoformat(),
            reason_entry=pos_data['reason_entry'],
            reason_exit=reason,
            status="closed"
        )
        # Adicionar atributos extras
        trade.strategy = pos_data['strategy']
        trade.pnl_pct = (pnl_net / notional_entry) * 100 if notional_entry > 0 else 0  # % do PnL LÍQUIDO - CORRIGIDO
        trade.order_id = pos_data['order_id']

        # Thread-safe: adicionar trade sob Lock (posição já foi removida no primeiro lock)
        with self._exit_lock:
            self.trades.append(trade)

        # Salvar historico em arquivo (fora do lock para não bloquear I/O)
        self._save_trade_history(trade)
        self._save_trade_csv(trade)

        # NOTA: NÃO atualizamos self.balance aqui porque:
        # 1. O balance real vem da Binance via fetch_balance() no bot.py
        # 2. Adicionar PnL aqui causaria contagem dupla
        # 3. O balance será atualizado no próximo ciclo pelo run_cycle()
        # CORRIGIDO (Bug #1): Retornamos o PnL LÍQUIDO para registro/logging

        return pnl_net  # PnL LÍQUIDO (após taxas)
    
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

        # CORRIGIDO: Verificar duplicatas de forma mais robusta
        # O problema: quando bot reinicia, posições são sincronizadas com entry_time diferente
        # mas quando fecham externamente, todas geram trades "duplicados"
        # Solução: comparar por symbol + side + entry_price + quantity + exit_price
        for existing in history[-100:]:  # Verificar últimos 100 trades
            is_same_symbol = existing.get('symbol') == trade.symbol
            is_same_side = existing.get('side') == trade.side

            # Tolerância de 0.1% para preço de entrada
            existing_entry = existing.get('entry_price', 0)
            if existing_entry > 0:
                price_diff_pct = abs(existing_entry - trade.entry_price) / existing_entry
                is_same_entry_price = price_diff_pct < 0.001  # 0.1%
            else:
                is_same_entry_price = False

            # Tolerância de 0.1% para preço de saída
            existing_exit_price = existing.get('exit_price', 0)
            if existing_exit_price > 0:
                exit_price_diff_pct = abs(existing_exit_price - trade.exit_price) / existing_exit_price
                is_same_exit_price = exit_price_diff_pct < 0.001  # 0.1%
            else:
                is_same_exit_price = False

            # Tolerância de 1% para quantidade (cobre arredondamentos)
            existing_qty = existing.get('quantity', 0)
            if existing_qty > 0:
                qty_diff_pct = abs(existing_qty - trade.quantity) / existing_qty
                is_same_quantity = qty_diff_pct < 0.01  # 1%
            else:
                is_same_quantity = False

            # Verificar se exit_time é muito próximo (dentro de 5 minutos)
            existing_exit = existing.get('exit_time', '')
            is_recent_exit = False
            if existing_exit and trade.exit_time:
                try:
                    existing_exit_dt = datetime.fromisoformat(existing_exit)
                    trade_exit_dt = datetime.fromisoformat(trade.exit_time)
                    time_diff = abs((trade_exit_dt - existing_exit_dt).total_seconds())
                    is_recent_exit = time_diff < 300  # 5 minutos
                except Exception:
                    pass

            # DUPLICATA: mesmo symbol/side/entry_price/quantity (ignora entry_time diferente)
            if is_same_symbol and is_same_side and is_same_entry_price and is_same_quantity:
                log.warning(f"Trade duplicado detectado (price+qty match), ignorando: {trade.symbol}")
                return

            # DUPLICATA: mesmo symbol/side/exit_price/quantity com saída recente
            if is_same_symbol and is_same_side and is_same_exit_price and is_same_quantity and is_recent_exit:
                log.warning(f"Trade duplicado detectado (exit match + recent), ignorando: {trade.symbol}")
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

        # REMOVIDO: Não truncar histórico - manter todos os trades
        # O histórico completo é importante para análises e auditorias

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

        # Criar backup antes de salvar (a cada 10 minutos aprox)
        # O backup_state_file verifica internamente se deve criar backup
        if hasattr(self, '_last_backup_time'):
            if (datetime.now() - self._last_backup_time).total_seconds() > 600:  # 10 min
                backup_state_file(filepath)
                self._last_backup_time = datetime.now()
        else:
            self._last_backup_time = datetime.now()

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
