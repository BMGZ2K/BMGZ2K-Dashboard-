"""
Bot Principal - Sistema de Trading Automatizado
Versao otimizada e simplificada

VERSAO: 3.1 - Mutex para evitar multiplas instancias
"""
import sys
import os
import time
import json
import logging
import atexit
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# MUTEX - Evitar multiplas instancias do bot
# =============================================================================
LOCK_FILE = 'state/bot.lock'


def acquire_lock() -> bool:
    """
    Tentar adquirir lock exclusivo para evitar multiplas instancias.
    Returns True se conseguiu o lock, False se ja existe outra instancia.
    """
    os.makedirs('state', exist_ok=True)

    if os.path.exists(LOCK_FILE):
        # Verificar se o processo ainda esta rodando
        try:
            with open(LOCK_FILE, 'r') as f:
                old_pid = int(f.read().strip())

            # No Windows, tentar verificar se processo existe
            import subprocess
            result = subprocess.run(
                ['tasklist', '/FI', f'PID eq {old_pid}'],
                capture_output=True, text=True
            )
            if str(old_pid) in result.stdout:
                return False  # Processo ainda rodando
        except Exception as e:
            # Se nao conseguir verificar, assume que processo antigo morreu
            pass  # Log não disponível ainda neste ponto

    # Criar lock file com PID atual
    try:
        with open(LOCK_FILE, 'w') as f:
            f.write(str(os.getpid()))
        return True
    except Exception as e:
        print(f"Erro ao criar lock file: {e}")
        return False


def release_lock():
    """Liberar lock ao sair."""
    try:
        if os.path.exists(LOCK_FILE):
            with open(LOCK_FILE, 'r') as f:
                stored_pid = int(f.read().strip())
            if stored_pid == os.getpid():
                os.remove(LOCK_FILE)
    except Exception as e:
        # Silencioso no shutdown - log pode não estar mais disponível
        pass


# Registrar cleanup ao sair
atexit.register(release_lock)

import asyncio
import pandas as pd
import ccxt

from core.config import (
    Config, API_KEY, SECRET_KEY, USE_TESTNET,
    get_validated_params, get_polling_interval,
    get_retry_config, get_circuit_breaker_config, get_auth_recovery_config
)
from core.trader import Trader
from core.signals import SignalGenerator
from core.utils import save_json_atomic, load_json_safe, setup_rotating_logger
from core.binance_fees import get_binance_fees
from core.error_handling import (
    retry_with_backoff, handle_exchange_errors,
    get_circuit_breaker, RetryConfig, OrderValidator,
    CriticalError, AuthRecoveryHandler, get_health_status
)
from core.mtf_engine import MultiTimeframeEngine, MTFSignal

# Diretório de logs
os.makedirs('logs', exist_ok=True)

# Logging com rotação automática (5MB por arquivo, 5 backups)
log = setup_rotating_logger(
    name='trading_bot',
    log_file='logs/bot.log',
    max_bytes=5 * 1024 * 1024,  # 5MB
    backup_count=5,
    level=logging.INFO
)

# Também adicionar handler para stdout
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
log.addHandler(console_handler)


class TradingBot:
    """Bot de trading principal."""
    
    def __init__(self):
        log.info("Inicializando bot...")

        # Inicializar circuit breakers (usando config)
        cb_exchange_cfg = get_circuit_breaker_config('exchange')
        cb_orders_cfg = get_circuit_breaker_config('orders')
        cb_data_cfg = get_circuit_breaker_config('data_fetch')

        self._exchange_cb = get_circuit_breaker(
            'exchange',
            failure_threshold=cb_exchange_cfg.get('failure_threshold', 5),
            recovery_timeout=cb_exchange_cfg.get('recovery_timeout', 60)
        )
        self._orders_cb = get_circuit_breaker(
            'orders',
            failure_threshold=cb_orders_cfg.get('failure_threshold', 3),
            recovery_timeout=cb_orders_cfg.get('recovery_timeout', 120)
        )
        self._data_cb = get_circuit_breaker(
            'data_fetch',
            failure_threshold=cb_data_cfg.get('failure_threshold', 10),
            recovery_timeout=cb_data_cfg.get('recovery_timeout', 30)
        )

        # Auth recovery handler
        auth_cfg = get_auth_recovery_config()
        self._auth_handler = AuthRecoveryHandler(
            pause_duration=auth_cfg.get('pause_duration', 300),
            max_retries=auth_cfg.get('max_retries', 3)
        )

        # Exchange
        self.exchange = self._create_exchange()

        # Carregar parametros otimizados
        self.params = self._load_params()

        # Trader
        self.trader = Trader(self.params)
        self.trader.load_state('state/trader_state.json')

        # Estado
        self.running = True
        self.last_prices = {}
        
        # Sincronizar posicoes existentes na Binance
        self._sync_existing_positions()

        # Carregar valores dinamicos do Config
        symbols = Config.get('symbols', [])
        timeframe = Config.get('timeframes.primary', '1h')

        log.info(f"Bot inicializado | Symbols: {len(symbols)} | TF: {timeframe}")
        log.info(f"Estrategia: {self.params.get('strategy', 'stoch_extreme')}")
    
    def _sync_existing_positions(self):
        """Sincronizar posicoes existentes na Binance com o estado local."""
        from core.trader import Position
        from core.signals import calculate_atr

        try:
            account = self.exchange.fapiPrivateV2GetAccount()
            positions = account.get('positions', [])
            open_positions = [p for p in positions if float(p.get('positionAmt', 0)) != 0]

            # Usar parâmetros do config (dinâmico)
            sl_atr_mult = self.params.get('sl_atr_mult', self.params.get('sl_mult', 3.0))
            tp_atr_mult = self.params.get('tp_atr_mult', self.params.get('tp_mult', 5.0))
            atr_period = self.params.get('atr_period', 14)
            # Fallback percentuais - calculados a partir dos ATR mults (assumindo ATR ~1% do preço)
            sl_pct_fallback = sl_atr_mult * 0.01  # ~3% se sl_atr_mult=3
            tp_pct_fallback = tp_atr_mult * 0.01  # ~5% se tp_atr_mult=5

            for p in open_positions:
                raw_sym = p.get('symbol', '')
                if raw_sym.endswith('USDT'):
                    sym = raw_sym[:-4] + '/USDT'
                else:
                    sym = raw_sym

                if sym not in self.trader.positions:
                    position_amt = float(p.get('positionAmt', 0))
                    side = 'long' if position_amt > 0 else 'short'
                    entry = float(p.get('entryPrice', 0))
                    leverage = int(p.get('leverage', 1))

                    # CORRIGIDO: Calcular SL/TP usando ATR (consistente com signals.py)
                    # Buscar ATR atual para cálculo dinâmico
                    try:
                        df = self.fetch_ohlcv(sym, limit=50)
                        if df is not None and len(df) >= atr_period:
                            atr = calculate_atr(df['high'], df['low'], df['close'], atr_period).iloc[-1]

                            if side == 'long':
                                stop_loss = entry - (atr * sl_atr_mult)
                                take_profit = entry + (atr * tp_atr_mult)
                            else:
                                stop_loss = entry + (atr * sl_atr_mult)
                                take_profit = entry - (atr * tp_atr_mult)
                        else:
                            # Fallback: usar % baseado nos ATR mults do config
                            if side == 'long':
                                stop_loss = entry * (1 - sl_pct_fallback)
                                take_profit = entry * (1 + tp_pct_fallback)
                            else:
                                stop_loss = entry * (1 + sl_pct_fallback)
                                take_profit = entry * (1 - tp_pct_fallback)
                    except Exception as e:
                        log.warning(f"Erro calculando ATR para {sym}: {e}, usando fallback")
                        if side == 'long':
                            stop_loss = entry * (1 - sl_pct_fallback)
                            take_profit = entry * (1 + tp_pct_fallback)
                        else:
                            stop_loss = entry * (1 + sl_pct_fallback)
                            take_profit = entry * (1 - tp_pct_fallback)

                    # Criar ID único para a posição (symbol + entryPrice + qty)
                    # Isso evita duplicatas ao sincronizar
                    position_id = f"{raw_sym}_{entry}_{abs(position_amt)}"

                    # CORRIGIDO: Tentar usar updateTime da Binance para entry_time mais preciso
                    # Isso melhora cálculo de funding e evita duplicatas
                    update_time_ms = p.get('updateTime', 0)
                    if update_time_ms:
                        try:
                            entry_time = datetime.fromtimestamp(update_time_ms / 1000).isoformat()
                        except Exception:
                            entry_time = datetime.now().isoformat()
                    else:
                        entry_time = datetime.now().isoformat()

                    # Adicionar ao estado local usando a dataclass Position
                    pos = Position(
                        symbol=sym,
                        side=side,
                        entry_price=entry,
                        quantity=abs(position_amt),
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        entry_time=entry_time,
                        max_price=entry,
                        min_price=entry,
                        reason_entry='Synced from Binance',
                        strategy=self.params.get('strategy', 'unknown'),
                        order_id=position_id
                    )
                    # Adicionar leverage como atributo extra
                    pos.leverage = leverage

                    self.trader.positions[sym] = pos
                    log.info(f"Posicao sincronizada: {sym} {side.upper()} @ {entry} (lev: {leverage}x) SL: {stop_loss:.6f} TP: {take_profit:.6f}")
            
            if open_positions:
                log.info(f"Total {len(open_positions)} posicoes sincronizadas da Binance")
                
        except Exception as e:
            log.error(f"Erro sincronizando posicoes: {e}")
    
    def _create_exchange(self):
        """Criar conexao com exchange."""
        # Carregar configurações do exchange do config
        recv_window = Config.get('exchange.recv_window', 60000)
        enable_rate_limit = Config.get('exchange.enable_rate_limit', True)
        adjust_time_diff = Config.get('exchange.adjust_for_time_diff', True)
        default_type = Config.get('exchange.default_type', 'future')

        exchange = ccxt.binance({
            'apiKey': API_KEY,
            'secret': SECRET_KEY,
            'enableRateLimit': enable_rate_limit,
            'options': {
                'defaultType': default_type,
                'adjustForTimeDifference': adjust_time_diff,
                'recvWindow': recv_window,
            }
        })
        
        if USE_TESTNET:
            exchange.set_sandbox_mode(True)
            log.info("Modo TESTNET ativo")
        
        # Sincronizar tempo com servidor
        try:
            exchange.load_time_difference()
            log.info("Tempo sincronizado com Binance")
        except Exception as e:
            log.warning(f"Erro sync tempo: {e}")
        
        return exchange
    
    def _load_params(self) -> dict:
        """Carregar parametros do Config centralizado.

        FONTE UNICA DE VERDADE: config/settings.json via Config
        Todos os parametros vem do sistema Config centralizado.
        """
        # Obter parametros do Config centralizado
        params = Config.get_strategy_params()
        strategy_name = params.get('strategy', params.get('name', 'stoch_extreme'))
        log.info(f"Params carregados do Config centralizado: {strategy_name}")
        return params

    def check_reload_params(self):
        """
        Verificar se precisa recarregar parametros via Config.reload().

        Para acionar hot-reload, crie o arquivo: state/reload_params.flag
        O bot irá recarregar config/settings.json e atualizar os parâmetros.
        """
        flag_path = 'state/reload_params.flag'

        if os.path.exists(flag_path):
            try:
                # Recarregar Config centralizado
                Config.reload()

                # Obter novos parametros
                new_params = Config.get_strategy_params()

                # Se estrategia mudou, recriar signal generator
                old_strategy = self.params.get('strategy')
                new_strategy = new_params.get('strategy')
                if new_strategy != old_strategy:
                    log.info(f"Estrategia alterada: {old_strategy} -> {new_strategy}")

                self.params = new_params
                self.trader.params = new_params
                self.trader.signal_generator = SignalGenerator(new_params)

                log.info(f"Config recarregado: {new_strategy}")

                # Remover flag
                os.remove(flag_path)

            except Exception as e:
                log.error(f"Erro recarregando config: {e}")
    
    def fetch_ohlcv(self, symbol: str, limit: int = 100, timeframe: str = None) -> pd.DataFrame:
        """Buscar dados OHLCV com retry automatico."""
        # Verificar circuit breaker
        if not self._data_cb.can_execute():
            log.warning(f"Data circuit breaker OPEN - {symbol} bloqueado")
            return pd.DataFrame()

        retry_cfg = get_retry_config()
        max_attempts = retry_cfg.get('max_attempts', 3)
        base_delay = retry_cfg.get('base_delay', 0.5)

        if timeframe is None:
            timeframe = Config.get('timeframes.primary', '1h')

        for attempt in range(max_attempts):
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                if not ohlcv:
                    return pd.DataFrame()

                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

                # Sucesso - registrar no circuit breaker
                self._data_cb.record_success()
                return df

            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as e:
                self._data_cb.record_failure()
                if attempt < max_attempts - 1:
                    delay = base_delay * (2 ** attempt)
                    log.warning(f"Retry {attempt+1}/{max_attempts} fetch {symbol}: {type(e).__name__} - aguardando {delay:.1f}s")
                    time.sleep(delay)
                else:
                    log.error(f"Max retries fetch {symbol}: {e}")
                    return pd.DataFrame()

            except ccxt.AuthenticationError as e:
                log.critical(f"Auth error fetch {symbol}: {e}")
                self._handle_auth_error(e)
                return pd.DataFrame()

            except Exception as e:
                log.error(f"Erro fetch {symbol}: {e}")
                return pd.DataFrame()

        return pd.DataFrame()

    def fetch_ohlcv_mtf(self, symbol: str, timeframes: list = None, limit: int = 100) -> dict:
        """Buscar dados OHLCV para multiplos timeframes."""
        if timeframes is None:
            timeframes = Config.get('timeframes.active_timeframes', ['5m', '15m', '1h'])

        result = {}
        for tf in timeframes:
            df = self.fetch_ohlcv(symbol, limit=limit, timeframe=tf)
            if not df.empty:
                result[tf] = df

        return result
    
    def fetch_balance(self) -> dict:
        """Buscar saldo com retry automatico."""
        # Verificar circuit breaker
        if not self._exchange_cb.can_execute():
            log.warning("Exchange circuit breaker OPEN - balance bloqueado")
            return {'total': 0, 'free': 0}

        retry_cfg = get_retry_config()
        max_attempts = retry_cfg.get('max_attempts', 3)
        base_delay = retry_cfg.get('base_delay', 0.5)

        for attempt in range(max_attempts):
            try:
                balance = self.exchange.fetch_balance()
                self._exchange_cb.record_success()
                return {
                    'total': float(balance['total'].get('USDT', 0)),
                    'free': float(balance['free'].get('USDT', 0))
                }

            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as e:
                self._exchange_cb.record_failure()
                if attempt < max_attempts - 1:
                    delay = base_delay * (2 ** attempt)
                    log.warning(f"Retry {attempt+1}/{max_attempts} fetch balance: {type(e).__name__} - aguardando {delay:.1f}s")
                    time.sleep(delay)
                else:
                    log.error(f"Max retries fetch balance: {e}")
                    return {'total': 0, 'free': 0}

            except ccxt.AuthenticationError as e:
                log.critical(f"Auth error fetch balance: {e}")
                self._handle_auth_error(e)
                return {'total': 0, 'free': 0}

            except Exception as e:
                log.error(f"Erro fetch balance: {e}")
                return {'total': 0, 'free': 0}

        return {'total': 0, 'free': 0}

    def _handle_auth_error(self, error: Exception) -> None:
        """Tratar erro de autenticacao - pausar e tentar reconectar."""
        def reconnect():
            """Callback para reconectar ao exchange."""
            self.exchange = self._create_exchange()
            # Testar conexao
            self.exchange.fetch_balance()

        recovered = self._auth_handler.handle_auth_error(error, reconnect)
        if not recovered:
            log.critical("Falha na recuperacao de autenticacao - PARANDO BOT")
            self.running = False
    
    def fetch_positions(self) -> list:
        """Buscar posicoes abertas."""
        try:
            positions = self.exchange.fetch_positions()
            return [p for p in positions if abs(float(p.get('contracts', 0))) > 0]
        except Exception as e:
            log.error(f"Erro fetch positions: {e}")
            return []
    
    def validate_quantity(self, symbol: str, quantity: float) -> float:
        """Validar e ajustar quantidade para limites do mercado."""
        try:
            market = self.exchange.market(symbol)
            limits = market.get('limits', {}).get('amount', {})
            precision = market.get('precision', {}).get('amount', 8)
            
            min_qty = limits.get('min', 0)
            max_qty = limits.get('max', float('inf'))
            
            # Ajustar para limites
            if quantity < min_qty:
                return 0  # Muito pequeno
            if quantity > max_qty:
                quantity = max_qty * 0.95  # 95% do maximo
            
            # Ajustar precisao
            if precision:
                quantity = round(quantity, precision)

            return quantity
        except Exception as e:
            log.debug(f"Erro validando quantidade para {symbol}: {e}")
            return quantity
    
    def set_leverage(self, symbol: str, leverage: int = 5) -> bool:
        """Configurar leverage para um simbolo."""
        try:
            self.exchange.set_leverage(leverage, symbol)
            return True
        except Exception as e:
            if 'No need to change' not in str(e):
                log.warning(f"Erro set leverage {symbol}: {e}")
            return False
    
    def execute_order(self, symbol: str, side: str, quantity: float, reduce_only: bool = False,
                       stop_loss: float = None, take_profit: float = None) -> bool:
        """Executar ordem com validacao, retry e circuit breaker."""
        # Verificar circuit breaker
        if not self._orders_cb.can_execute():
            log.warning(f"Orders circuit breaker OPEN - {symbol} bloqueado")
            return False

        retry_cfg = get_retry_config()
        max_attempts = retry_cfg.get('max_attempts', 5)
        base_delay = retry_cfg.get('base_delay', 0.5)

        # Validar quantidade usando OrderValidator
        validated_qty = OrderValidator.validate_quantity(symbol, quantity, self.exchange)
        if validated_qty <= 0:
            log.warning(f"Quantidade invalida para {symbol} apos validacao")
            return False

        # Validar notional
        price = self.last_prices.get(symbol, 0)
        min_notional = self.params.get('min_notional', 6.0)
        if price > 0 and not OrderValidator.validate_notional(validated_qty, price, min_notional):
            log.warning(f"Notional muito pequeno para {symbol}: {validated_qty * price:.2f} < {min_notional}")
            return False

        for attempt in range(max_attempts):
            try:
                # Configurar leverage
                leverage = self.params.get('leverage', self.params.get('max_leverage', 5))
                self.set_leverage(symbol, leverage)

                params = {'reduceOnly': True} if reduce_only else {}

                order = self.exchange.create_market_order(
                    symbol, side, validated_qty, params=params
                )

                # Sucesso - registrar no circuit breaker
                self._orders_cb.record_success()
                log.info(f"Ordem executada: {side.upper()} {symbol} qty={validated_qty:.4f}")

                # Criar ordens SL/TP na exchange para proteger posicao
                if not reduce_only and stop_loss and take_profit:
                    self._place_sl_tp_orders(symbol, side, validated_qty, stop_loss, take_profit)

                return True

            except ccxt.InvalidOrder as e:
                error_msg = str(e)
                # Tentar corrigir erro de precisao
                if '-1111' in error_msg or 'precision' in error_msg.lower():
                    fixed_qty = self._round_to_precision(validated_qty, symbol, 'quantity')
                    if fixed_qty != validated_qty and fixed_qty > 0:
                        log.info(f"Corrigindo precisao {symbol}: {validated_qty} -> {fixed_qty}")
                        validated_qty = fixed_qty
                        continue  # Tentar novamente com quantidade corrigida
                log.error(f"Ordem invalida {symbol}: {e}")
                return False

            except ccxt.InsufficientFunds as e:
                log.warning(f"Saldo insuficiente {symbol}: {e}")
                return False

            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as e:
                self._orders_cb.record_failure()
                if attempt < max_attempts - 1:
                    delay = base_delay * (2 ** attempt)
                    log.warning(f"Retry {attempt+1}/{max_attempts} ordem {symbol}: {type(e).__name__} - aguardando {delay:.1f}s")
                    time.sleep(delay)
                else:
                    log.error(f"Max retries ordem {symbol}: {e}")
                    return False

            except ccxt.AuthenticationError as e:
                log.critical(f"Auth error ordem {symbol}: {e}")
                self._handle_auth_error(e)
                return False

            except Exception as e:
                self._orders_cb.record_failure()
                log.error(f"Erro ordem {symbol}: {e}")
                return False

        return False

    def _round_to_precision(self, value: float, symbol: str, precision_type: str = 'quantity') -> float:
        """
        Arredondar valor para precisão do símbolo (requisito Binance).

        FASE 4.0: Fix para erro -1111 "Precision is over the maximum defined"

        Args:
            value: Valor a arredondar
            symbol: Par de trading (ex: 'BTC/USDT')
            precision_type: 'quantity' ou 'price'

        Returns:
            Valor arredondado para a precisão correta
        """
        try:
            binance_fees = get_binance_fees(use_testnet=self.params.get('use_testnet', True))
            raw_symbol = symbol.replace('/', '')
            symbol_info = binance_fees.get_symbol_info(raw_symbol)

            if symbol_info:
                if precision_type == 'quantity':
                    precision = symbol_info.quantity_precision
                else:  # price
                    precision = symbol_info.price_precision
                return round(value, precision)
        except Exception as e:
            log.debug(f"Não foi possível obter precisão para {symbol}: {e}")

        # Fallback: precisão genérica (quantity=3, price=2 para maioria)
        fallback_precision = 3 if precision_type == 'quantity' else 2
        return round(value, fallback_precision)

    def _place_sl_tp_orders(self, symbol: str, entry_side: str, quantity: float,
                            stop_loss: float, take_profit: float) -> bool:
        """
        Colocar ordens STOP_MARKET e TAKE_PROFIT_MARKET na exchange.
        Estas protegem a posição mesmo se o bot crashar.

        FASE 4.0: Corrigido para usar precisão correta do símbolo.
        """
        try:
            # Lado oposto para fechar a posição
            close_side = 'SELL' if entry_side.lower() == 'buy' else 'BUY'
            raw_symbol = symbol.replace('/', '')

            # FASE 4.0: Arredondar para precisão correta do símbolo
            qty_rounded = self._round_to_precision(quantity, symbol, 'quantity')
            sl_rounded = self._round_to_precision(stop_loss, symbol, 'price')
            tp_rounded = self._round_to_precision(take_profit, symbol, 'price')

            # Validar que arredondamento não zerou a quantidade
            if qty_rounded <= 0:
                log.warning(f"Quantidade arredondada para 0: {symbol} {quantity} -> {qty_rounded}")
                return False

            # Cancelar ordens existentes para este símbolo (evitar duplicatas)
            try:
                self.exchange.cancel_all_orders(symbol)
            except Exception:
                pass  # OK se não houver ordens

            # Ordem STOP_MARKET (Stop Loss)
            try:
                sl_order = self.exchange.fapiPrivatePostOrder({
                    'symbol': raw_symbol,
                    'side': close_side,
                    'type': 'STOP_MARKET',
                    'quantity': str(qty_rounded),
                    'stopPrice': str(sl_rounded),
                    'reduceOnly': 'true'
                })
                log.info(f"SL order criada: {symbol} @ {sl_rounded} qty={qty_rounded}")
            except Exception as e:
                log.warning(f"Erro criando SL order {symbol}: {e}")

            # Ordem TAKE_PROFIT_MARKET (Take Profit)
            try:
                tp_order = self.exchange.fapiPrivatePostOrder({
                    'symbol': raw_symbol,
                    'side': close_side,
                    'type': 'TAKE_PROFIT_MARKET',
                    'quantity': str(qty_rounded),
                    'stopPrice': str(tp_rounded),
                    'reduceOnly': 'true'
                })
                log.info(f"TP order criada: {symbol} @ {tp_rounded} qty={qty_rounded}")
            except Exception as e:
                log.warning(f"Erro criando TP order {symbol}: {e}")

            return True
        except Exception as e:
            log.error(f"Erro colocando SL/TP orders {symbol}: {e}")
            return False

    def _cancel_sl_tp_orders(self, symbol: str) -> bool:
        """Cancelar ordens SL/TP quando posição é fechada manualmente."""
        try:
            self.exchange.cancel_all_orders(symbol)
            log.debug(f"Ordens SL/TP canceladas para {symbol}")
            return True
        except Exception as e:
            log.debug(f"Erro cancelando ordens {symbol}: {e}")
            return False

    def _execute_position_rotation(self, new_symbol: str, new_action: dict) -> bool:
        """
        Executar rotação inteligente de posições.

        ROTAÇÃO INTELIGENTE:
        1. Identifica a posição mais fraca usando weakness score
        2. Fecha a posição fraca
        3. Abre nova posição com sinal forte

        Args:
            new_symbol: Símbolo do novo sinal
            new_action: Ação do novo sinal (open)

        Returns:
            True se rotação executada com sucesso, False caso contrário
        """
        try:
            new_strength = new_action.get('strength', 0)

            # Encontrar posição mais fraca (excluindo o novo símbolo)
            weakest_symbol = self.trader.get_weakest_position(
                self.last_prices,
                exclude_symbols=[new_symbol]
            )

            if not weakest_symbol:
                log.debug(f"Rotação: nenhuma posição fraca suficiente para {new_symbol}")
                return False

            # Obter dados da posição fraca
            weak_pos = self.trader.positions.get(weakest_symbol)
            if not weak_pos:
                return False

            # Calcular weakness score para log
            weak_price = self.last_prices.get(weakest_symbol, weak_pos.entry_price)
            weakness_score = self.trader.calculate_position_weakness(weakest_symbol, weak_price)

            # Calcular PnL% da posição fraca
            if weak_pos.side == 'long':
                weak_pnl_pct = ((weak_price - weak_pos.entry_price) / weak_pos.entry_price) * 100
            else:
                weak_pnl_pct = ((weak_pos.entry_price - weak_price) / weak_pos.entry_price) * 100

            log.info(f"[ROTATION] Substituindo {weakest_symbol} (weakness={weakness_score:.1f}, PnL={weak_pnl_pct:.2f}%) por {new_symbol} (strength={new_strength:.1f})")

            # PASSO 1: Fechar posição fraca
            close_side = 'sell' if weak_pos.side == 'long' else 'buy'
            close_success = self.execute_order(
                weakest_symbol,
                close_side,
                weak_pos.quantity,
                reduce_only=True
            )

            if not close_success:
                log.warning(f"[ROTATION] Falha ao fechar {weakest_symbol}")
                return False

            # Cancelar ordens SL/TP da posição fechada
            self._cancel_sl_tp_orders(weakest_symbol)

            # Registrar saída da posição fraca
            exit_price = self._get_current_price(weakest_symbol)
            if exit_price <= 0:
                exit_price = weak_price
            self.trader.record_exit(weakest_symbol, exit_price, f'ROTATION_OUT (-> {new_symbol})')

            # PASSO 2: Abrir nova posição
            open_success = self.execute_order(
                new_symbol,
                new_action['side'],
                new_action['quantity'],
                reduce_only=False,
                stop_loss=new_action['stop_loss'],
                take_profit=new_action['take_profit']
            )

            if open_success:
                self.trader.record_entry(
                    symbol=new_symbol,
                    side=new_action['side'],
                    price=new_action['price'],
                    quantity=new_action['quantity'],
                    sl=new_action['stop_loss'],
                    tp=new_action['take_profit'],
                    reason=f"ROTATION_IN (<- {weakest_symbol}) | {new_action['reason']}",
                    strategy=self.params.get('strategy', 'unknown')
                )
                log.info(f"[ROTATION] Concluída: {weakest_symbol} -> {new_symbol}")
                return True
            else:
                log.warning(f"[ROTATION] Falha ao abrir {new_symbol} após fechar {weakest_symbol}")
                return False

        except Exception as e:
            log.error(f"[ROTATION] Erro: {e}")
            return False

    def process_symbol(self, symbol: str, timeframe: str = None) -> dict:
        """Processar um simbolo em um timeframe especifico."""
        result = {'symbol': symbol, 'action': None, 'timeframe': timeframe}

        try:
            df = self.fetch_ohlcv(symbol, limit=100, timeframe=timeframe)
            if df.empty:
                return result

            action = self.trader.process_candle(symbol, df, timeframe=timeframe)

            if action:
                result['action'] = action

                if action['action'] == 'open':
                    # Destacar visualmente: + para BUY, - para SELL (ASCII seguro)
                    side_icon = "[+]" if action['side'].lower() == 'buy' else "[-]"
                    tf_label = f"[{timeframe or 'primary'}]" if timeframe else ""
                    log.info(f"{side_icon} SINAL {symbol} {tf_label}: {action['side'].upper()} | Reason: {action['reason']} | Strength: {action['strength']:.1f}")
                elif action['action'] == 'close':
                    log.info(f"[X] EXIT {symbol}: {action['reason']}")

            # Guardar ultimo preco
            self.last_prices[symbol] = df.iloc[-1]['close']

        except Exception as e:
            log.error(f"Erro processando {symbol}: {e}")

        return result

    def process_symbol_mtf(self, symbol: str) -> list:
        """
        Processar simbolo em multiplos timeframes.
        Retorna lista de acoes (uma por timeframe com sinal).
        """
        results = []
        active_tfs = Config.get('timeframes.active_timeframes', ['5m', '15m', '1h'])
        mtf_validation = Config.get('timeframes.mtf_validation', {})

        # Buscar dados de todos os timeframes
        tf_data = self.fetch_ohlcv_mtf(symbol, active_tfs)
        if not tf_data:
            return results

        # Calcular direcao do HTF (4h) como filtro
        htf = Config.get('timeframes.htf', '4h')
        htf_df = self.fetch_ohlcv(symbol, limit=100, timeframe=htf)
        htf_direction = 'neutral'

        if not htf_df.empty and len(htf_df) >= 21:
            ema_fast = htf_df['close'].ewm(span=9, adjust=False).mean().iloc[-1]
            ema_slow = htf_df['close'].ewm(span=21, adjust=False).mean().iloc[-1]
            current_price = htf_df['close'].iloc[-1]

            if ema_fast > ema_slow and current_price > ema_fast:
                htf_direction = 'bullish'
            elif ema_fast < ema_slow and current_price < ema_fast:
                htf_direction = 'bearish'

        # Processar cada timeframe
        tf_signals = {}
        for tf in active_tfs:
            if tf not in tf_data:
                continue

            df = tf_data[tf]
            action = self.trader.process_candle(symbol, df, timeframe=tf)

            if action and action['action'] == 'open':
                # Guardar sinal
                tf_signals[tf] = {
                    'direction': 'long' if action['side'].lower() == 'buy' else 'short',
                    'strength': action.get('strength', 0),
                    'action': action
                }

        # Validar sinais com MTF
        for tf, signal_info in tf_signals.items():
            required_tfs = mtf_validation.get(f"{tf}_requires", [])
            direction = signal_info['direction']

            # Filtro HTF (desativado - muito restritivo)
            # if htf_direction != 'neutral':
            #     expected = 'long' if htf_direction == 'bullish' else 'short'
            #     if direction != expected:
            #         log.debug(f"MTF filtro: {symbol} {tf} {direction} rejeitado (HTF: {htf_direction})")
            #         continue

            # Validar com timeframes superiores
            aligned_count = 0
            mtf_bonus = 0.0

            for req_tf in required_tfs:
                if req_tf in tf_signals:
                    req_dir = tf_signals[req_tf]['direction']
                    if req_dir == direction:
                        aligned_count += 1
                        mtf_bonus += 1.5
                    elif req_dir != 'neutral':
                        mtf_bonus -= 1.0

            # Para scalping (5m) - relaxado para permitir mais sinais
            # if tf == '5m' and aligned_count < 1:
            #     log.debug(f"MTF filtro: {symbol} 5m {direction} rejeitado (sem alinhamento)")
            #     continue

            # Rejeitar se penalidade excessiva (relaxado para permitir mais sinais)
            if mtf_bonus < -2.0:  # Era -0.5, agora muito mais permissivo
                continue

            # Sinal validado - adicionar bonus
            action = signal_info['action'].copy()
            action['strength'] = action.get('strength', 0) + max(0, mtf_bonus)
            action['mtf_bonus'] = mtf_bonus
            action['aligned_count'] = aligned_count
            action['entry_timeframe'] = tf

            side_icon = "[+]" if action['side'].lower() == 'buy' else "[-]"
            log.info(f"{side_icon} MTF SINAL {symbol} [{tf}]: {action['side'].upper()} | Strength: {action['strength']:.1f} | Aligned: {aligned_count} TFs")

            results.append({'symbol': symbol, 'action': action, 'timeframe': tf})

        # Guardar ultimo preco
        for tf, df in tf_data.items():
            if not df.empty:
                self.last_prices[symbol] = df.iloc[-1]['close']
                break

        return results
    
    def sync_positions(self):
        """Sincronizar posicoes com exchange."""
        try:
            # Usar fapiPrivateV2GetAccount para lista correta de posições
            account = self.exchange.fapiPrivateV2GetAccount()
            positions = account.get('positions', [])
            open_positions = [p for p in positions if float(p.get('positionAmt', 0)) != 0]

            # Normalizar símbolos (BTCUSDT -> BTC/USDT)
            exchange_symbols = set()
            for p in open_positions:
                raw_sym = p.get('symbol', '')
                if raw_sym.endswith('USDT'):
                    sym = raw_sym[:-4] + '/USDT'
                else:
                    sym = raw_sym
                exchange_symbols.add(sym)

            # Verificar posicoes fechadas externamente
            for symbol in list(self.trader.positions.keys()):
                if symbol not in exchange_symbols:
                    log.info(f"Posicao {symbol} fechada externamente")
                    # Buscar preço atual real da exchange (não usar last_prices que pode estar desatualizado)
                    exit_price = self._get_current_price(symbol)
                    if exit_price > 0:
                        self.trader.record_exit(symbol, exit_price, 'EXTERNAL')
                    elif symbol in self.last_prices:
                        # Fallback para last_prices se não conseguir preço atual
                        self.trader.record_exit(symbol, self.last_prices[symbol], 'EXTERNAL')

        except Exception as e:
            log.error(f"Erro sync: {e}")

    def _get_current_price(self, symbol: str) -> float:
        """Buscar preço atual da exchange."""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return float(ticker.get('last', 0) or ticker.get('close', 0))
        except Exception as e:
            log.warning(f"Erro buscando preço atual de {symbol}: {e}")
            return 0.0

    def _save_mtf_signals(self, actions: list, use_mtf: bool = True):
        """
        Salvar sinais MTF em arquivo para o dashboard.
        Mantém sinais por até 5 minutos mesmo sem novos sinais.
        """
        try:
            # Carregar dados existentes ou criar novo
            existing = load_json_safe('state/mtf_signals.json') or {
                'signals': [],
                'stats': {'5m': 0, '15m': 0, '1h': 0, '4h': 0},
                'last_update': None
            }

            now = datetime.now()
            now_str = now.isoformat()

            # Se não há novos sinais, manter os existentes se ainda forem válidos (5 min)
            if not actions:
                old_signals = existing.get('signals', [])
                valid_signals = []

                for sig in old_signals:
                    try:
                        sig_time = datetime.fromisoformat(sig.get('timestamp', ''))
                        age_seconds = (now - sig_time).total_seconds()
                        if age_seconds < 300:  # 5 minutos
                            valid_signals.append(sig)
                    except:
                        pass

                # Se ainda há sinais válidos, manter stats atualizadas
                if valid_signals:
                    stats = {'5m': 0, '15m': 0, '1h': 0, '4h': 0}
                    for sig in valid_signals:
                        tf = sig.get('timeframe', '15m')
                        if tf in stats:
                            stats[tf] += 1

                    data = {
                        'signals': valid_signals,
                        'stats': stats,
                        'total_this_cycle': 0,
                        'last_update': now_str,
                        'mode': 'MTF' if use_mtf else 'SINGLE_TF',
                        'active_timeframes': Config.get('timeframes.active_timeframes', ['5m', '15m', '1h'])
                    }
                    save_json_atomic('state/mtf_signals.json', data)
                    return

            # Criar novos sinais
            new_signals = []
            stats = {'5m': 0, '15m': 0, '1h': 0, '4h': 0}

            for result in actions[:15]:  # Top 15 sinais
                action = result.get('action', {})
                tf = result.get('timeframe', '15m')

                signal = {
                    'symbol': result.get('symbol', ''),
                    'timeframe': tf,
                    'side': action.get('side', 'unknown'),
                    'strength': round(action.get('strength', 0), 1),
                    'mtf_bonus': round(action.get('mtf_bonus', 0), 1),
                    'aligned_count': action.get('aligned_count', 0),
                    'price': round(action.get('price', 0), 6),
                    'stop_loss': round(action.get('stop_loss', 0), 6),
                    'take_profit': round(action.get('take_profit', 0), 6),
                    'reason': action.get('reason', ''),
                    'timestamp': now_str
                }
                new_signals.append(signal)

                # Contar por timeframe
                if tf in stats:
                    stats[tf] += 1

            # Atualizar dados
            data = {
                'signals': new_signals,
                'stats': stats,
                'total_this_cycle': len(actions),
                'last_update': now_str,
                'mode': 'MTF' if use_mtf else 'SINGLE_TF',
                'active_timeframes': Config.get('timeframes.active_timeframes', ['5m', '15m', '1h'])
            }

            # Salvar atomicamente
            save_json_atomic('state/mtf_signals.json', data)

        except Exception as e:
            log.debug(f"Erro salvando sinais MTF: {e}")

    def run_cycle(self):
        """Executar um ciclo de trading."""
        # Verificar se precisa recarregar parametros (após auto_evolve)
        self.check_reload_params()

        # Atualizar balance
        balance = self.fetch_balance()
        balance_total = balance['total']
        
        log.debug(f"Balance: ${balance_total:.2f}")
        
        # Se balance for 0, pode ser erro de conexao - nao atualizar
        if balance_total > 0:
            self.trader.update_balance(balance_total)
        else:
            log.warning(f"Balance zerado - mantendo estado anterior")
        
        if self.trader.is_halted:
            log.warning("CIRCUIT BREAKER ATIVO - Operacoes pausadas")
            return
        
        # Sincronizar posicoes
        self.sync_positions()
        
        # Processar simbolos em paralelo com MTF (lista dinamica do Config)
        actions_to_execute = []
        symbols = Config.get('symbols', [])
        use_mtf = len(Config.get('timeframes.active_timeframes', [])) > 1

        # Aumentar workers para processar mais simbolos em paralelo (150+ simbolos)
        max_workers = min(len(symbols), 20)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            if use_mtf:
                # Modo MTF: processa cada simbolo em multiplos timeframes
                futures = {executor.submit(self.process_symbol_mtf, s): s for s in symbols}

                for future in as_completed(futures):
                    try:
                        results = future.result()
                        # process_symbol_mtf retorna lista de resultados
                        for result in results:
                            if result.get('action'):
                                actions_to_execute.append(result)
                    except Exception as e:
                        log.error(f"Erro processando MTF: {e}")
            else:
                # Modo classico: um timeframe por simbolo
                futures = {executor.submit(self.process_symbol, s): s for s in symbols}

                for future in as_completed(futures):
                    result = future.result()
                    if result.get('action'):
                        actions_to_execute.append(result)

        # Ordenar por strength para priorizar sinais mais fortes
        actions_to_execute.sort(key=lambda x: x['action'].get('strength', 0), reverse=True)

        # Salvar sinais MTF em arquivo para o dashboard
        self._save_mtf_signals(actions_to_execute, use_mtf)

        # Executar acoes
        for result in actions_to_execute:
            action = result['action']
            symbol = result['symbol']

            if action['action'] == 'open':
                # Verificar se pode abrir
                max_pos = self.params.get('max_positions', 10)
                signal_strength = action.get('strength', 0)
                enable_rotation = Config.get('trader.enable_position_rotation', True)
                enable_proactive = Config.get('trader.enable_proactive_rotation', True)

                # ROTAÇÃO PROATIVA: Mesmo sem atingir limite, substituir posições fracas por sinais muito fortes
                if enable_rotation and enable_proactive and len(self.trader.positions) >= 3:
                    if self.trader.should_rotate_position(signal_strength, proactive=True):
                        rotation_result = self._execute_position_rotation(symbol, action)
                        if rotation_result:
                            continue  # Rotação executada, próximo sinal

                # ROTAÇÃO NO LIMITE: Se no limite e sinal forte, tentar rotação
                if len(self.trader.positions) >= max_pos:
                    if enable_rotation and self.trader.should_rotate_position(signal_strength):
                        # Tentar rotação inteligente
                        rotation_result = self._execute_position_rotation(symbol, action)
                        if rotation_result:
                            continue  # Rotação executada, próximo sinal

                    # Sem rotação - apenas log debug
                    log.debug(f"Posições no limite ({len(self.trader.positions)}/{max_pos}) - aguardando")
                    continue

                # Verificar margem disponivel (usa max_margin_usage do WFO)
                max_margin = self.params.get('max_margin_usage', 0.80)
                try:
                    account = self.exchange.fapiPrivateV2GetAccount()
                    margin_balance = float(account.get('totalMarginBalance', 1))
                    margin_used = float(account.get('totalInitialMargin', 0))
                    margin_used_pct = margin_used / margin_balance if margin_balance > 0 else 0

                    if margin_used_pct > max_margin:
                        # Log informativo, não warning - sistema está funcionando corretamente
                        log.debug(f"Margem em uso: {margin_used_pct*100:.1f}% (max: {max_margin*100:.0f}%) - {symbol} aguardando")
                        continue
                except Exception as e:
                    log.debug(f"Erro verificando margem: {e}")
                
                # CORRIGIDO: Passar SL/TP para criar ordens na exchange
                success = self.execute_order(
                    symbol,
                    action['side'],
                    action['quantity'],
                    reduce_only=False,
                    stop_loss=action['stop_loss'],
                    take_profit=action['take_profit']
                )

                if success:
                    self.trader.record_entry(
                        symbol=symbol,
                        side=action['side'],
                        price=action['price'],
                        quantity=action['quantity'],
                        sl=action['stop_loss'],
                        tp=action['take_profit'],
                        reason=action['reason'],
                        strategy=self.params.get('strategy', 'unknown')
                    )

            elif action['action'] == 'partial_close':
                # FASE 3.1: Saída parcial
                if symbol not in self.trader.positions:
                    log.debug(f"Posição {symbol} já foi fechada, ignorando partial_close")
                    continue

                partial_exit = action.get('partial_exit', {})
                qty_to_close = action['quantity']

                # Executar ordem parcial
                success = self.execute_order(symbol, action['side'], qty_to_close, reduce_only=True)

                if success:
                    # Buscar preço real de execução
                    exit_price = self._get_current_price(symbol)
                    if exit_price <= 0 and symbol in self.last_prices:
                        exit_price = self.last_prices[symbol]

                    if exit_price > 0:
                        # Executar partial exit (atualiza posição)
                        pnl = self.trader.execute_partial_exit(symbol, partial_exit, exit_price)
                        log.info(f"PARTIAL EXIT {symbol}: {partial_exit.get('close_pct', 0)*100:.0f}% @ {exit_price:.4f} | PnL: ${pnl:.2f}")

                        # Atualizar ordens SL/TP na exchange com nova quantidade
                        if symbol in self.trader.positions:
                            pos = self.trader.positions[symbol]
                            if pos.quantity > 0:
                                self._cancel_sl_tp_orders(symbol)
                                entry_side = 'buy' if pos.side == 'long' else 'sell'
                                self._place_sl_tp_orders(symbol, entry_side, pos.quantity, pos.stop_loss, pos.take_profit)

            elif action['action'] == 'close':
                # CORRIGIDO: Verificar se posição ainda existe antes de fechar
                if symbol not in self.trader.positions:
                    log.debug(f"Posição {symbol} já foi fechada, ignorando ação de close")
                    continue

                success = self.execute_order(symbol, action['side'], action['quantity'], reduce_only=True)

                if success:
                    # Cancelar ordens SL/TP da exchange quando fechamos manualmente
                    self._cancel_sl_tp_orders(symbol)
                    # Buscar preço real de execução
                    exit_price = self._get_current_price(symbol)
                    if exit_price <= 0 and symbol in self.last_prices:
                        exit_price = self.last_prices[symbol]
                    if exit_price > 0:
                        self.trader.record_exit(symbol, exit_price, action['reason'])
        
        # Salvar estado
        self.trader.save_state('state/trader_state.json')

        # Log stats (reduzido para a cada 5 min ou quando há mudança significativa)
        stats = self.trader.get_stats()
        now = time.time()
        last_log_time = getattr(self, '_last_balance_log_time', 0)
        last_log_balance = getattr(self, '_last_balance_log_value', 0)

        current_balance = stats['balance']
        balance_change_pct = abs(current_balance - last_log_balance) / max(last_log_balance, 1) * 100

        # Log se: passou 5 min OU mudança > 0.5% OU primeira vez
        should_log = (now - last_log_time > 300) or (balance_change_pct > 0.5) or (last_log_time == 0)

        if should_log:
            log.info(f"Balance: ${current_balance:.2f} | Positions: {len(self.trader.positions)} | Trades: {stats['total_trades']} | WR: {stats['win_rate']:.1f}%")
            self._last_balance_log_time = now
            self._last_balance_log_value = current_balance
    
    def run(self):
        """Loop principal."""
        log.info("Iniciando loop de trading...")

        # Verificar se MTF esta ativo
        active_tfs = Config.get('timeframes.active_timeframes', [])
        use_mtf = len(active_tfs) > 1

        if use_mtf:
            # Para MTF, usar o intervalo do timeframe mais curto (scalping)
            scalp_tf = Config.get('timeframes.scalp', '5m')
            interval = Config.get(f'timeframes.polling_intervals.{scalp_tf}', 10)
            log.info(f"MTF MODE ATIVO | Timeframes: {active_tfs} | Polling: {interval}s")
        else:
            # Intervalo dinâmico baseado no timeframe (do Config centralizado)
            timeframe = Config.get('timeframes.primary', '1h')
            interval = get_polling_interval(timeframe)
            log.info(f"Polling interval: {interval}s para timeframe {timeframe}")

        while self.running:
            try:
                self.run_cycle()
                time.sleep(interval)

            except KeyboardInterrupt:
                log.info("Bot parado pelo usuario")
                self.running = False
            except Exception as e:
                log.error(f"Erro no loop: {e}")
                time.sleep(interval)  # Usar mesmo intervalo no erro

        # Salvar estado final
        self.trader.save_state('state/trader_state.json')
        log.info("Bot finalizado")


def main():
    os.makedirs('logs', exist_ok=True)
    os.makedirs('state', exist_ok=True)

    # Verificar se ja existe outra instancia rodando
    if not acquire_lock():
        print("=" * 60)
        print("  ERRO: Ja existe uma instancia do bot rodando!")
        print("  Se isso estiver incorreto, delete o arquivo:")
        print(f"  {os.path.abspath(LOCK_FILE)}")
        print("=" * 60)
        sys.exit(1)

    # Carregar config para exibir no banner
    strategy_name = Config.get_strategy_params().get('strategy', 'unknown')
    use_kelly = Config.get('trader.use_kelly_sizing', False)
    mode = "TESTNET" if USE_TESTNET else "PRODUCTION"
    symbols_count = len(Config.get('symbols', []))
    timeframe = Config.get('timeframes.primary', '1h')

    print("=" * 60)
    print("  BOT DE TRADING INICIADO")
    print(f"  PID: {os.getpid()} | Mode: {mode}")
    print(f"  Estrategia: {strategy_name}")
    print(f"  Symbols: {symbols_count} | Timeframe: {timeframe}")
    print(f"  Kelly Sizing: {'ON' if use_kelly else 'OFF'}")
    print("=" * 60)

    bot = TradingBot()
    bot.run()


if __name__ == "__main__":
    main()
