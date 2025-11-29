"""
Error Handling Module - Tratamento robusto de erros para trading bot
================================================================================
Fornece decorators, circuit breakers e handlers para operacoes de trading.

COMPONENTES:
- RetryConfig: Configuracao de retry
- @retry_with_backoff: Decorator para retry com exponential backoff
- @handle_exchange_errors: Decorator para tratamento de erros de exchange
- CircuitBreaker: Padrao circuit breaker para prevenir falhas em cascata
- OrderValidator: Validacao de ordens antes da execucao

VERSAO: 1.0
================================================================================
"""
import ccxt
import functools
import time
import random
import logging
from typing import Callable, Type, Optional, Dict, Any, Tuple, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from threading import Lock
from enum import Enum

log = logging.getLogger(__name__)


# =============================================================================
# EXCECOES CUSTOMIZADAS
# =============================================================================

class TradingBotError(Exception):
    """Excecao base para erros do trading bot."""
    pass


class RetryableError(TradingBotError):
    """Erro que pode ser retentado (rede, timeout, rate limit)."""
    pass


class NonRetryableError(TradingBotError):
    """Erro que NAO deve ser retentado (ordem invalida, saldo insuficiente)."""
    pass


class CriticalError(TradingBotError):
    """Erro critico que requer atencao imediata (falha de autenticacao)."""
    pass


# =============================================================================
# CLASSIFICACAO DE EXCECOES CCXT
# =============================================================================

RETRYABLE_EXCEPTIONS: Tuple[Type[Exception], ...] = (
    ccxt.NetworkError,
    ccxt.ExchangeNotAvailable,
    ccxt.RateLimitExceeded,
    ccxt.RequestTimeout,
    ccxt.DDoSProtection,
)

NON_RETRYABLE_EXCEPTIONS: Tuple[Type[Exception], ...] = (
    ccxt.InvalidOrder,
    ccxt.InsufficientFunds,
    ccxt.OrderNotFound,
    ccxt.InvalidNonce,
)

CRITICAL_EXCEPTIONS: Tuple[Type[Exception], ...] = (
    ccxt.AuthenticationError,
    ccxt.PermissionDenied,
    ccxt.AccountSuspended,
)


# =============================================================================
# CONFIGURACAO DE RETRY
# =============================================================================

@dataclass
class RetryConfig:
    """Configuracao para comportamento de retry."""
    max_attempts: int = 5
    base_delay: float = 0.5
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True

    @classmethod
    def from_dict(cls, config: Dict) -> 'RetryConfig':
        """Criar RetryConfig a partir de dicionario."""
        return cls(
            max_attempts=config.get('max_attempts', 5),
            base_delay=config.get('base_delay', 0.5),
            max_delay=config.get('max_delay', 30.0),
            exponential_base=config.get('exponential_base', 2.0),
            jitter=config.get('jitter', True)
        )

    @classmethod
    def default(cls) -> 'RetryConfig':
        """Retorna configuracao padrao."""
        return cls()


def calculate_delay(attempt: int, config: RetryConfig) -> float:
    """
    Calcular delay com exponential backoff e jitter opcional.

    Args:
        attempt: Numero da tentativa (0-indexed)
        config: Configuracao de retry

    Returns:
        Delay em segundos
    """
    delay = min(
        config.base_delay * (config.exponential_base ** attempt),
        config.max_delay
    )
    if config.jitter:
        # Adiciona jitter de 50-150% do delay calculado
        delay *= (0.5 + random.random())
    return delay


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================

class CircuitBreakerState(Enum):
    """Estados do circuit breaker."""
    CLOSED = "closed"       # Operacao normal
    OPEN = "open"           # Bloqueando chamadas
    HALF_OPEN = "half_open" # Testando recuperacao


@dataclass
class CircuitBreaker:
    """
    Implementacao do padrao Circuit Breaker.

    Transicoes de estado:
    CLOSED -> OPEN (apos failure_threshold falhas)
    OPEN -> HALF_OPEN (apos recovery_timeout)
    HALF_OPEN -> CLOSED (em sucesso) ou OPEN (em falha)

    Exemplo:
        cb = CircuitBreaker(name="orders", failure_threshold=3, recovery_timeout=60)
        if cb.can_execute():
            try:
                result = execute_order()
                cb.record_success()
            except Exception:
                cb.record_failure()
    """
    name: str
    failure_threshold: int = 5
    recovery_timeout: int = 60  # segundos
    half_open_max_calls: int = 3

    # Campos internos (nao inicializados pelo construtor)
    _state: CircuitBreakerState = field(default=CircuitBreakerState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _success_count: int = field(default=0, init=False)
    _last_failure_time: Optional[datetime] = field(default=None, init=False)
    _half_open_calls: int = field(default=0, init=False)
    _lock: Lock = field(default_factory=Lock, init=False)

    def can_execute(self) -> bool:
        """
        Verificar se execucao e permitida.

        Returns:
            True se pode executar, False se circuit breaker esta aberto
        """
        with self._lock:
            if self._state == CircuitBreakerState.CLOSED:
                return True

            elif self._state == CircuitBreakerState.OPEN:
                # Verificar se timeout de recuperacao passou
                if self._last_failure_time:
                    elapsed = (datetime.now() - self._last_failure_time).total_seconds()
                    if elapsed >= self.recovery_timeout:
                        self._state = CircuitBreakerState.HALF_OPEN
                        self._half_open_calls = 0
                        log.info(f"CircuitBreaker [{self.name}]: OPEN -> HALF_OPEN (apos {elapsed:.0f}s)")
                        return True
                return False

            else:  # HALF_OPEN
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

    def record_success(self) -> None:
        """Registrar chamada bem-sucedida."""
        with self._lock:
            self._success_count += 1

            if self._state == CircuitBreakerState.HALF_OPEN:
                self._state = CircuitBreakerState.CLOSED
                self._failure_count = 0
                log.info(f"CircuitBreaker [{self.name}]: HALF_OPEN -> CLOSED (recuperado)")

            elif self._state == CircuitBreakerState.CLOSED:
                # Reset failure count em sucesso
                self._failure_count = 0

    def record_failure(self) -> None:
        """Registrar chamada com falha."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = datetime.now()

            if self._state == CircuitBreakerState.HALF_OPEN:
                self._state = CircuitBreakerState.OPEN
                log.warning(f"CircuitBreaker [{self.name}]: HALF_OPEN -> OPEN (falha durante recuperacao)")

            elif self._state == CircuitBreakerState.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    self._state = CircuitBreakerState.OPEN
                    log.warning(
                        f"CircuitBreaker [{self.name}]: CLOSED -> OPEN "
                        f"(falhas={self._failure_count}/{self.failure_threshold})"
                    )

    def reset(self) -> None:
        """Resetar circuit breaker para estado inicial."""
        with self._lock:
            self._state = CircuitBreakerState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            self._half_open_calls = 0
            log.info(f"CircuitBreaker [{self.name}]: RESET")

    @property
    def state(self) -> str:
        """Retorna estado atual como string."""
        return self._state.value

    @property
    def is_open(self) -> bool:
        """Verificar se circuit breaker esta aberto."""
        return self._state == CircuitBreakerState.OPEN

    @property
    def stats(self) -> Dict[str, Any]:
        """Retorna estatisticas do circuit breaker."""
        with self._lock:
            return {
                'name': self.name,
                'state': self._state.value,
                'failure_count': self._failure_count,
                'success_count': self._success_count,
                'failure_threshold': self.failure_threshold,
                'recovery_timeout': self.recovery_timeout,
                'last_failure': self._last_failure_time.isoformat() if self._last_failure_time else None
            }


# =============================================================================
# SINGLETON CIRCUIT BREAKERS
# =============================================================================

_circuit_breakers: Dict[str, CircuitBreaker] = {}
_cb_lock = Lock()


def get_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: int = 60,
    half_open_max_calls: int = 3
) -> CircuitBreaker:
    """
    Obter ou criar um circuit breaker nomeado (singleton).

    Args:
        name: Nome unico do circuit breaker
        failure_threshold: Numero de falhas para abrir
        recovery_timeout: Segundos ate tentar recuperar
        half_open_max_calls: Chamadas permitidas em half-open

    Returns:
        Instancia do CircuitBreaker
    """
    with _cb_lock:
        if name not in _circuit_breakers:
            _circuit_breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                half_open_max_calls=half_open_max_calls
            )
            log.debug(f"CircuitBreaker [{name}] criado")
        return _circuit_breakers[name]


def get_all_circuit_breakers() -> Dict[str, CircuitBreaker]:
    """Retorna todos os circuit breakers registrados."""
    with _cb_lock:
        return dict(_circuit_breakers)


def reset_all_circuit_breakers() -> None:
    """Reseta todos os circuit breakers."""
    with _cb_lock:
        for cb in _circuit_breakers.values():
            cb.reset()


# =============================================================================
# DECORATOR RETRY WITH BACKOFF
# =============================================================================

def retry_with_backoff(
    config: Optional[RetryConfig] = None,
    max_attempts: Optional[int] = None,
    base_delay: Optional[float] = None,
    retryable_exceptions: Optional[Tuple] = None,
    on_retry: Optional[Callable] = None
) -> Callable:
    """
    Decorator para retry de funcoes com exponential backoff.

    Args:
        config: RetryConfig com parametros de retry (ou usar parametros individuais)
        max_attempts: Numero maximo de tentativas
        base_delay: Delay base em segundos
        retryable_exceptions: Tuple de excecoes para retry
        on_retry: Callback chamado antes de cada retry: on_retry(attempt, exception, delay)

    Exemplo:
        @retry_with_backoff(max_attempts=5, base_delay=0.5)
        def fetch_data():
            return exchange.fetch_ohlcv(symbol)

        @retry_with_backoff(config=RetryConfig(max_attempts=3))
        def place_order():
            return exchange.create_order(...)
    """
    # Construir config se parametros individuais foram passados
    if config is None:
        config = RetryConfig(
            max_attempts=max_attempts or 5,
            base_delay=base_delay or 0.5
        )

    if retryable_exceptions is None:
        retryable_exceptions = RETRYABLE_EXCEPTIONS

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)

                except retryable_exceptions as e:
                    last_exception = e

                    if attempt < config.max_attempts - 1:
                        delay = calculate_delay(attempt, config)
                        log.warning(
                            f"Retry {attempt + 1}/{config.max_attempts}: {func.__name__} | "
                            f"{type(e).__name__} | aguardando {delay:.2f}s"
                        )

                        if on_retry:
                            try:
                                on_retry(attempt, e, delay)
                            except Exception:
                                pass  # Ignorar erros no callback

                        time.sleep(delay)
                    else:
                        log.error(
                            f"Max retries ({config.max_attempts}) atingido para {func.__name__}: "
                            f"{type(e).__name__}: {e}"
                        )

                except NON_RETRYABLE_EXCEPTIONS as e:
                    # Nao faz retry, propaga excecao imediatamente
                    log.error(f"Erro nao-retentavel em {func.__name__}: {type(e).__name__}: {e}")
                    raise

                except CRITICAL_EXCEPTIONS as e:
                    # Erro critico - propaga como CriticalError
                    log.critical(f"Erro CRITICO em {func.__name__}: {type(e).__name__}: {e}")
                    raise CriticalError(f"Erro critico: {e}") from e

            # Todas as tentativas falharam
            if last_exception:
                raise last_exception

        return wrapper
    return decorator


# =============================================================================
# EXCHANGE ERROR HANDLER
# =============================================================================

class ExchangeErrorHandler:
    """Handlers para erros especificos de exchange."""

    @staticmethod
    def handle_network_error(e: ccxt.NetworkError) -> Dict[str, Any]:
        """Tratar erros de conectividade de rede."""
        return {
            'error_type': 'network',
            'retryable': True,
            'message': str(e),
            'recommended_wait': 5.0
        }

    @staticmethod
    def handle_rate_limit(e: ccxt.RateLimitExceeded) -> Dict[str, Any]:
        """Tratar erros de rate limit."""
        # Tentar parsear header Retry-After
        retry_after = 60.0  # Tempo de espera padrao

        error_str = str(e)

        # Binance geralmente inclui tempo no erro
        if 'retry after' in error_str.lower():
            try:
                import re
                match = re.search(r'(\d+)\s*(ms|s|seconds?)', error_str.lower())
                if match:
                    value = int(match.group(1))
                    unit = match.group(2)
                    if 'ms' in unit:
                        retry_after = value / 1000.0
                    else:
                        retry_after = float(value)
            except Exception:
                pass

        return {
            'error_type': 'rate_limit',
            'retryable': True,
            'message': str(e),
            'recommended_wait': retry_after
        }

    @staticmethod
    def handle_invalid_order(e: ccxt.InvalidOrder, symbol: str = '') -> Dict[str, Any]:
        """
        Tratar erros de ordem invalida (precisao, notional, etc).

        Codigos comuns Binance:
        - -1111: Precisao acima do maximo definido
        - -1121: Simbolo invalido
        - -4164: Notional da ordem deve ser no minimo X
        """
        error_msg = str(e)

        result = {
            'error_type': 'invalid_order',
            'retryable': False,
            'message': error_msg,
            'symbol': symbol,
            'fix_applicable': False,
            'suggestion': None
        }

        # Erro de precisao - pode tentar corrigir
        if '-1111' in error_msg or 'precision' in error_msg.lower():
            result['suggestion'] = 'reduce_precision'
            result['fix_applicable'] = True

        # Erro de notional minimo
        elif '-4164' in error_msg or 'notional' in error_msg.lower():
            result['suggestion'] = 'increase_quantity'
            result['fix_applicable'] = True

        # Simbolo invalido
        elif '-1121' in error_msg or 'invalid symbol' in error_msg.lower():
            result['suggestion'] = 'check_symbol'
            result['fix_applicable'] = False

        return result

    @staticmethod
    def handle_insufficient_funds(e: ccxt.InsufficientFunds) -> Dict[str, Any]:
        """Tratar erros de saldo insuficiente."""
        return {
            'error_type': 'insufficient_funds',
            'retryable': False,
            'message': str(e),
            'suggestion': 'reduce_position_size'
        }

    @staticmethod
    def handle_auth_error(e: ccxt.AuthenticationError) -> Dict[str, Any]:
        """Tratar erros de autenticacao - critico."""
        return {
            'error_type': 'authentication',
            'retryable': False,
            'critical': True,
            'message': str(e),
            'action': 'pause_and_retry'  # Pausar 5min e tentar reconectar
        }


def handle_exchange_errors(
    circuit_breaker: Optional[CircuitBreaker] = None,
    on_auth_error: Optional[Callable] = None
) -> Callable:
    """
    Decorator para tratamento de erros de exchange com circuit breaker.

    Args:
        circuit_breaker: CircuitBreaker para protecao (opcional)
        on_auth_error: Callback para erros de autenticacao

    Exemplo:
        cb = get_circuit_breaker('orders')

        @handle_exchange_errors(circuit_breaker=cb)
        def execute_order(symbol, side, quantity):
            return exchange.create_market_order(symbol, side, quantity)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Verificar circuit breaker antes da execucao
            if circuit_breaker and not circuit_breaker.can_execute():
                log.warning(f"CircuitBreaker [{circuit_breaker.name}] ABERTO - {func.__name__} bloqueado")
                return None

            try:
                result = func(*args, **kwargs)

                # Registrar sucesso
                if circuit_breaker:
                    circuit_breaker.record_success()

                return result

            except ccxt.RateLimitExceeded as e:
                info = ExchangeErrorHandler.handle_rate_limit(e)
                log.warning(f"Rate limit atingido: aguardando {info['recommended_wait']}s")
                if circuit_breaker:
                    circuit_breaker.record_failure()
                raise

            except ccxt.InvalidOrder as e:
                # Extrair symbol dos argumentos se possivel
                symbol = ''
                if args:
                    symbol = args[0] if isinstance(args[0], str) else ''
                elif 'symbol' in kwargs:
                    symbol = kwargs['symbol']

                info = ExchangeErrorHandler.handle_invalid_order(e, symbol)
                log.error(f"Ordem invalida ({symbol}): {info['message']}")
                # Nao conta como falha de circuit breaker (erro do usuario, nao do sistema)
                raise

            except ccxt.InsufficientFunds as e:
                info = ExchangeErrorHandler.handle_insufficient_funds(e)
                log.warning(f"Saldo insuficiente: {info['message']}")
                # Nao conta como falha de circuit breaker
                raise

            except ccxt.AuthenticationError as e:
                info = ExchangeErrorHandler.handle_auth_error(e)
                log.critical(f"Falha de autenticacao: {info['message']}")

                # Callback para handler especial de auth error
                if on_auth_error:
                    try:
                        on_auth_error(e)
                    except Exception:
                        pass

                # Pausar e retry (conforme decisao do usuario)
                raise CriticalError(f"Autenticacao falhou - pausando: {e}") from e

            except RETRYABLE_EXCEPTIONS as e:
                if circuit_breaker:
                    circuit_breaker.record_failure()
                raise

            except Exception as e:
                # Erro inesperado
                if circuit_breaker:
                    circuit_breaker.record_failure()
                log.error(f"Erro inesperado em {func.__name__}: {type(e).__name__}: {e}")
                raise

        return wrapper
    return decorator


# =============================================================================
# ORDER VALIDATOR
# =============================================================================

class OrderValidator:
    """Validacao de ordens antes da execucao."""

    @staticmethod
    def validate_quantity(
        symbol: str,
        quantity: float,
        exchange: Any,
        fallback_precision: int = 3
    ) -> float:
        """
        Validar e corrigir precisao de quantidade.

        Usa market info do exchange para garantir precisao correta.

        Args:
            symbol: Simbolo do par (ex: 'BTC/USDT')
            quantity: Quantidade original
            exchange: Instancia do exchange (CCXT)
            fallback_precision: Precisao fallback se nao conseguir obter do exchange

        Returns:
            Quantidade validada e arredondada
        """
        try:
            market = exchange.market(symbol)
            precision = market.get('precision', {}).get('amount', fallback_precision)

            # Arredondar para precisao do exchange
            if isinstance(precision, int):
                validated_qty = round(quantity, precision)
            else:
                # Alguns exchanges usam step size ao inves de decimais
                validated_qty = round(quantity, 8)

            # Verificar limites min/max
            limits = market.get('limits', {}).get('amount', {})
            min_qty = limits.get('min', 0)
            max_qty = limits.get('max', float('inf'))

            if validated_qty < min_qty:
                log.warning(f"{symbol}: quantidade {validated_qty} < minimo {min_qty}")
                return 0.0

            if validated_qty > max_qty:
                log.warning(f"{symbol}: quantidade {validated_qty} > maximo {max_qty}, limitando")
                validated_qty = max_qty * 0.95

            # Limite extra para testnet: notional maximo de $10000 por ordem
            # Isso evita erro "Quantity greater than max quantity"
            try:
                ticker = exchange.fetch_ticker(symbol)
                price = ticker.get('last', 0) or ticker.get('close', 0)
                if price > 0:
                    max_notional = 10000  # $10k max por ordem no testnet
                    max_qty_by_notional = max_notional / price
                    if validated_qty > max_qty_by_notional:
                        log.info(f"{symbol}: limitando qty de {validated_qty:.4f} para {max_qty_by_notional:.4f} (max notional $10k)")
                        validated_qty = round(max_qty_by_notional * 0.95, precision if isinstance(precision, int) else 8)
            except Exception:
                pass  # Ignora se falhar ao buscar ticker

            return validated_qty

        except Exception as e:
            log.debug(f"Erro validando quantidade para {symbol}: {e}")
            return round(quantity, fallback_precision)

    @staticmethod
    def validate_price(
        symbol: str,
        price: float,
        exchange: Any,
        fallback_precision: int = 2
    ) -> float:
        """
        Validar e corrigir precisao de preco.

        Args:
            symbol: Simbolo do par
            price: Preco original
            exchange: Instancia do exchange
            fallback_precision: Precisao fallback

        Returns:
            Preco validado e arredondado
        """
        try:
            market = exchange.market(symbol)
            precision = market.get('precision', {}).get('price', fallback_precision)

            if isinstance(precision, int):
                return round(price, precision)
            else:
                return round(price, 8)

        except Exception:
            return round(price, fallback_precision)

    @staticmethod
    def validate_notional(
        quantity: float,
        price: float,
        min_notional: float = 5.0
    ) -> bool:
        """
        Verificar se ordem atinge valor notional minimo.

        Args:
            quantity: Quantidade da ordem
            price: Preco estimado
            min_notional: Valor notional minimo (padrao $5)

        Returns:
            True se notional e suficiente
        """
        notional = abs(quantity * price)
        return notional >= min_notional

    @staticmethod
    def validate_order(
        symbol: str,
        quantity: float,
        price: float,
        exchange: Any,
        min_notional: float = 5.0
    ) -> Tuple[bool, float, str]:
        """
        Validacao completa de ordem.

        Args:
            symbol: Simbolo do par
            quantity: Quantidade
            price: Preco estimado
            exchange: Instancia do exchange
            min_notional: Valor notional minimo

        Returns:
            Tuple (is_valid, validated_quantity, error_message)
        """
        # Validar quantidade
        validated_qty = OrderValidator.validate_quantity(symbol, quantity, exchange)

        if validated_qty <= 0:
            return False, 0.0, f"Quantidade invalida para {symbol}"

        # Validar notional
        if not OrderValidator.validate_notional(validated_qty, price, min_notional):
            notional = validated_qty * price
            return False, 0.0, f"Notional {notional:.2f} < minimo {min_notional}"

        return True, validated_qty, ""


# =============================================================================
# AUTH RECOVERY HANDLER
# =============================================================================

class AuthRecoveryHandler:
    """
    Handler para recuperacao de erros de autenticacao.
    Implementa pausar e retry conforme decisao do usuario.
    """

    def __init__(
        self,
        pause_duration: int = 300,  # 5 minutos
        max_retries: int = 3
    ):
        self.pause_duration = pause_duration
        self.max_retries = max_retries
        self._retry_count = 0
        self._last_error_time: Optional[datetime] = None
        self._lock = Lock()

    def handle_auth_error(
        self,
        error: Exception,
        reconnect_callback: Optional[Callable] = None
    ) -> bool:
        """
        Tratar erro de autenticacao.

        Args:
            error: Excecao de autenticacao
            reconnect_callback: Funcao para tentar reconectar

        Returns:
            True se conseguiu recuperar, False se deve parar
        """
        with self._lock:
            self._retry_count += 1
            self._last_error_time = datetime.now()

            if self._retry_count > self.max_retries:
                log.critical(
                    f"AuthRecovery: Max retries ({self.max_retries}) atingido. "
                    f"PARANDO BOT."
                )
                return False

            log.warning(
                f"AuthRecovery: Tentativa {self._retry_count}/{self.max_retries}. "
                f"Pausando por {self.pause_duration}s..."
            )

            time.sleep(self.pause_duration)

            if reconnect_callback:
                try:
                    reconnect_callback()
                    log.info("AuthRecovery: Reconexao bem-sucedida!")
                    self._retry_count = 0  # Reset counter on success
                    return True
                except Exception as e:
                    log.error(f"AuthRecovery: Falha ao reconectar: {e}")
                    return self.handle_auth_error(e, reconnect_callback)

            return True

    def reset(self) -> None:
        """Resetar contador de tentativas."""
        with self._lock:
            self._retry_count = 0
            self._last_error_time = None

    @property
    def stats(self) -> Dict[str, Any]:
        """Estatisticas do handler."""
        with self._lock:
            return {
                'retry_count': self._retry_count,
                'max_retries': self.max_retries,
                'pause_duration': self.pause_duration,
                'last_error': self._last_error_time.isoformat() if self._last_error_time else None
            }


# =============================================================================
# HEALTH STATUS
# =============================================================================

def get_health_status() -> Dict[str, Any]:
    """
    Obter status de saude do sistema de error handling.

    Returns:
        Dicionario com status de todos os circuit breakers e estatisticas
    """
    breakers = get_all_circuit_breakers()

    all_closed = all(
        cb.state == 'closed'
        for cb in breakers.values()
    )

    return {
        'status': 'healthy' if all_closed else 'degraded',
        'timestamp': datetime.now().isoformat(),
        'circuit_breakers': {
            name: cb.stats
            for name, cb in breakers.items()
        },
        'total_breakers': len(breakers),
        'open_breakers': sum(1 for cb in breakers.values() if cb.is_open)
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Exceptions
    'TradingBotError',
    'RetryableError',
    'NonRetryableError',
    'CriticalError',

    # Exception tuples
    'RETRYABLE_EXCEPTIONS',
    'NON_RETRYABLE_EXCEPTIONS',
    'CRITICAL_EXCEPTIONS',

    # Config
    'RetryConfig',
    'calculate_delay',

    # Circuit Breaker
    'CircuitBreaker',
    'CircuitBreakerState',
    'get_circuit_breaker',
    'get_all_circuit_breakers',
    'reset_all_circuit_breakers',

    # Decorators
    'retry_with_backoff',
    'handle_exchange_errors',

    # Handlers
    'ExchangeErrorHandler',
    'OrderValidator',
    'AuthRecoveryHandler',

    # Health
    'get_health_status',
]
