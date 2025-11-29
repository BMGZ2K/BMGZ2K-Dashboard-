"""
Configuration Module - Sistema Centralizado e Dinamico
================================================================================
FONTE UNICA DE VERDADE - TODAS AS CONFIGURACOES PASSAM POR AQUI
================================================================================

COMO USAR:
    from core.config import Config

    # Obter parametro
    value = Config.get('strategy.rsi_period', default=14)

    # Obter secao inteira
    strategy_params = Config.get_section('strategy')

    # Recarregar configs (hot-reload)
    Config.reload()

NAO FACA:
    - Hardcode valores em outros modulos
    - Criar arquivos de config separados
    - Duplicar parametros

VERSAO: 3.0 - Configuracao 100% dinamica e centralizada
================================================================================
"""
import os
import json
import threading
import logging
from typing import Any, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(override=True)

log = logging.getLogger(__name__)

# =============================================================================
# ARQUIVO DE CONFIGURACAO CENTRALIZADO
# =============================================================================
CONFIG_FILE = 'config/settings.json'
STATE_FILE = 'state/trader_state.json'

# =============================================================================
# CONFIGURACAO PADRAO (usada se config.json nao existir)
# =============================================================================
DEFAULT_CONFIG = {
    # === METADATA ===
    "version": "3.0",
    "last_updated": "",

    # === API ===
    "api": {
        "use_testnet": True,
    },

    # === SYMBOLS ===
    "symbols": [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "DOGE/USDT",
        "XRP/USDT", "ADA/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT",
        "LTC/USDT", "TRX/USDT", "UNI/USDT", "ATOM/USDT", "NEAR/USDT",
        "APT/USDT", "FIL/USDT", "SUI/USDT", "ARB/USDT", "OP/USDT",
        "INJ/USDT", "GRT/USDT", "ALGO/USDT", "XLM/USDT",
        "CRV/USDT", "RUNE/USDT", "ETC/USDT",
        "AAVE/USDT", "EOS/USDT", "QNT/USDT", "FTM/USDT",
        "DYDX/USDT", "APE/USDT", "XTZ/USDT", "WIF/USDT",
        "SAND/USDT", "MANA/USDT", "AXS/USDT", "ENJ/USDT",
        "CHZ/USDT", "GALA/USDT", "GMT/USDT", "THETA/USDT",
        "VET/USDT", "HBAR/USDT", "ZIL/USDT", "SNX/USDT",
        "FLOW/USDT", "MASK/USDT", "ENS/USDT",
    ],

    # === TIMEFRAMES ===
    "timeframes": {
        "primary": "1h",
        "htf": "4h",
        "ltf": "15m",
        "polling_intervals": {
            "1m": 10,
            "5m": 30,
            "15m": 60,
            "30m": 120,
            "1h": 60,
            "4h": 300,
        }
    },

    # === STRATEGY PARAMS (WFO VALIDATED) ===
    "strategy": {
        "name": "stoch_extreme",

        # RSI
        "rsi_period": 14,
        "rsi_oversold": 25,
        "rsi_overbought": 75,
        "rsi_exit_long": 70,
        "rsi_exit_short": 30,

        # Stochastic
        "stoch_k": 14,
        "stoch_d": 3,
        "stoch_oversold": 20,
        "stoch_overbought": 80,

        # EMAs
        "ema_fast": 9,
        "ema_slow": 21,
        "ema_trend": 50,

        # ADX
        "adx_period": 14,
        "adx_min": 20,
        "adx_strong": 25,
        "adx_very_strong": 35,
        "adx_aggressive": 30,

        # ATR / SL / TP
        "atr_period": 14,
        "sl_atr_mult": 3.0,
        "tp_atr_mult": 5.0,

        # Bollinger Bands
        "bb_period": 20,
        "bb_std": 2.0,

        # Signal Strength
        "min_score_to_open": 5.0,
        "min_signal_strength": 5,
        "max_signal_strength": 10,
        "base_strength": 6.0,
        "stoch_base_strength": 6.0,

        # Bias (baseado em performance real)
        "long_bias": 1.1,
        "short_penalty": 0.9,

        # Momentum
        "momentum_adx_min": 22,
        "momentum_min_move": 0.7,
        "momentum_rsi_long_min": 45,
        "momentum_rsi_long_max": 65,
        "momentum_rsi_short_min": 35,
        "momentum_rsi_short_max": 55,

        # Mean Reversion
        "mr_adx_max": 22,
        "mr_rsi_long": 35,
        "mr_rsi_short": 65,
        "mr_bb_touch_pct": 0.02,

        # Filters
        "min_data_points": 50,
        "use_volume_filter": True,
        "volume_mult_threshold": 1.2,
    },

    # === RISK MANAGEMENT ===
    "risk": {
        "risk_per_trade": 0.01,
        "max_positions": 10,
        "max_margin_usage": 0.8,
        "max_position_pct": 0.15,
        "min_position_pct": 0.05,
        "max_leverage": 10,
        "max_drawdown": 0.20,
        "circuit_breaker_drawdown": 0.25,
        "use_compounding": True,
        "min_notional": 6.0,
    },

    # === EXECUTION ===
    "execution": {
        "order_type": "market",
        "retry_attempts": 5,
        "retry_delay_base": 0.5,
        "slippage_tolerance": 0.002,
    },

    # === FEES ===
    "fees": {
        "maker_fee": 0.0002,
        "taker_fee": 0.0004,
        "slippage": 0.0002,
        "funding_rate": 0.0001,
    },

    # === BACKTEST ===
    "backtest": {
        "initial_capital": 10000,
        "max_drawdown_halt": 0.20,
    },

    # === VALIDATION THRESHOLDS ===
    "validation": {
        "min_sharpe": 1.0,
        "min_profit_factor": 1.3,
        "min_win_rate": 0.40,
        "max_drawdown": 0.25,
        "min_trades": 50,
    },

    # === ERROR HANDLING ===
    "error_handling": {
        "retry": {
            "max_attempts": 5,
            "base_delay": 0.5,
            "max_delay": 30.0,
            "exponential_base": 2.0,
            "jitter": True,
        },
        "circuit_breaker": {
            "exchange": {
                "failure_threshold": 5,
                "recovery_timeout": 60,
                "half_open_max_calls": 3,
            },
            "orders": {
                "failure_threshold": 3,
                "recovery_timeout": 120,
                "half_open_max_calls": 2,
            },
            "data_fetch": {
                "failure_threshold": 10,
                "recovery_timeout": 30,
                "half_open_max_calls": 5,
            },
        },
        "auth_recovery": {
            "pause_duration": 300,
            "max_retries": 3,
        },
    },
}


class ConfigManager:
    """
    Gerenciador de configuracoes centralizado e dinamico.
    Singleton thread-safe com hot-reload.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._config: Dict = {}
        self._last_load: float = 0
        self._config_lock = threading.RLock()
        self._load_config()
        self._initialized = True

    def _load_config(self):
        """Carregar configuracao do arquivo JSON."""
        with self._config_lock:
            # Comecar com defaults
            self._config = json.loads(json.dumps(DEFAULT_CONFIG))

            # Tentar carregar do arquivo
            if os.path.exists(CONFIG_FILE):
                try:
                    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                        file_config = json.load(f)
                    # Mesclar recursivamente
                    self._deep_merge(self._config, file_config)
                    log.info(f"Config carregado de {CONFIG_FILE}")
                except Exception as e:
                    log.warning(f"Erro carregando config: {e}. Usando defaults.")
            else:
                # Criar arquivo de config com defaults
                self._save_config()
                log.info(f"Config criado em {CONFIG_FILE}")

            self._last_load = datetime.now().timestamp()

    def _deep_merge(self, base: Dict, override: Dict):
        """Mescla recursivamente override em base."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def _save_config(self):
        """Salvar configuracao atual no arquivo."""
        os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
        self._config['last_updated'] = datetime.now().isoformat()
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(self._config, f, indent=2, ensure_ascii=False)

    def reload(self):
        """Recarregar configuracoes do arquivo."""
        self._load_config()
        log.info("Configuracoes recarregadas")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Obter valor de configuracao por chave.
        Suporta notacao de ponto: 'strategy.rsi_period'
        """
        with self._config_lock:
            keys = key.split('.')
            value = self._config

            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default

            return value

    def get_section(self, section: str) -> Dict:
        """Obter secao inteira de configuracao."""
        with self._config_lock:
            return self._config.get(section, {}).copy()

    def set(self, key: str, value: Any, save: bool = True):
        """
        Definir valor de configuracao.
        Suporta notacao de ponto: 'strategy.rsi_period'
        """
        with self._config_lock:
            keys = key.split('.')
            config = self._config

            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]

            config[keys[-1]] = value

            if save:
                self._save_config()

    def get_all(self) -> Dict:
        """Obter todas as configuracoes."""
        with self._config_lock:
            return json.loads(json.dumps(self._config))

    def get_strategy_params(self) -> Dict:
        """
        Obter parametros de estrategia em formato flat (compatibilidade).
        Combina strategy + risk em um unico dict.
        """
        with self._config_lock:
            params = {}

            # Strategy params
            strategy = self._config.get('strategy', {})
            for k, v in strategy.items():
                params[k] = v

            # Risk params
            risk = self._config.get('risk', {})
            for k, v in risk.items():
                params[k] = v

            # Add strategy name
            params['strategy'] = strategy.get('name', 'stoch_extreme')

            return params


# =============================================================================
# INSTANCIA GLOBAL (Singleton)
# =============================================================================
_config_manager: Optional[ConfigManager] = None


def _get_manager() -> ConfigManager:
    """Obter instancia do ConfigManager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


# =============================================================================
# API PUBLICA - FUNCOES HELPER
# =============================================================================
class Config:
    """Interface estatica para acessar configuracoes."""

    @staticmethod
    def get(key: str, default: Any = None) -> Any:
        """Obter valor por chave (suporta 'section.key')."""
        return _get_manager().get(key, default)

    @staticmethod
    def get_section(section: str) -> Dict:
        """Obter secao inteira."""
        return _get_manager().get_section(section)

    @staticmethod
    def set(key: str, value: Any, save: bool = True):
        """Definir valor."""
        _get_manager().set(key, value, save)

    @staticmethod
    def reload():
        """Recarregar do arquivo."""
        _get_manager().reload()

    @staticmethod
    def get_all() -> Dict:
        """Obter todas configs."""
        return _get_manager().get_all()

    @staticmethod
    def get_strategy_params() -> Dict:
        """Obter params de estrategia (flat dict para compatibilidade)."""
        return _get_manager().get_strategy_params()


# =============================================================================
# FUNCOES DE COMPATIBILIDADE (para codigo legado)
# =============================================================================
def get_validated_params() -> Dict:
    """COMPATIBILIDADE: Retorna parametros de estrategia."""
    return Config.get_strategy_params()


def get_param(name: str, default: Any = None) -> Any:
    """COMPATIBILIDADE: Obter parametro especifico."""
    # Primeiro tentar em strategy
    value = Config.get(f'strategy.{name}')
    if value is not None:
        return value
    # Depois em risk
    value = Config.get(f'risk.{name}')
    if value is not None:
        return value
    # Fallback para default
    return default


def get_backtest_config() -> Dict:
    """COMPATIBILIDADE: Retorna config de backtest."""
    config = Config.get_section('backtest')
    config.update(Config.get_section('fees'))
    config.update({
        'max_positions': Config.get('risk.max_positions', 10),
        'max_margin_usage': Config.get('risk.max_margin_usage', 0.8),
        'max_leverage': Config.get('risk.max_leverage', 10),
    })
    return config


def get_polling_interval(timeframe: str) -> int:
    """COMPATIBILIDADE: Retorna intervalo de polling."""
    intervals = Config.get('timeframes.polling_intervals', {})
    return intervals.get(timeframe, 60)


# =============================================================================
# VARIAVEIS DE COMPATIBILIDADE (para imports diretos)
# =============================================================================
# API
API_KEY = os.getenv('BINANCE_API_KEY', os.getenv('Binanceapikey', '')).strip()
SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', os.getenv('BinanceSecretkey', '')).strip()
USE_TESTNET = os.getenv('TESTNET', 'True').lower() == 'true'

# Lazy loading para evitar circular imports
def _get_symbols():
    return Config.get('symbols', DEFAULT_CONFIG['symbols'])

def _get_primary_tf():
    return Config.get('timeframes.primary', '1h')

# Propriedades que carregam dinamicamente
class _DynamicVars:
    @property
    def SYMBOLS(self):
        return _get_symbols()

    @property
    def PRIMARY_TIMEFRAME(self):
        return _get_primary_tf()

    @property
    def WFO_VALIDATED_PARAMS(self):
        return Config.get_strategy_params()

    @property
    def BACKTEST_CONFIG(self):
        return get_backtest_config()

_dynamic = _DynamicVars()

# CORRIGIDO (Bug #17): Removida duplicação de variáveis
# Aliases de compatibilidade para imports legados
SYMBOLS = DEFAULT_CONFIG['symbols'].copy()
PRIMARY_TIMEFRAME = DEFAULT_CONFIG['timeframes']['primary']

# Constantes derivadas
WFO_VALIDATED_PARAMS = DEFAULT_CONFIG['strategy'].copy()
WFO_VALIDATED_PARAMS.update(DEFAULT_CONFIG['risk'])
BACKTEST_CONFIG = DEFAULT_CONFIG['backtest'].copy()
DEFAULT_STRATEGY_PARAMS = WFO_VALIDATED_PARAMS.copy()

# Constantes que nao mudam
MAX_POSITIONS = 10
LEVERAGE_CAP = 10
RISK_PER_TRADE = 0.01
LOG_FILE = "logs/trades.csv"
STATE_FILE = "state/trader_state.json"

WFO_CONFIG = DEFAULT_CONFIG.copy()
PARAM_GRID = {
    'rsi_period': [14],
    'stoch_k': [14],
    'adx_min': [20],
    'sl_atr_mult': [3.0],
    'tp_atr_mult': [5.0],
}


def get_wfo_config() -> Dict:
    """COMPATIBILIDADE: Retorna config WFO para dashboard."""
    return Config.get_all()


# =============================================================================
# ERROR HANDLING CONFIG HELPERS
# =============================================================================
def get_error_handling_config() -> Dict:
    """Retorna configuracao de error handling."""
    return Config.get_section('error_handling')


def get_retry_config() -> Dict:
    """Retorna configuracao de retry."""
    return Config.get('error_handling.retry', {})


def get_circuit_breaker_config(name: str) -> Dict:
    """Retorna configuracao de um circuit breaker especifico."""
    return Config.get(f'error_handling.circuit_breaker.{name}', {})


def get_auth_recovery_config() -> Dict:
    """Retorna configuracao de recuperacao de autenticacao."""
    return Config.get('error_handling.auth_recovery', {})
