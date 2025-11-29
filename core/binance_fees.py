"""
Binance Fees & Rates Module
============================
Busca taxas dinâmicas da Binance Futures API.

Endpoints utilizados:
- GET /fapi/v1/premiumIndex - Mark price e funding rate atual
- GET /fapi/v1/fundingRate - Histórico de funding rates
- GET /fapi/v1/exchangeInfo - Informações de trading (filtros, precision)
- GET /fapi/v1/commissionRate - Taxa de comissão do usuário (requer auth)

Referências:
- https://developers.binance.com/docs/derivatives/usds-margined-futures/
- https://www.binance.com/en/support/faq/introduction-to-binance-futures-funding-rates-360033525031
- https://www.binance.com/en/support/faq/how-to-calculate-profit-and-loss-for-futures-contracts-3a55a23768cb416fb404f06ffedde4b2
"""

import requests
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import os
import time
import hmac
import hashlib
import logging
from urllib.parse import urlencode

from .config import Config

logger = logging.getLogger(__name__)

# Cache para evitar chamadas excessivas à API
_cache = {}
_cache_expiry = {}
CACHE_DURATION = 300  # 5 minutos


@dataclass
class SymbolInfo:
    """
    Informações de um símbolo conforme Binance API.

    Referência: GET /fapi/v1/exchangeInfo
    """
    symbol: str
    price_precision: int
    quantity_precision: int
    min_notional: float
    tick_size: float
    step_size: float
    liquidation_fee: float          # Taxa de liquidação (ex: 0.0125 = 1.25%)
    max_leverage: int               # Leverage máximo permitido
    maintenance_margin_rate: float  # MMR base (ex: 0.025 = 2.5%)
    initial_margin_rate: float      # IMR base = 1/leverage (ex: 0.05 = 5% para 20x)


@dataclass
class FundingInfo:
    """Informações de funding rate."""
    symbol: str
    funding_rate: float
    funding_time: datetime
    mark_price: float
    index_price: float
    next_funding_time: datetime


@dataclass
class CommissionRates:
    """Taxas de comissão."""
    symbol: str
    maker_rate: float
    taker_rate: float


@dataclass
class LeverageBracket:
    """
    Bracket de alavancagem conforme Binance.

    Referência: https://www.binance.com/en/support/faq/leverage-and-margin-of-usd-m-futures-360033162192
    """
    bracket: int                    # Número do bracket (1, 2, 3...)
    initial_leverage: int           # Alavancagem máxima neste tier
    notional_floor: float           # Valor mínimo de posição (USDT)
    notional_cap: float             # Valor máximo de posição (USDT)
    maint_margin_rate: float        # Taxa de margem de manutenção (decimal)
    cum_maintenance: float          # Valor cumulativo de manutenção (USDT)


class BinanceFees:
    """
    Gerenciador de taxas da Binance Futures.

    Busca taxas dinâmicas para cada símbolo:
    - Funding rates (a cada 8h)
    - Commission rates (maker/taker)
    - Symbol info (precision, filtros)
    """

    BASE_URL = "https://fapi.binance.com"
    TESTNET_URL = "https://testnet.binancefuture.com"

    # Taxas padrão (VIP0) - fallback se API falhar
    DEFAULT_MAKER_FEE = 0.0002  # 0.02%
    DEFAULT_TAKER_FEE = 0.0004  # 0.04%
    DEFAULT_FUNDING_RATE = 0.0001  # 0.01% (taxa base de juros / 3)

    def __init__(self, use_testnet: bool = True, api_key: str = None, api_secret: str = None):
        """
        Inicializar gerenciador de taxas.

        Args:
            use_testnet: Usar testnet ao invés de produção
            api_key: API key para endpoints autenticados (opcional)
            api_secret: API secret (opcional)
        """
        self.base_url = self.TESTNET_URL if use_testnet else self.BASE_URL
        self.use_testnet = use_testnet

        # Carregar API keys de variáveis de ambiente se não fornecidas
        self.api_key = api_key or os.environ.get('BINANCE_API_KEY') or os.environ.get('BINANCE_TESTNET_API_KEY')
        self.api_secret = api_secret or os.environ.get('BINANCE_API_SECRET') or os.environ.get('BINANCE_TESTNET_API_SECRET')

        self.session = requests.Session()

        # Cache local
        self._symbol_info_cache: Dict[str, SymbolInfo] = {}
        self._funding_cache: Dict[str, FundingInfo] = {}
        self._commission_cache: Dict[str, CommissionRates] = {}
        self._exchange_info = None
        self._last_exchange_info_update = None

    def _generate_signature(self, params: Dict) -> str:
        """
        Gerar assinatura HMAC SHA256 para requests autenticados.

        Args:
            params: Parâmetros do request

        Returns:
            Assinatura hexadecimal
        """
        if not self.api_secret:
            return ""

        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature

    def _signed_request(self, method: str, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """
        Fazer request autenticado com assinatura.

        Args:
            method: 'GET' ou 'POST'
            endpoint: Endpoint da API (ex: '/fapi/v1/commissionRate')
            params: Parâmetros adicionais

        Returns:
            Resposta JSON ou None em caso de erro
        """
        if not self.api_key or not self.api_secret:
            logger.debug("API keys não configuradas, usando taxas padrão")
            return None

        params = params or {}
        params['timestamp'] = int(time.time() * 1000)
        params['signature'] = self._generate_signature(params)

        headers = {'X-MBX-APIKEY': self.api_key}
        url = f"{self.base_url}{endpoint}"

        # Usar timeout do config
        http_timeout = Config.get('exchange.http_timeout', 10)

        try:
            if method.upper() == 'GET':
                response = self.session.get(url, params=params, headers=headers, timeout=http_timeout)
            else:
                response = self.session.post(url, data=params, headers=headers, timeout=http_timeout)

            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 401:
                logger.warning("Autenticação falhou - verificar API keys")
            elif response.status_code == 403:
                logger.warning("Acesso negado - API key pode não ter permissão")
            else:
                logger.warning(f"Erro HTTP na request autenticada: {e}")
            return None
        except Exception as e:
            logger.warning(f"Erro na request autenticada: {e}")
            return None

    def _get_cached(self, key: str, fetch_func, ttl: int = CACHE_DURATION):
        """Obter valor do cache ou buscar novo."""
        now = time.time()
        if key in _cache and key in _cache_expiry:
            if now < _cache_expiry[key]:
                return _cache[key]

        value = fetch_func()
        _cache[key] = value
        _cache_expiry[key] = now + ttl
        return value

    def get_exchange_info(self) -> Dict:
        """
        Obter informações da exchange (símbolos, filtros, etc).

        GET /fapi/v1/exchangeInfo
        """
        def fetch():
            try:
                response = self.session.get(f"{self.base_url}/fapi/v1/exchangeInfo")
                response.raise_for_status()
                return response.json()
            except Exception as e:
                logger.warning(f"Erro ao buscar exchange info: {e}")
                return None

        return self._get_cached("exchange_info", fetch, ttl=3600)  # 1 hora

    def get_symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        """
        Obter informações de um símbolo específico.

        Inclui:
        - Precisão de preço e quantidade
        - Notional mínimo
        - Taxa de liquidação
        - Leverage máximo
        """
        if symbol in self._symbol_info_cache:
            return self._symbol_info_cache[symbol]

        exchange_info = self.get_exchange_info()
        if not exchange_info:
            return None

        for sym in exchange_info.get('symbols', []):
            if sym['symbol'] == symbol:
                # Extrair filtros
                filters = {f['filterType']: f for f in sym.get('filters', [])}

                price_filter = filters.get('PRICE_FILTER', {})
                lot_size = filters.get('LOT_SIZE', {})
                min_notional = filters.get('MIN_NOTIONAL', {})

                # Extrair dados de margem conforme documentação Binance
                # maintMarginPercent = Maintenance Margin Rate (ex: "2.5000" = 2.5%)
                # requiredMarginPercent = Initial Margin Rate (ex: "5.0000" = 5%)
                maint_margin_pct = float(sym.get('maintMarginPercent', 2.5))
                init_margin_pct = float(sym.get('requiredMarginPercent', 5.0))

                info = SymbolInfo(
                    symbol=symbol,
                    price_precision=sym.get('pricePrecision', 2),
                    quantity_precision=sym.get('quantityPrecision', 3),
                    min_notional=float(min_notional.get('notional', 5)),
                    tick_size=float(price_filter.get('tickSize', 0.01)),
                    step_size=float(lot_size.get('stepSize', 0.001)),
                    liquidation_fee=float(sym.get('liquidationFee', 0.0125)),
                    max_leverage=int(sym.get('maxLeverage', 125)),
                    maintenance_margin_rate=maint_margin_pct / 100,  # Converter para decimal
                    initial_margin_rate=init_margin_pct / 100        # Converter para decimal
                )

                self._symbol_info_cache[symbol] = info
                return info

        return None

    def get_all_funding_rates(self) -> Dict[str, FundingInfo]:
        """
        Obter funding rates atuais de todos os símbolos.

        GET /fapi/v1/premiumIndex

        Returns:
            Dict com símbolo -> FundingInfo
        """
        def fetch():
            try:
                response = self.session.get(f"{self.base_url}/fapi/v1/premiumIndex")
                response.raise_for_status()
                data = response.json()

                result = {}
                for item in data:
                    symbol = item['symbol']
                    result[symbol] = FundingInfo(
                        symbol=symbol,
                        funding_rate=float(item.get('lastFundingRate', 0)),
                        funding_time=datetime.fromtimestamp(int(item.get('time', 0)) / 1000),
                        mark_price=float(item.get('markPrice', 0)),
                        index_price=float(item.get('indexPrice', 0)),
                        next_funding_time=datetime.fromtimestamp(
                            int(item.get('nextFundingTime', 0)) / 1000
                        )
                    )
                return result
            except Exception as e:
                logger.warning(f"Erro ao buscar funding rates: {e}")
                return {}

        return self._get_cached("all_funding_rates", fetch, ttl=60)  # 1 minuto

    def get_funding_rate(self, symbol: str) -> float:
        """
        Obter funding rate atual de um símbolo.

        Args:
            symbol: Par de trading (ex: BTCUSDT)

        Returns:
            Funding rate (ex: 0.0001 = 0.01%)
        """
        all_rates = self.get_all_funding_rates()
        if symbol in all_rates:
            return all_rates[symbol].funding_rate
        return self.DEFAULT_FUNDING_RATE

    def get_funding_rate_history(
        self,
        symbol: str,
        start_time: datetime = None,
        end_time: datetime = None,
        limit: int = 100
    ) -> List[Tuple[datetime, float]]:
        """
        Obter histórico de funding rates.

        GET /fapi/v1/fundingRate

        Args:
            symbol: Par de trading
            start_time: Início do período
            end_time: Fim do período
            limit: Máximo de registros (até 1000)

        Returns:
            Lista de (timestamp, rate)
        """
        params = {'symbol': symbol, 'limit': min(limit, 1000)}

        if start_time:
            params['startTime'] = int(start_time.timestamp() * 1000)
        if end_time:
            params['endTime'] = int(end_time.timestamp() * 1000)

        try:
            response = self.session.get(
                f"{self.base_url}/fapi/v1/fundingRate",
                params=params
            )
            response.raise_for_status()
            data = response.json()

            return [
                (
                    datetime.fromtimestamp(int(item['fundingTime']) / 1000),
                    float(item['fundingRate'])
                )
                for item in data
            ]
        except Exception as e:
            logger.warning(f"Erro ao buscar histórico de funding: {e}")
            return []

    def get_average_funding_rate(self, symbol: str, days: int = 30, use_current_as_estimate: bool = True) -> float:
        """
        Calcular funding rate médio dos últimos N dias.

        Args:
            symbol: Par de trading
            days: Número de dias para média
            use_current_as_estimate: Se True, usa rate atual como estimativa (mais rápido)

        Returns:
            Funding rate médio
        """
        # Cache por 6 horas para evitar chamadas API repetidas
        cache_key = f"avg_funding_{symbol}_{days}"

        def fetch():
            # Opção rápida: usar o funding rate atual como estimativa
            # Isso evita uma chamada API extra para histórico
            if use_current_as_estimate:
                current_rate = self.get_funding_rate(symbol)
                if current_rate != 0:
                    return current_rate

            # Fallback: buscar histórico (mais lento)
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)

            history = self.get_funding_rate_history(
                symbol,
                start_time=start_time,
                end_time=end_time,
                limit=days * 3  # 3 funding periods por dia
            )

            if not history:
                return self.DEFAULT_FUNDING_RATE

            rates = [rate for _, rate in history]
            return sum(rates) / len(rates)

        return self._get_cached(cache_key, fetch, ttl=21600)  # Cache 6 horas

    def get_commission_rates(self, symbol: str = "BTCUSDT") -> CommissionRates:
        """
        Obter taxas de comissão da API (autenticado).

        Endpoint: GET /fapi/v1/commissionRate (requer autenticação)
        Documentação: https://developers.binance.com/docs/derivatives/usds-margined-futures/account/rest-api/User-Commission-Rate

        Args:
            symbol: Par de trading (ex: BTCUSDT)

        Returns:
            CommissionRates com maker/taker fees personalizadas do usuário
        """
        # Verificar cache primeiro
        if symbol in self._commission_cache:
            return self._commission_cache[symbol]

        def fetch_from_api():
            # Tentar buscar da API autenticada
            data = self._signed_request('GET', '/fapi/v1/commissionRate', {'symbol': symbol})

            if data:
                maker_rate = float(data.get('makerCommissionRate', self.DEFAULT_MAKER_FEE))
                taker_rate = float(data.get('takerCommissionRate', self.DEFAULT_TAKER_FEE))
                logger.info(f"Taxas obtidas da API para {symbol}: maker={maker_rate*100:.4f}%, taker={taker_rate*100:.4f}%")
                return CommissionRates(
                    symbol=symbol,
                    maker_rate=maker_rate,
                    taker_rate=taker_rate
                )
            else:
                # Fallback para taxas padrão se API falhar
                logger.debug(f"Usando taxas padrão para {symbol} (API não disponível)")
                return CommissionRates(
                    symbol=symbol,
                    maker_rate=self.DEFAULT_MAKER_FEE,
                    taker_rate=self.DEFAULT_TAKER_FEE
                )

        cache_key = f"commission_{symbol}"
        result = self._get_cached(cache_key, fetch_from_api, ttl=3600)  # Cache 1 hora

        # Armazenar no cache local também
        self._commission_cache[symbol] = result
        return result

    def get_mark_price(self, symbol: str) -> float:
        """
        Obter mark price atual de um símbolo.

        Args:
            symbol: Par de trading

        Returns:
            Mark price
        """
        all_rates = self.get_all_funding_rates()
        if symbol in all_rates:
            return all_rates[symbol].mark_price
        return 0.0

    def calculate_funding_fee(
        self,
        symbol: str,
        position_size: float,
        entry_price: float,
        side: str = 'long'
    ) -> float:
        """
        Calcular taxa de funding para uma posição.

        Fórmula Binance:
        Funding Fee = Position Value × Funding Rate
        Position Value = Mark Price × Position Size

        Args:
            symbol: Par de trading
            position_size: Tamanho da posição (em coins)
            entry_price: Preço de entrada (usado se mark price indisponível)
            side: 'long' ou 'short'

        Returns:
            Taxa de funding (positivo = paga, negativo = recebe)
        """
        funding_rate = self.get_funding_rate(symbol)
        mark_price = self.get_mark_price(symbol) or entry_price

        position_value = mark_price * abs(position_size)
        funding_fee = position_value * funding_rate

        # Long paga quando rate positivo, short recebe
        # Short paga quando rate negativo, long recebe
        if side == 'long':
            return funding_fee  # Positivo = paga
        else:
            return -funding_fee  # Negativo = recebe

    def calculate_trading_fee(
        self,
        symbol: str,
        notional_value: float,
        is_maker: bool = False
    ) -> float:
        """
        Calcular taxa de trading.

        Args:
            symbol: Par de trading
            notional_value: Valor da ordem
            is_maker: True para maker, False para taker

        Returns:
            Taxa de trading
        """
        rates = self.get_commission_rates(symbol)
        rate = rates.maker_rate if is_maker else rates.taker_rate
        return notional_value * rate

    def get_all_fees_for_symbol(self, symbol: str) -> Dict:
        """
        Obter todas as taxas relevantes para um símbolo.

        Returns:
            Dict com todas as taxas
        """
        funding = self.get_all_funding_rates().get(symbol)
        commission = self.get_commission_rates(symbol)
        symbol_info = self.get_symbol_info(symbol)

        return {
            'symbol': symbol,
            'funding_rate': funding.funding_rate if funding else self.DEFAULT_FUNDING_RATE,
            'next_funding_time': funding.next_funding_time if funding else None,
            'mark_price': funding.mark_price if funding else 0,
            'maker_fee': commission.maker_rate,
            'taker_fee': commission.taker_rate,
            'liquidation_fee': symbol_info.liquidation_fee if symbol_info else 0.0125,
            'min_notional': symbol_info.min_notional if symbol_info else 5,
            'maintenance_margin_rate': symbol_info.maintenance_margin_rate if symbol_info else 0.025,
            'initial_margin_rate': symbol_info.initial_margin_rate if symbol_info else 0.05,
            'max_leverage': symbol_info.max_leverage if symbol_info else 20
        }

    def get_fees_for_symbols(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Obter taxas para múltiplos símbolos.

        Args:
            symbols: Lista de pares de trading

        Returns:
            Dict com símbolo -> taxas
        """
        return {symbol: self.get_all_fees_for_symbol(symbol) for symbol in symbols}

    def get_leverage_brackets(self, symbol: str) -> List[LeverageBracket]:
        """
        Obter brackets de alavancagem para um símbolo.

        Conforme Binance, cada par tem diferentes tiers de alavancagem
        baseados no tamanho da posição (notional value).

        Referência: https://www.binance.com/en/support/faq/leverage-and-margin-of-usd-m-futures-360033162192
        """
        # Brackets padrão para BTCUSDT (mais conservadores)
        # Fonte: Binance Leverage Tiers (atualizado 2025)
        btc_brackets = [
            LeverageBracket(1, 125, 0, 50000, 0.004, 0),
            LeverageBracket(2, 100, 50000, 250000, 0.005, 50),
            LeverageBracket(3, 50, 250000, 1000000, 0.01, 1300),
            LeverageBracket(4, 20, 1000000, 10000000, 0.025, 16300),
            LeverageBracket(5, 10, 10000000, 50000000, 0.05, 266300),
            LeverageBracket(6, 5, 50000000, 100000000, 0.10, 2766300),
            LeverageBracket(7, 4, 100000000, 200000000, 0.125, 5266300),
            LeverageBracket(8, 3, 200000000, 300000000, 0.15, 10266300),
            LeverageBracket(9, 2, 300000000, 500000000, 0.25, 40266300),
            LeverageBracket(10, 1, 500000000, float('inf'), 0.50, 165266300),
        ]

        # Brackets para ETH (similares a BTC mas com valores menores)
        eth_brackets = [
            LeverageBracket(1, 100, 0, 10000, 0.005, 0),
            LeverageBracket(2, 75, 10000, 100000, 0.0065, 15),
            LeverageBracket(3, 50, 100000, 500000, 0.01, 515),
            LeverageBracket(4, 25, 500000, 2000000, 0.02, 5515),
            LeverageBracket(5, 10, 2000000, 10000000, 0.05, 65515),
            LeverageBracket(6, 5, 10000000, 20000000, 0.10, 565515),
            LeverageBracket(7, 2, 20000000, 50000000, 0.25, 3565515),
            LeverageBracket(8, 1, 50000000, float('inf'), 0.50, 16065515),
        ]

        # Brackets padrão para outras altcoins (mais conservadores)
        default_brackets = [
            LeverageBracket(1, 50, 0, 5000, 0.01, 0),
            LeverageBracket(2, 25, 5000, 25000, 0.02, 50),
            LeverageBracket(3, 10, 25000, 100000, 0.05, 800),
            LeverageBracket(4, 5, 100000, 500000, 0.10, 5800),
            LeverageBracket(5, 2, 500000, 1000000, 0.25, 80800),
            LeverageBracket(6, 1, 1000000, float('inf'), 0.50, 330800),
        ]

        if 'BTC' in symbol:
            return btc_brackets
        elif 'ETH' in symbol:
            return eth_brackets
        else:
            return default_brackets

    def get_bracket_for_notional(self, symbol: str, notional: float) -> LeverageBracket:
        """
        Obter o bracket apropriado para um determinado valor de posição.

        Args:
            symbol: Par de trading
            notional: Valor da posição em USDT

        Returns:
            LeverageBracket correspondente ao notional value
        """
        brackets = self.get_leverage_brackets(symbol)
        for bracket in brackets:
            if bracket.notional_floor <= notional < bracket.notional_cap:
                return bracket
        return brackets[-1]

    def calculate_maintenance_margin(
        self,
        symbol: str,
        notional: float
    ) -> Tuple[float, float, float]:
        """
        Calcular margem de manutenção conforme Binance.

        Fórmula: Maintenance Margin = Notional × MMR - cum

        Args:
            symbol: Par de trading
            notional: Valor da posição em USDT

        Returns:
            Tuple (maintenance_margin, mmr, cum)
        """
        bracket = self.get_bracket_for_notional(symbol, notional)
        mmr = bracket.maint_margin_rate
        cum = bracket.cum_maintenance
        maintenance_margin = notional * mmr - cum
        return maintenance_margin, mmr, cum

    def calculate_liquidation_price(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        wallet_balance: float,
        leverage: int = None
    ) -> float:
        """
        Calcular preço de liquidação conforme fórmula da Binance.

        Fórmula (Isolated Margin Mode):
        Liq_Price = (WB + Maint_Amount - Contract_Qty × Entry_Price) /
                    (|Contract_Qty| × (MMR - Sign))

        Onde:
        - WB = Wallet Balance (margem isolada da posição)
        - Maint_Amount = Valor cumulativo de manutenção (cum)
        - Contract_Qty = Quantidade (positiva para LONG, negativa para SHORT)
        - Entry_Price = Preço de entrada
        - MMR = Maintenance Margin Rate
        - Sign = +1 para LONG, -1 para SHORT

        Args:
            symbol: Par de trading
            side: 'long' ou 'short'
            entry_price: Preço de entrada
            quantity: Quantidade absoluta
            wallet_balance: Margem isolada (initial margin)
            leverage: Alavancagem (opcional, para determinar notional)

        Returns:
            Preço de liquidação
        """
        # Valor da posição
        notional = quantity * entry_price

        # Obter bracket e valores de manutenção
        _, mmr, cum = self.calculate_maintenance_margin(symbol, notional)

        # Definir sign baseado na direção
        if side == 'long':
            sign = 1
            contract_qty = quantity  # Positivo para LONG
        else:
            sign = -1
            contract_qty = -quantity  # Negativo para SHORT

        # Fórmula de liquidação
        # Liq = (WB + cum - Qty × Entry) / (|Qty| × (MMR - Sign))
        numerator = wallet_balance + cum - contract_qty * entry_price
        denominator = abs(contract_qty) * (mmr - sign)

        if denominator == 0:
            # Evitar divisão por zero
            return 0 if side == 'long' else float('inf')

        liq_price = numerator / denominator

        # Validar resultado
        if side == 'long':
            # Para LONG, liq_price deve ser menor que entry_price
            if liq_price >= entry_price or liq_price <= 0:
                # Fallback para cálculo simplificado
                liq_price = entry_price * (1 - 1 / leverage + mmr) if leverage else entry_price * 0.5
        else:
            # Para SHORT, liq_price deve ser maior que entry_price
            if liq_price <= entry_price:
                # Fallback para cálculo simplificado
                liq_price = entry_price * (1 + 1 / leverage - mmr) if leverage else entry_price * 1.5

        return max(0, liq_price)


# Singleton para uso global
_binance_fees_instance = None

def get_binance_fees(use_testnet: bool = True) -> BinanceFees:
    """Obter instância singleton do gerenciador de taxas."""
    global _binance_fees_instance
    if _binance_fees_instance is None:
        _binance_fees_instance = BinanceFees(use_testnet=use_testnet)
    return _binance_fees_instance


# Funções de conveniência
def get_funding_rate(symbol: str) -> float:
    """Obter funding rate de um símbolo."""
    return get_binance_fees().get_funding_rate(symbol)

def get_trading_fees(symbol: str) -> Tuple[float, float]:
    """Obter taxas maker/taker de um símbolo."""
    rates = get_binance_fees().get_commission_rates(symbol)
    return rates.maker_rate, rates.taker_rate

def get_all_fees(symbols: List[str]) -> Dict[str, Dict]:
    """Obter todas as taxas para uma lista de símbolos."""
    return get_binance_fees().get_fees_for_symbols(symbols)


if __name__ == "__main__":
    # Teste do módulo
    print("=" * 60)
    print("TESTE DO MÓDULO DE TAXAS BINANCE")
    print("=" * 60)

    fees = BinanceFees(use_testnet=False)  # Usar produção para ter dados reais

    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'DOGEUSDT']

    print("\nBuscando taxas atuais...")

    for symbol in symbols:
        print(f"\n{symbol}:")
        info = fees.get_all_fees_for_symbol(symbol)

        print(f"  Funding Rate: {info['funding_rate']*100:.4f}%")
        print(f"  Mark Price: ${info['mark_price']:,.2f}")
        print(f"  Maker Fee: {info['maker_fee']*100:.3f}%")
        print(f"  Taker Fee: {info['taker_fee']*100:.3f}%")
        print(f"  Liquidation Fee: {info['liquidation_fee']*100:.2f}%")
        print(f"  Min Notional: ${info['min_notional']}")

    # Teste de histórico de funding
    print(f"\n\nHistórico de Funding (BTCUSDT últimos 7 dias):")
    history = fees.get_funding_rate_history('BTCUSDT', limit=21)  # 3 por dia * 7 dias
    if history:
        avg_rate = sum(r for _, r in history) / len(history)
        print(f"  Registros: {len(history)}")
        print(f"  Taxa média: {avg_rate*100:.4f}%")
        print(f"  Taxa mínima: {min(r for _, r in history)*100:.4f}%")
        print(f"  Taxa máxima: {max(r for _, r in history)*100:.4f}%")
