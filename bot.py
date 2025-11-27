"""
Bot Principal - Sistema de Trading Automatizado
Versao otimizada e simplificada
"""
import sys
import os
import time
import json
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import ccxt

from core.config import (
    API_KEY, SECRET_KEY, USE_TESTNET,
    SYMBOLS, PRIMARY_TIMEFRAME, MAX_POSITIONS,
    WFO_VALIDATED_PARAMS, get_validated_params
)
from core.trader import Trader
from core.signals import SignalGenerator

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler("logs/bot.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)


class TradingBot:
    """Bot de trading principal."""
    
    def __init__(self):
        log.info("Inicializando bot...")
        
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
        
        log.info(f"Bot inicializado | Symbols: {len(SYMBOLS)} | TF: {PRIMARY_TIMEFRAME}")
        log.info(f"Estrategia: {self.params.get('strategy', 'rsi_extremes')}")
    
    def _sync_existing_positions(self):
        """Sincronizar posicoes existentes na Binance com o estado local."""
        from core.trader import Position
        from core.signals import calculate_atr

        try:
            account = self.exchange.fapiPrivateV2GetAccount()
            positions = account.get('positions', [])
            open_positions = [p for p in positions if float(p.get('positionAmt', 0)) != 0]

            # CORRIGIDO: Usar ATR multipliers (não porcentagem!)
            # Estes são multiplicadores de ATR, não percentuais
            sl_atr_mult = self.params.get('sl_atr_mult', self.params.get('sl_mult', 3.0))
            tp_atr_mult = self.params.get('tp_atr_mult', self.params.get('tp_mult', 5.0))
            atr_period = self.params.get('atr_period', 14)

            for p in open_positions:
                raw_sym = p.get('symbol', '')
                sym = raw_sym.replace('USDT', '/USDT')

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
                            # Fallback: usar % conservador se não conseguir ATR
                            sl_pct = 0.03  # 3% fallback
                            tp_pct = 0.05  # 5% fallback
                            if side == 'long':
                                stop_loss = entry * (1 - sl_pct)
                                take_profit = entry * (1 + tp_pct)
                            else:
                                stop_loss = entry * (1 + sl_pct)
                                take_profit = entry * (1 - tp_pct)
                    except Exception as e:
                        log.warning(f"Erro calculando ATR para {sym}: {e}, usando fallback")
                        sl_pct = 0.03
                        tp_pct = 0.05
                        if side == 'long':
                            stop_loss = entry * (1 - sl_pct)
                            take_profit = entry * (1 + tp_pct)
                        else:
                            stop_loss = entry * (1 + sl_pct)
                            take_profit = entry * (1 - tp_pct)

                    # Adicionar ao estado local usando a dataclass Position
                    pos = Position(
                        symbol=sym,
                        side=side,
                        entry_price=entry,
                        quantity=abs(position_amt),
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        entry_time=datetime.now().isoformat(),
                        max_price=entry,
                        min_price=entry,
                        reason_entry='Synced from Binance',
                        strategy=self.params.get('strategy', 'unknown')
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
        exchange = ccxt.binance({
            'apiKey': API_KEY,
            'secret': SECRET_KEY,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True,
                'recvWindow': 60000,
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
        """Carregar parametros otimizados - VALIDADOS WFO.

        Ordem de prioridade:
        1. current_best.json (estratégia ativa)
        2. trader_state.json (estado do trader)
        3. WFO_VALIDATED_PARAMS do core/config.py (defaults centralizados)
        """
        # Default centralizado - fonte única de verdade
        default = get_validated_params()

        # PRIORIDADE 1: current_best.json (estratégia ativa)
        try:
            if os.path.exists('state/current_best.json'):
                with open('state/current_best.json', 'r') as f:
                    data = json.load(f)
                    params = data.get('params', {})
                    if params:
                        # Normalizar nomes de parâmetros (suportar ambos formatos)
                        params = self._normalize_params(params)
                        log.info(f"Params carregados do current_best: {data.get('strategy_name', 'unknown')}")
                        return {**default, **params}
        except Exception as e:
            log.warning(f"Erro ao carregar current_best: {e}")

        # PRIORIDADE 2: trader_state (atualizado pelo auto_evolve)
        try:
            if os.path.exists('state/trader_state.json'):
                with open('state/trader_state.json', 'r') as f:
                    data = json.load(f)
                    params = data.get('params', {})
                    if params:
                        log.info(f"Params carregados do trader_state: strategy={params.get('strategy')}")
                        return {**default, **params}
        except Exception as e:
            log.warning(f"Erro ao carregar params do trader_state: {e}")

        # PRIORIDADE 3: optimized_params
        try:
            if os.path.exists('state/optimized_params.json'):
                with open('state/optimized_params.json', 'r') as f:
                    data = json.load(f)
                    params = data.get('params', {})
                    return {**default, **params}
        except Exception as e:
            log.warning(f"Erro ao carregar params: {e}")

        log.info("Usando params default (validados)")
        return default

    def _normalize_params(self, params: dict) -> dict:
        """Normalizar nomes de parâmetros para consistência."""
        normalized = params.copy()

        # Suportar ambos formatos de nome para SL/TP
        if 'sl_mult' in params and 'sl_atr_mult' not in params:
            normalized['sl_atr_mult'] = params['sl_mult']
        if 'tp_mult' in params and 'tp_atr_mult' not in params:
            normalized['tp_atr_mult'] = params['tp_mult']

        # Garantir que ambos formatos existem
        if 'sl_atr_mult' in normalized:
            normalized['sl_mult'] = normalized['sl_atr_mult']
        if 'tp_atr_mult' in normalized:
            normalized['tp_mult'] = normalized['tp_atr_mult']

        return normalized

    def check_reload_params(self):
        """Verificar se precisa recarregar parametros."""
        flag_path = 'state/reload_params.flag'

        if os.path.exists(flag_path):
            try:
                # Recarregar parametros
                new_params = self._load_params()

                # Se estrategia mudou, recriar signal generator
                if new_params.get('strategy') != self.params.get('strategy'):
                    log.info(f"Estrategia alterada: {self.params.get('strategy')} -> {new_params.get('strategy')}")

                self.params = new_params
                self.trader.params = new_params
                self.trader.signal_generator = SignalGenerator(new_params)

                log.info(f"Parametros recarregados: {new_params.get('strategy')}")

                # Remover flag
                os.remove(flag_path)

            except Exception as e:
                log.error(f"Erro recarregando params: {e}")
    
    def fetch_ohlcv(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Buscar dados OHLCV."""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, PRIMARY_TIMEFRAME, limit=limit)
            if not ohlcv:
                return pd.DataFrame()
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            log.error(f"Erro fetch {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_balance(self) -> dict:
        """Buscar saldo."""
        try:
            balance = self.exchange.fetch_balance()
            return {
                'total': float(balance['total'].get('USDT', 0)),
                'free': float(balance['free'].get('USDT', 0))
            }
        except Exception as e:
            log.error(f"Erro fetch balance: {e}")
            return {'total': 0, 'free': 0}
    
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
        except:
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
        """Executar ordem com validacao e SL/TP na exchange."""
        try:
            # Validar quantidade
            quantity = self.validate_quantity(symbol, quantity)
            if quantity <= 0:
                log.warning(f"Quantidade invalida para {symbol}")
                return False

            # Configurar leverage (usar parametro ao inves de hardcoded)
            leverage = self.params.get('leverage', self.params.get('max_leverage', 5))
            self.set_leverage(symbol, leverage)

            params = {'reduceOnly': True} if reduce_only else {}

            order = self.exchange.create_market_order(
                symbol, side, quantity, params=params
            )

            log.info(f"Ordem executada: {side.upper()} {symbol} qty={quantity:.4f}")

            # NOVO: Criar ordens SL/TP na exchange para proteger posição
            if not reduce_only and stop_loss and take_profit:
                self._place_sl_tp_orders(symbol, side, quantity, stop_loss, take_profit)

            return True
        except Exception as e:
            log.error(f"Erro ordem {symbol}: {e}")
            return False

    def _place_sl_tp_orders(self, symbol: str, entry_side: str, quantity: float,
                            stop_loss: float, take_profit: float) -> bool:
        """
        Colocar ordens STOP_MARKET e TAKE_PROFIT_MARKET na exchange.
        Estas protegem a posição mesmo se o bot crashar.
        """
        try:
            # Lado oposto para fechar a posição
            close_side = 'SELL' if entry_side.lower() == 'buy' else 'BUY'
            raw_symbol = symbol.replace('/', '')

            # Cancelar ordens existentes para este símbolo (evitar duplicatas)
            try:
                self.exchange.cancel_all_orders(symbol)
            except Exception:
                pass  # OK se não houver ordens

            # Ordem STOP_MARKET (Stop Loss)
            # CORRIGIDO: timeInForce removido (não necessário para STOP_MARKET)
            # CORRIGIDO: reduceOnly como string 'true' (API Binance aceita string)
            try:
                sl_order = self.exchange.fapiPrivatePostOrder({
                    'symbol': raw_symbol,
                    'side': close_side,
                    'type': 'STOP_MARKET',
                    'quantity': str(quantity),
                    'stopPrice': str(round(stop_loss, 8)),
                    'reduceOnly': 'true'
                })
                log.info(f"SL order criada: {symbol} @ {stop_loss:.6f}")
            except Exception as e:
                log.warning(f"Erro criando SL order {symbol}: {e}")

            # Ordem TAKE_PROFIT_MARKET (Take Profit)
            # CORRIGIDO: timeInForce removido (não necessário para TAKE_PROFIT_MARKET)
            try:
                tp_order = self.exchange.fapiPrivatePostOrder({
                    'symbol': raw_symbol,
                    'side': close_side,
                    'type': 'TAKE_PROFIT_MARKET',
                    'quantity': str(quantity),
                    'stopPrice': str(round(take_profit, 8)),
                    'reduceOnly': 'true'
                })
                log.info(f"TP order criada: {symbol} @ {take_profit:.6f}")
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
    
    def process_symbol(self, symbol: str) -> dict:
        """Processar um simbolo."""
        result = {'symbol': symbol, 'action': None}
        
        try:
            df = self.fetch_ohlcv(symbol, limit=100)
            if df.empty:
                return result
            
            action = self.trader.process_candle(symbol, df)
            
            if action:
                result['action'] = action
                
                if action['action'] == 'open':
                    log.info(f"SINAL {symbol}: {action['side'].upper()} | Reason: {action['reason']} | Strength: {action['strength']:.1f}")
                elif action['action'] == 'close':
                    log.info(f"EXIT {symbol}: {action['reason']}")
            
            # Guardar ultimo preco
            self.last_prices[symbol] = df.iloc[-1]['close']
            
        except Exception as e:
            log.error(f"Erro processando {symbol}: {e}")
        
        return result
    
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
                sym = raw_sym.replace('USDT', '/USDT')
                exchange_symbols.add(sym)
            
            # Verificar posicoes fechadas externamente
            for symbol in list(self.trader.positions.keys()):
                if symbol not in exchange_symbols:
                    log.info(f"Posicao {symbol} fechada externamente")
                    if symbol in self.last_prices:
                        self.trader.record_exit(symbol, self.last_prices[symbol], 'EXTERNAL')
            
        except Exception as e:
            log.error(f"Erro sync: {e}")
    
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
        
        # Processar simbolos em paralelo
        actions_to_execute = []
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(self.process_symbol, s): s for s in SYMBOLS}
            
            for future in as_completed(futures):
                result = future.result()
                if result.get('action'):
                    actions_to_execute.append(result)
        
        # Executar acoes
        for result in actions_to_execute:
            action = result['action']
            symbol = result['symbol']
            
            if action['action'] == 'open':
                # Verificar se pode abrir
                max_pos = self.params.get('max_positions', 10)
                if len(self.trader.positions) >= max_pos:
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

            elif action['action'] == 'close':
                # CORRIGIDO: Verificar se posição ainda existe antes de fechar
                if symbol not in self.trader.positions:
                    log.debug(f"Posição {symbol} já foi fechada, ignorando ação de close")
                    continue

                success = self.execute_order(symbol, action['side'], action['quantity'], reduce_only=True)

                if success and symbol in self.last_prices:
                    # Cancelar ordens SL/TP da exchange quando fechamos manualmente
                    self._cancel_sl_tp_orders(symbol)
                    self.trader.record_exit(symbol, self.last_prices[symbol], action['reason'])
        
        # Salvar estado
        self.trader.save_state('state/trader_state.json')
        
        # Log stats
        stats = self.trader.get_stats()
        log.info(f"Balance: ${stats['balance']:.2f} | Positions: {len(self.trader.positions)} | Trades: {stats['total_trades']} | WR: {stats['win_rate']:.1f}%")
    
    def run(self):
        """Loop principal."""
        log.info("Iniciando loop de trading...")
        
        # Intervalo baseado no timeframe
        if PRIMARY_TIMEFRAME == '1h':
            interval = 60  # 1 minuto (verificar a cada minuto)
        elif PRIMARY_TIMEFRAME == '4h':
            interval = 300  # 5 minutos
        else:
            interval = 30  # 30 segundos para timeframes menores
        
        while self.running:
            try:
                self.run_cycle()
                time.sleep(interval)
                
            except KeyboardInterrupt:
                log.info("Bot parado pelo usuario")
                self.running = False
            except Exception as e:
                log.error(f"Erro no loop: {e}")
                time.sleep(60)
        
        # Salvar estado final
        self.trader.save_state('state/trader_state.json')
        log.info("Bot finalizado")


def main():
    os.makedirs('logs', exist_ok=True)
    os.makedirs('state', exist_ok=True)
    
    bot = TradingBot()
    bot.run()


if __name__ == "__main__":
    main()
