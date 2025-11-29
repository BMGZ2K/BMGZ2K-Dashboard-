"""
DASHBOARD DE TRADING - Interface Web
=====================================
Monitora estrategias, metricas e resultados em tempo real

VERSÃO: 2.0 - Melhorias de estabilidade e UX
- Exception handling melhorado
- Cache otimizado (1s TTL)
- Logging adequado
"""
import sys
import os
import json
import time
import logging
import threading
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, jsonify, request
import ccxt

from core.config import (
    API_KEY, SECRET_KEY, USE_TESTNET, SYMBOLS, PRIMARY_TIMEFRAME,
    LEVERAGE_CAP, get_wfo_config, Config
)
from core.evolution import get_storage
from core.utils import load_json_safe
from core.error_handling import get_health_status, get_all_circuit_breakers

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# =============================================================================
# CACHE UNIFICADO - Todos os dados sincronizados (THREAD-SAFE)
# =============================================================================
# TTL padrão para todos os caches (do config)
UNIFIED_CACHE_TTL = Config.get('timeouts.dashboard_cache_ttl', 1.0)

# Lock global para thread-safety dos caches
_cache_lock = threading.Lock()

# Cache de estado local (para evitar leituras repetidas de arquivo)
_trader_state_cache = {
    'data': {},
    'timestamp': 0
}

# Exchange connection cache
_exchange_cache = {
    'instance': None,
    'timestamp': 0
}
EXCHANGE_CACHE_TTL = Config.get('timeouts.exchange_cache_ttl', 60)

# Cache de dados da Binance - UNIFICADO para garantir consistência
# Todos os endpoints usam os mesmos dados quando dentro do TTL
_binance_data_cache = {
    'account': {'data': None, 'timestamp': 0},
    'tickers': {'data': None, 'timestamp': 0},
    'last_sync': 0,  # Timestamp da última sincronização completa
}


def get_cached_account():
    """Retorna dados da conta com cache (thread-safe, sem race condition)."""
    global _binance_data_cache
    now = time.time()

    with _cache_lock:
        cache = _binance_data_cache['account']

        # Usar cache se ainda válido
        if cache['data'] and (now - cache['timestamp']) < UNIFIED_CACHE_TTL:
            return cache['data']

        # Buscar DENTRO do lock para evitar double-fetch (race condition fix)
        try:
            exchange = get_exchange()
            account = exchange.fapiPrivateV2GetAccount()
            _binance_data_cache['account']['data'] = account
            _binance_data_cache['account']['timestamp'] = now
            _binance_data_cache['last_sync'] = now
            return account
        except Exception as e:
            logger.error(f"Erro ao buscar conta: {e}")
            return cache['data']  # Retorna cache antigo se falhar


def get_cached_tickers():
    """
    Retorna tickers/preços com cache (thread-safe).
    Usa fapiPublicGetPremiumIndex que é muito mais rápido que fetch_tickers.
    """
    global _binance_data_cache
    now = time.time()

    with _cache_lock:
        cache = _binance_data_cache['tickers']
        # Usar cache se ainda válido
        if cache['data'] and (now - cache['timestamp']) < UNIFIED_CACHE_TTL:
            return cache['data']

    # Buscar fora do lock para não bloquear outras threads
    try:
        exchange = get_exchange()

        # Buscar mark prices (0.5s para todos os símbolos vs 8s+ para fetch_tickers)
        mark_prices = exchange.fapiPublicGetPremiumIndex()
        tickers = {}

        for p in mark_prices:
            raw_sym = p.get('symbol', '')
            sym = raw_sym.replace('USDT', '/USDT')
            mark_price = float(p.get('markPrice', 0) or 0)
            if mark_price > 0:
                tickers[sym] = {'last': mark_price, 'symbol': sym}

        with _cache_lock:
            _binance_data_cache['tickers']['data'] = tickers
            _binance_data_cache['tickers']['timestamp'] = now
        return tickers
    except Exception as e:
        logger.error(f"Erro ao buscar tickers: {e}")
        with _cache_lock:
            return _binance_data_cache['tickers']['data'] or {}


def get_exchange():
    """Criar conexao com exchange (com cache)."""
    global _exchange_cache
    now = time.time()

    # Reusar conexão existente se ainda válida
    if _exchange_cache['instance'] and (now - _exchange_cache['timestamp']) < EXCHANGE_CACHE_TTL:
        return _exchange_cache['instance']

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
    try:
        exchange.load_time_difference()
    except ccxt.NetworkError as e:
        logger.warning(f"Erro de rede ao sincronizar tempo: {e}")
    except Exception as e:
        logger.warning(f"Erro ao sincronizar tempo: {e}")

    _exchange_cache['instance'] = exchange
    _exchange_cache['timestamp'] = now
    return exchange





def get_trader_state_cached():
    """Retorna trader_state com cache unificado (thread-safe)."""
    global _trader_state_cache
    now = time.time()

    with _cache_lock:
        if now - _trader_state_cache['timestamp'] > UNIFIED_CACHE_TTL:
            _trader_state_cache['data'] = load_json_safe('state/trader_state.json')
            _trader_state_cache['timestamp'] = now
        return _trader_state_cache['data']


@app.route('/')
def index():
    """Pagina principal."""
    return render_template('dashboard.html')


@app.route('/api/health')
def api_health():
    """
    Health check endpoint - Status dos circuit breakers e sistema.

    Retorna:
        - status: 'healthy' ou 'degraded'
        - circuit_breakers: estado de cada circuit breaker
        - timestamp: momento da verificacao
        - bot_running: se o bot esta rodando (via lock file)
    """
    try:
        # Obter status dos circuit breakers
        health = get_health_status()

        # Verificar se bot esta rodando (via lock file)
        bot_lock_file = 'state/bot.lock'
        bot_running = os.path.exists(bot_lock_file)

        if bot_running:
            try:
                with open(bot_lock_file, 'r') as f:
                    bot_pid = f.read().strip()
                health['bot_pid'] = bot_pid
            except Exception:
                pass

        health['bot_running'] = bot_running

        # Uptime do dashboard (aproximado)
        health['dashboard_uptime'] = time.time() - app.config.get('start_time', time.time())

        return jsonify(health)

    except Exception as e:
        logger.error(f"Erro em /api/health: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/api/status')
def api_status():
    """Status geral do sistema - com cache para evitar rate limits."""
    try:
        # Usar cache para evitar chamadas excessivas à Binance
        account = get_cached_account()
        if not account:
            raise Exception("Dados da conta indisponíveis")
        
        # Balances
        wallet_balance = float(account.get('totalWalletBalance', 0))
        margin_balance = float(account.get('totalMarginBalance', 0))
        available_balance = float(account.get('availableBalance', 0))
        
        # Margens
        initial_margin = float(account.get('totalInitialMargin', 0))
        maint_margin = float(account.get('totalMaintMargin', 0))
        
        # Calcular margin ratio (quanto mais baixo, mais seguro)
        margin_ratio = (maint_margin / margin_balance * 100) if margin_balance > 0 else 0
        margin_used_pct = (initial_margin / margin_balance * 100) if margin_balance > 0 else 0
        
        # Posicoes e PnL nao realizado
        all_positions = account.get('positions', [])
        open_positions = [p for p in all_positions if float(p.get('positionAmt', 0)) != 0]
        
        unrealized_pnl = float(account.get('totalUnrealizedProfit', 0))
        
        # Carregar historico de trades para PnL realizado e stats
        trade_history = load_json_safe('state/trade_history.json')
        if isinstance(trade_history, list) and len(trade_history) > 0:
            realized_pnl = sum(t.get('pnl', 0) for t in trade_history)
            total_trades = len(trade_history)
            wins = sum(1 for t in trade_history if t.get('pnl', 0) > 0)
            losses = total_trades - wins
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
            
            # Profit factor
            gross_profit = sum(t.get('pnl', 0) for t in trade_history if t.get('pnl', 0) > 0)
            gross_loss = abs(sum(t.get('pnl', 0) for t in trade_history if t.get('pnl', 0) < 0))
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else gross_profit
            
            # Calcular taxas totais (separadas)
            total_commission = sum(t.get('commission', 0) for t in trade_history)
            total_funding = sum(t.get('funding_cost', 0) for t in trade_history)
            total_fees = total_commission + total_funding
        else:
            realized_pnl = 0
            total_trades = 0
            wins = 0
            losses = 0
            win_rate = 0
            profit_factor = 0
            total_fees = 0
            total_commission = 0
            total_funding = 0
        
        # Carregar estado local (com cache)
        trader = get_trader_state_cached()
        
        # PnL total = realizado + nao realizado
        total_pnl = realized_pnl + unrealized_pnl
        
        # Balance inicial
        initial_balance = trader.get('initial_balance', wallet_balance - realized_pnl)
        return_pct = (total_pnl / initial_balance * 100) if initial_balance > 0 else 0
        
        # Drawdown
        high_water_mark = max(margin_balance, initial_balance)
        drawdown_pct = ((high_water_mark - margin_balance) / high_water_mark * 100) if high_water_mark > 0 else 0
        
        # Timestamp de sincronização do cache (quando os dados foram buscados da Binance)
        sync_time = _binance_data_cache.get('last_sync', time.time())

        # Obter status do Kelly Criterion
        use_kelly = Config.get('trader.use_kelly_sizing', False)

        return jsonify({
            'timestamp': datetime.now().isoformat(),
            'data_sync_time': datetime.fromtimestamp(sync_time).isoformat(),
            'mode': 'TESTNET' if USE_TESTNET else 'PRODUCTION',
            'timeframe': PRIMARY_TIMEFRAME,
            'symbols_count': len(Config.get('symbols', SYMBOLS)),
            'use_kelly': use_kelly,
            # Balances
            'wallet_balance': round(wallet_balance, 2),
            'margin_balance': round(margin_balance, 2),
            'available_balance': round(available_balance, 2),
            'initial_balance': round(initial_balance, 2),
            'high_water_mark': round(high_water_mark, 2),
            # Margens
            'initial_margin': round(initial_margin, 2),
            'maint_margin': round(maint_margin, 2),
            'margin_ratio': round(margin_ratio, 2),
            'margin_used_pct': round(margin_used_pct, 2),
            # Status
            'is_halted': trader.get('is_halted', False),
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': round(win_rate, 1),
            'profit_factor': round(profit_factor, 2),
            # PnL
            'realized_pnl': round(realized_pnl, 2),
            'unrealized_pnl': round(unrealized_pnl, 2),
            'total_pnl': round(total_pnl, 2),
            # Taxas detalhadas
            'total_fees': round(total_fees, 2),
            'total_commission': round(total_commission, 2),
            'total_funding': round(total_funding, 2),
            'return_pct': round(return_pct, 2),
            'drawdown_pct': round(drawdown_pct, 2),
            'open_positions': len(open_positions),
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        })


@app.route('/api/positions')
def api_positions():
    """Posicoes abertas - com cache para evitar rate limits."""
    try:
        # Usar cache para evitar chamadas excessivas
        account = get_cached_account()
        if not account:
            raise Exception("Dados da conta indisponíveis")

        all_positions = account.get('positions', [])
        open_positions = [p for p in all_positions if float(p.get('positionAmt', 0)) != 0]

        # Buscar precos atuais (com cache)
        tickers = get_cached_tickers()
        
        positions = []
        total_unrealized_pnl = 0
        
        # Carregar estado local para pegar strategy e reason (com cache)
        trader = get_trader_state_cached()
        local_positions = trader.get('positions', {})
        params = trader.get('params', {})
        
        for p in open_positions:
            # Normalizar simbolo (BTCUSDT -> BTC/USDT)
            raw_sym = p.get('symbol', '')
            sym = raw_sym.replace('USDT', '/USDT')
            
            position_amt = float(p.get('positionAmt', 0))
            side = 'long' if position_amt > 0 else 'short'
            contracts = abs(position_amt)
            entry = float(p.get('entryPrice', 0) or 0)
            pnl = float(p.get('unrealizedProfit', 0) or 0)
            leverage = int(p.get('leverage', 1) or 1)
            liq_price = float(p.get('liquidationPrice', 0) or 0)
            notional = abs(float(p.get('notional', 0) or 0))
            
            # Pegar preco atual do ticker (CORRIGIDO: sym já é BTC/USDT, não adicionar :USDT)
            # Para CCXT futures, o formato é "BTC/USDT:USDT" ou apenas "BTC/USDT"
            ticker_sym = sym  # Já está no formato correto BTC/USDT
            mark = float(tickers.get(ticker_sym, tickers.get(sym + ':USDT', {})).get('last', 0) or entry)
            
            total_unrealized_pnl += pnl
            
            # Pegar info local se existir
            local = local_positions.get(sym, {})
            
            # Calcular distancia para SL e TP
            sl = local.get('stop_loss', 0)
            tp = local.get('take_profit', 0)
            
            if mark > 0:
                if side == 'long':
                    sl_distance = ((mark - sl) / mark * 100) if sl > 0 else 0
                    tp_distance = ((tp - mark) / mark * 100) if tp > 0 else 0
                else:
                    sl_distance = ((sl - mark) / mark * 100) if sl > 0 else 0
                    tp_distance = ((mark - tp) / mark * 100) if tp > 0 else 0
            else:
                sl_distance = 0
                tp_distance = 0
            
            pnl_pct = (pnl / notional * 100) if notional > 0 else 0
            
            positions.append({
                'symbol': sym,
                'side': side,
                'strategy': local.get('strategy', params.get('strategy', 'unknown')),
                'reason_entry': local.get('reason_entry', ''),
                'entry_price': round(entry, 6),
                'current_price': round(mark, 6),
                'quantity': round(contracts, 6),
                'notional': round(notional, 2),
                'stop_loss': round(sl, 6),
                'take_profit': round(tp, 6),
                'sl_distance': round(sl_distance, 2),
                'tp_distance': round(tp_distance, 2),
                'unrealized_pnl': round(pnl, 3),
                'pnl_pct': round(pnl_pct, 3),
                'leverage': leverage,
                'liquidation_price': round(liq_price, 6),
                'entry_time': local.get('entry_time', ''),
            })
        
        # Timestamp de sincronização (mesmo que /api/status)
        sync_time = _binance_data_cache.get('last_sync', time.time())

        return jsonify({
            'count': len(positions),
            'total_unrealized_pnl': round(total_unrealized_pnl, 3),
            'positions': positions,
            'data_sync_time': datetime.fromtimestamp(sync_time).isoformat(),
        })
    except Exception as e:
        return jsonify({
            'count': 0,
            'total_unrealized_pnl': 0,
            'positions': [],
            'error': str(e)
        })


@app.route('/api/strategies')
def api_strategies():
    """Estrategias validadas WFO - usando sistema de armazenamento."""
    try:
        storage = get_storage(reload=True)  # Sempre recarregar do arquivo

        # Sincronização automática: verificar e corrigir se necessário
        sync_status = storage.get_sync_status()
        if not sync_status.get('is_synced', True):
            logger.info("Sincronização automática: corrigindo inconsistências...")
            storage.force_sync_all()
            storage = get_storage(reload=True)  # Recarregar após sync

        data = storage.export_for_dashboard()

        # Carregar estado atual do trader para saber estratégia ativa (com cache)
        trader = get_trader_state_cached()
        params = trader.get('params', {})

        # Enriquecer dados para exibição
        enriched = []
        for s in data.get('strategies', []):
            metrics = s.get('metrics', {})
            wfo = s.get('wfo', {})
            status = s.get('status', {})

            # Calcular leverage seguro e retorno anual estimado (usando config)
            wfo_config = get_wfo_config()
            test_days = wfo_config.get('test_days', 10)

            max_dd = metrics.get('max_drawdown', 1) or 1
            # Safe leverage: min(config max, 50/drawdown)
            safe_lev = min(LEVERAGE_CAP, int(50 / max_dd)) if max_dd > 0 else LEVERAGE_CAP // 2
            avg_ret = metrics.get('avg_return', 0)

            # avg_ret é por fold (~test_days). Anualização com compounding:
            num_periods = 365 / test_days
            if avg_ret > -100:  # Evitar erro matemático
                annual = ((1 + avg_ret / 100) ** num_periods - 1) * 100
            else:
                annual = -100

            enriched.append({
                'id': s.get('id'),
                'name': s.get('name'),
                'strategy_type': s.get('strategy_type'),
                'symbols': s.get('symbols', []),
                'timeframe': s.get('timeframe'),
                'params': s.get('params', {}),
                # Métricas
                'avg_return': metrics.get('avg_return', 0),
                'min_return': metrics.get('min_return', 0),
                'max_return': metrics.get('max_return', 0),
                'std_return': metrics.get('std_return', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),
                'avg_sharpe': metrics.get('avg_sharpe', 0),
                'avg_sortino': metrics.get('avg_sortino', 0),
                'avg_profit_factor': metrics.get('avg_profit_factor', 0),
                'avg_win_rate': metrics.get('avg_win_rate', 0),
                'total_trades': metrics.get('total_trades', 0),
                # WFO
                'num_folds': wfo.get('num_folds', 0),
                'wfo_score': wfo.get('wfo_score', 0),
                'robustness': wfo.get('robustness', 0),
                'fold_results': wfo.get('fold_results', []),
                # Calculados
                'safe_leverage': safe_lev,
                'annual_expected': round(annual, 1),
                # Status
                'is_active': status.get('is_active', False),
                'created_at': status.get('created_at'),
                'last_used': status.get('last_used'),
                # Performance real
                'real_trades': s.get('real_performance', {}).get('trades', 0),
                'real_pnl': s.get('real_performance', {}).get('pnl', 0),
                'real_win_rate': s.get('real_performance', {}).get('win_rate', 0),
            })

        return jsonify({
            'active_strategy': params.get('strategy', 'N/A'),
            'active_params': params,
            'validated_strategies': enriched,
            'total_strategies': data.get('total_strategies', 0),
            'active_count': data.get('active_count', 0),
            'updated_at': data.get('updated_at')
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'active_strategy': 'N/A',
            'validated_strategies': []
        })


@app.route('/api/strategies/<strategy_id>')
def api_strategy_detail(strategy_id):
    """Detalhes de uma estratégia específica."""
    try:
        storage = get_storage(reload=True)  # Sempre recarregar do arquivo
        strategy = storage.get_strategy(strategy_id)

        if not strategy:
            return jsonify({'error': 'Estratégia não encontrada'}), 404

        # Retornar todos os detalhes
        return jsonify({
            'id': strategy.id,
            'name': strategy.name,
            'strategy_type': strategy.strategy_type,
            'symbols': strategy.symbols,
            'timeframe': strategy.timeframe,
            'params': strategy.params,
            'metrics': {
                'avg_return': strategy.avg_return_pct,
                'min_return': strategy.min_return_pct,
                'max_return': strategy.max_return_pct,
                'std_return': strategy.std_return_pct,
                'avg_drawdown': strategy.avg_drawdown_pct,
                'max_drawdown': strategy.max_drawdown_pct,
                'avg_sharpe': strategy.avg_sharpe,
                'avg_sortino': strategy.avg_sortino,
                'avg_profit_factor': strategy.avg_profit_factor,
                'avg_win_rate': strategy.avg_win_rate,
                'total_trades': strategy.total_trades,
            },
            'wfo': {
                'num_folds': strategy.num_folds,
                'wfo_score': strategy.wfo_score,
                'robustness': strategy.robustness_score,  # Padronizado
                'robustness_score': strategy.robustness_score,  # Alias para compatibilidade
                'fold_results': strategy.fold_results,
            },
            'status': {
                'is_active': strategy.is_active,
                'created_at': strategy.created_at,
                'validated_at': strategy.validated_at,
                'last_used': strategy.last_used,
            },
            'real_performance': {
                'trades': strategy.real_trades,
                'pnl': strategy.real_pnl,
                'win_rate': strategy.real_win_rate,
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/trades')
def api_trades():
    """Historico de trades com paginação."""
    trader = get_trader_state_cached()
    stats = trader.get('stats', {})

    # Parâmetros de paginação
    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', 50, type=int)
    limit = min(limit, 200)  # Máximo 200 por página

    # Ler historico de trades JSON
    all_trades = []
    try:
        if os.path.exists('state/trade_history.json'):
            with open('state/trade_history.json', 'r') as f:
                all_trades = json.load(f)
    except Exception as e:
        logger.warning(f"Erro ao carregar trade_history.json: {e}")

    total_trades = len(all_trades)
    total_pages = (total_trades + limit - 1) // limit if total_trades > 0 else 1

    # Paginação (mais recentes primeiro)
    all_trades.reverse()
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    trades = all_trades[start_idx:end_idx]

    # Calcular estatisticas por estrategia (sobre todos os trades)
    strategy_stats = {}
    for t in all_trades:
        strat = t.get('strategy', 'unknown')
        if strat not in strategy_stats:
            strategy_stats[strat] = {'trades': 0, 'wins': 0, 'pnl': 0}
        strategy_stats[strat]['trades'] += 1
        strategy_stats[strat]['pnl'] += t.get('pnl', 0)
        if t.get('pnl', 0) > 0:
            strategy_stats[strat]['wins'] += 1

    # Calcular win rate por estrategia
    for strat, data in strategy_stats.items():
        data['win_rate'] = (data['wins'] / data['trades'] * 100) if data['trades'] > 0 else 0

    return jsonify({
        'stats': stats,
        'recent_trades': trades,
        'strategy_stats': strategy_stats,
        'pagination': {
            'page': page,
            'limit': limit,
            'total_trades': total_trades,
            'total_pages': total_pages,
            'has_next': page < total_pages,
            'has_prev': page > 1
        }
    })


@app.route('/api/logs')
def api_logs():
    """Ultimos logs do bot."""
    logs = []
    try:
        if os.path.exists('logs/bot.log'):
            with open('logs/bot.log', 'r', encoding='utf-8') as f:
                lines = f.readlines()
                logs = [l.strip() for l in lines[-100:]]
    except Exception as e:
        logger.warning(f"Erro ao ler logs/bot.log: {e}")

    return jsonify({'logs': logs})


@app.route('/api/equity-history')
def api_equity_history():
    """
    Historico de equity (balance) ao longo do tempo.
    Reconstroi a curva de equity a partir do trade_history.
    """
    try:
        # Carregar trade history
        trade_history = load_json_safe('state/trade_history.json')
        if not isinstance(trade_history, list):
            trade_history = []

        # Carregar balance inicial do trader_state
        trader = get_trader_state_cached()
        initial_balance = trader.get('initial_balance', 10000)

        # Se nao houver trades, retornar apenas o ponto inicial
        if len(trade_history) == 0:
            now = datetime.now().isoformat()
            return jsonify({
                'equity_curve': [{'timestamp': now, 'equity': initial_balance, 'drawdown': 0}],
                'initial_balance': initial_balance,
                'current_balance': initial_balance,
                'max_drawdown': 0,
                'total_return_pct': 0
            })

        # Ordenar trades por exit_time
        sorted_trades = sorted(trade_history, key=lambda x: x.get('exit_time', ''))

        # Construir curva de equity
        equity_curve = []
        running_balance = initial_balance
        high_water_mark = initial_balance
        max_drawdown = 0

        # Ponto inicial
        first_trade_time = sorted_trades[0].get('exit_time', datetime.now().isoformat())
        equity_curve.append({
            'timestamp': first_trade_time,
            'equity': initial_balance,
            'drawdown': 0,
            'trade_id': 0
        })

        # Adicionar cada trade
        for i, trade in enumerate(sorted_trades):
            pnl = trade.get('pnl', 0)
            running_balance += pnl
            high_water_mark = max(high_water_mark, running_balance)

            # Calcular drawdown atual
            if high_water_mark > 0:
                current_dd = (high_water_mark - running_balance) / high_water_mark * 100
            else:
                current_dd = 0

            max_drawdown = max(max_drawdown, current_dd)

            equity_curve.append({
                'timestamp': trade.get('exit_time', ''),
                'equity': round(running_balance, 2),
                'drawdown': round(current_dd, 2),
                'trade_id': trade.get('id', i + 1),
                'symbol': trade.get('symbol', ''),
                'pnl': round(pnl, 2)
            })

        # Calcular retorno total
        total_return_pct = ((running_balance - initial_balance) / initial_balance * 100) if initial_balance > 0 else 0

        return jsonify({
            'equity_curve': equity_curve,
            'initial_balance': initial_balance,
            'current_balance': round(running_balance, 2),
            'high_water_mark': round(high_water_mark, 2),
            'max_drawdown': round(max_drawdown, 2),
            'total_return_pct': round(total_return_pct, 2),
            'total_trades': len(sorted_trades)
        })

    except Exception as e:
        logger.error(f"Erro em /api/equity-history: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/pnl-by-symbol')
def api_pnl_by_symbol():
    """
    PnL agregado por simbolo.
    Retorna dados para grafico de barras ou pie chart.
    """
    try:
        # Carregar trade history
        trade_history = load_json_safe('state/trade_history.json')
        if not isinstance(trade_history, list):
            trade_history = []

        # Agregar por simbolo
        symbol_stats = {}
        for trade in trade_history:
            symbol = trade.get('symbol', 'UNKNOWN')
            pnl = trade.get('pnl', 0)

            if symbol not in symbol_stats:
                symbol_stats[symbol] = {
                    'symbol': symbol,
                    'total_pnl': 0,
                    'trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'gross_profit': 0,
                    'gross_loss': 0
                }

            symbol_stats[symbol]['total_pnl'] += pnl
            symbol_stats[symbol]['trades'] += 1

            if pnl > 0:
                symbol_stats[symbol]['wins'] += 1
                symbol_stats[symbol]['gross_profit'] += pnl
            else:
                symbol_stats[symbol]['losses'] += 1
                symbol_stats[symbol]['gross_loss'] += abs(pnl)

        # Calcular metricas adicionais
        result = []
        for sym, stats in symbol_stats.items():
            win_rate = (stats['wins'] / stats['trades'] * 100) if stats['trades'] > 0 else 0
            profit_factor = (stats['gross_profit'] / stats['gross_loss']) if stats['gross_loss'] > 0 else stats['gross_profit']

            result.append({
                'symbol': sym,
                'total_pnl': round(stats['total_pnl'], 2),
                'trades': stats['trades'],
                'wins': stats['wins'],
                'losses': stats['losses'],
                'win_rate': round(win_rate, 1),
                'profit_factor': round(profit_factor, 2),
                'avg_pnl': round(stats['total_pnl'] / stats['trades'], 2) if stats['trades'] > 0 else 0
            })

        # Ordenar por PnL total (maior primeiro)
        result.sort(key=lambda x: x['total_pnl'], reverse=True)

        return jsonify({
            'symbols': result,
            'total_symbols': len(result),
            'best_symbol': result[0] if result else None,
            'worst_symbol': result[-1] if result else None
        })

    except Exception as e:
        logger.error(f"Erro em /api/pnl-by-symbol: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/pnl-distribution')
def api_pnl_distribution():
    """
    Distribuicao de PnL para histograma.
    Retorna bins e frequencias.
    """
    try:
        # Carregar trade history
        trade_history = load_json_safe('state/trade_history.json')
        if not isinstance(trade_history, list) or len(trade_history) == 0:
            return jsonify({
                'distribution': [],
                'stats': {
                    'mean': 0,
                    'median': 0,
                    'std': 0,
                    'min': 0,
                    'max': 0,
                    'total_trades': 0
                }
            })

        # Extrair PnLs
        pnls = [t.get('pnl', 0) for t in trade_history]

        # Calcular estatisticas
        import statistics
        mean_pnl = statistics.mean(pnls)
        median_pnl = statistics.median(pnls)
        std_pnl = statistics.stdev(pnls) if len(pnls) > 1 else 0
        min_pnl = min(pnls)
        max_pnl = max(pnls)

        # Criar bins para histograma
        num_bins = min(20, max(5, len(pnls) // 5))
        bin_width = (max_pnl - min_pnl) / num_bins if max_pnl != min_pnl else 1

        bins = []
        for i in range(num_bins):
            bin_start = min_pnl + i * bin_width
            bin_end = bin_start + bin_width
            count = sum(1 for p in pnls if bin_start <= p < bin_end)
            if i == num_bins - 1:  # Ultimo bin inclui o max
                count = sum(1 for p in pnls if bin_start <= p <= bin_end)

            bins.append({
                'bin_start': round(bin_start, 2),
                'bin_end': round(bin_end, 2),
                'count': count,
                'label': f"${bin_start:.0f} - ${bin_end:.0f}"
            })

        return jsonify({
            'distribution': bins,
            'pnls': [round(p, 2) for p in pnls],  # Lista completa para Plotly
            'stats': {
                'mean': round(mean_pnl, 2),
                'median': round(median_pnl, 2),
                'std': round(std_pnl, 2),
                'min': round(min_pnl, 2),
                'max': round(max_pnl, 2),
                'total_trades': len(pnls),
                'positive_trades': sum(1 for p in pnls if p > 0),
                'negative_trades': sum(1 for p in pnls if p < 0)
            }
        })

    except Exception as e:
        logger.error(f"Erro em /api/pnl-distribution: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/signals')
def api_signals():
    """
    Sinais MTF em tempo real.
    Le do arquivo state/mtf_signals.json que o bot atualiza a cada ciclo.
    """
    try:
        # Ler sinais MTF do arquivo
        signals_data = load_json_safe('state/mtf_signals.json')
        if not signals_data:
            signals_data = {'signals': [], 'last_update': None}

        # Usar lista de simbolos do Config (dinamico)
        all_symbols = Config.get('symbols', SYMBOLS)

        return jsonify({
            'signals': signals_data.get('signals', [])[:15],  # Top 15 sinais
            'total_signals': len(signals_data.get('signals', [])),
            'last_update': signals_data.get('last_update'),
            'monitored_symbols': len(Config.get('symbols', SYMBOLS)),
            'active_timeframes': Config.get('timeframes.active_timeframes', ['5m', '15m', '1h']),
            'mtf_stats': signals_data.get('stats', {})
        })
    except Exception as e:
        logger.error(f"Erro em /api/signals: {e}")
        return jsonify({'error': str(e), 'signals': []})


@app.route('/api/bot-status')
def api_bot_status():
    """
    Status detalhado do bot (running, PID, uptime, scanning info).
    """
    try:
        bot_lock_file = 'state/bot.lock'
        bot_running = os.path.exists(bot_lock_file)

        response = {
            'running': bot_running,
            'pid': None,
            'uptime_seconds': 0,
            'start_time': None,
            'mode': 'TESTNET' if USE_TESTNET else 'PRODUCTION',
            'symbols_count': len(Config.get('symbols', SYMBOLS)),
            'timeframe': PRIMARY_TIMEFRAME,
            'active_timeframes': Config.get('timeframes.active_timeframes', ['5m', '15m', '1h']),
            'mtf_enabled': True,
            'kelly_enabled': Config.get('trader.use_kelly_sizing', False),
            'scanning_symbols': []
        }

        if bot_running:
            try:
                with open(bot_lock_file, 'r') as f:
                    bot_pid = f.read().strip()
                    response['pid'] = bot_pid

                # Tentar obter uptime do arquivo (se existir)
                lock_stat = os.stat(bot_lock_file)
                start_time = datetime.fromtimestamp(lock_stat.st_mtime)
                uptime = (datetime.now() - start_time).total_seconds()
                response['start_time'] = start_time.isoformat()
                response['uptime_seconds'] = int(uptime)
                response['uptime_formatted'] = f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m {int(uptime % 60)}s"
            except Exception:
                pass

        # Adicionar ultimos simbolos escaneados (do log)
        try:
            if os.path.exists('logs/bot.log'):
                with open('logs/bot.log', 'r', encoding='utf-8') as f:
                    lines = f.readlines()[-50:]
                    scanning = []
                    for line in reversed(lines):
                        if 'MTF SINAL' in line:
                            # Extrair simbolo do log
                            parts = line.split('MTF SINAL')
                            if len(parts) > 1:
                                sym_part = parts[1].strip().split()[0]
                                if sym_part not in scanning:
                                    scanning.append(sym_part)
                                if len(scanning) >= 5:
                                    break
                    response['scanning_symbols'] = scanning
        except Exception:
            pass

        return jsonify(response)
    except Exception as e:
        logger.error(f"Erro em /api/bot-status: {e}")
        return jsonify({'error': str(e), 'running': False})


@app.route('/api/metrics')
def api_metrics():
    """Metricas de performance - com cache."""
    try:
        # Usar cache
        account = get_cached_account()
        if not account:
            raise Exception("Dados da conta indisponíveis")

        balance = float(account.get('totalMarginBalance', 0))
        
        # PnL nao realizado
        unrealized_pnl = float(account.get('totalUnrealizedProfit', 0))
        
        # PnL realizado
        trade_history = load_json_safe('state/trade_history.json')
        if isinstance(trade_history, list):
            realized_pnl = sum(t.get('pnl', 0) for t in trade_history)
        else:
            realized_pnl = 0
        
        total_pnl = realized_pnl + unrealized_pnl
        
        # Stats locais (com cache)
        trader = get_trader_state_cached()
        stats = trader.get('stats', {})
        initial = trader.get('initial_balance', balance - total_pnl)
        hwm = max(balance, initial)
        
        return_pct = (total_pnl / initial * 100) if initial > 0 else 0
        drawdown = ((hwm - balance) / hwm * 100) if hwm > 0 else 0
        
        return jsonify({
            'balance': round(balance, 2),
            'initial_balance': round(initial, 2),
            'high_water_mark': round(hwm, 2),
            'return_pct': round(return_pct, 2),
            'drawdown_pct': round(drawdown, 2),
            'total_trades': stats.get('total_trades', 0),
            'wins': stats.get('wins', 0),
            'losses': stats.get('losses', 0),
            'win_rate': stats.get('win_rate', 0),
            'profit_factor': stats.get('profit_factor', 0),
            'avg_win': stats.get('avg_win', 0),
            'avg_loss': stats.get('avg_loss', 0),
            'realized_pnl': round(realized_pnl, 2),
            'unrealized_pnl': round(unrealized_pnl, 2),
            'total_pnl': round(total_pnl, 2),
        })
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    # Criar pasta templates se nao existir
    os.makedirs('templates', exist_ok=True)
    
    print("=" * 60)
    print("  DASHBOARD DE TRADING")
    print("  Acesse: http://localhost:5000")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
