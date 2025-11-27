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
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, jsonify
import ccxt

from core.config import (
    API_KEY, SECRET_KEY, USE_TESTNET, SYMBOLS, PRIMARY_TIMEFRAME,
    LEVERAGE_CAP, get_wfo_config
)
from core.evolution import get_storage
from core.utils import load_json_safe

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# =============================================================================
# CACHE UNIFICADO - Todos os dados sincronizados
# =============================================================================
# TTL padrão para todos os caches (garante consistência)
UNIFIED_CACHE_TTL = 2.0  # 2 segundos

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
EXCHANGE_CACHE_TTL = 60  # 60 segundos - conexão é estável

# Cache de dados da Binance - UNIFICADO para garantir consistência
# Todos os endpoints usam os mesmos dados quando dentro do TTL
_binance_data_cache = {
    'account': {'data': None, 'timestamp': 0},
    'tickers': {'data': None, 'timestamp': 0},
    'last_sync': 0,  # Timestamp da última sincronização completa
}


def get_cached_account():
    """Retorna dados da conta com cache."""
    global _binance_data_cache
    now = time.time()
    cache = _binance_data_cache['account']

    # Usar cache se ainda válido
    if cache['data'] and (now - cache['timestamp']) < UNIFIED_CACHE_TTL:
        return cache['data']

    try:
        exchange = get_exchange()
        account = exchange.fapiPrivateV2GetAccount()
        cache['data'] = account
        cache['timestamp'] = now
        _binance_data_cache['last_sync'] = now
        return account
    except Exception as e:
        logger.error(f"Erro ao buscar conta: {e}")
        return cache['data']  # Retorna cache antigo se falhar


def get_cached_tickers():
    """
    Retorna tickers/preços com cache.
    Usa fapiPublicGetPremiumIndex que é muito mais rápido que fetch_tickers.
    """
    global _binance_data_cache
    now = time.time()
    cache = _binance_data_cache['tickers']

    # Usar cache se ainda válido
    if cache['data'] and (now - cache['timestamp']) < UNIFIED_CACHE_TTL:
        return cache['data']

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

        cache['data'] = tickers
        cache['timestamp'] = now
        return tickers
    except Exception as e:
        logger.error(f"Erro ao buscar tickers: {e}")
        return cache['data'] or {}


def get_exchange():
    """Criar conexao com exchange (com cache)."""
    global _exchange_cache
    now = time.time()

    # Reusar conexão existente se ainda válida
    if _exchange_cache['instance'] and (now - _exchange_cache['timestamp']) < EXCHANGE_CACHE_TTL:
        return _exchange_cache['instance']

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
    """Retorna trader_state com cache unificado."""
    global _trader_state_cache
    now = time.time()
    if now - _trader_state_cache['timestamp'] > UNIFIED_CACHE_TTL:
        _trader_state_cache['data'] = load_json_safe('state/trader_state.json')
        _trader_state_cache['timestamp'] = now
    return _trader_state_cache['data']


@app.route('/')
def index():
    """Pagina principal."""
    return render_template('dashboard.html')


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

        return jsonify({
            'timestamp': datetime.now().isoformat(),
            'data_sync_time': datetime.fromtimestamp(sync_time).isoformat(),
            'mode': 'TESTNET' if USE_TESTNET else 'PRODUCTION',
            'timeframe': PRIMARY_TIMEFRAME,
            'symbols_count': len(SYMBOLS),
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
                'unrealized_pnl': round(pnl, 2),
                'pnl_pct': round(pnl_pct, 2),
                'leverage': leverage,
                'liquidation_price': round(liq_price, 6),
                'entry_time': local.get('entry_time', ''),
            })
        
        # Timestamp de sincronização (mesmo que /api/status)
        sync_time = _binance_data_cache.get('last_sync', time.time())

        return jsonify({
            'count': len(positions),
            'total_unrealized_pnl': round(total_unrealized_pnl, 2),
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
    """Historico de trades."""
    trader = get_trader_state_cached()
    stats = trader.get('stats', {})
    
    # Ler historico de trades JSON
    trades = []
    try:
        if os.path.exists('state/trade_history.json'):
            with open('state/trade_history.json', 'r') as f:
                trades = json.load(f)
                trades = trades[-50:]  # Ultimos 50
                trades.reverse()  # Mais recentes primeiro
    except Exception as e:
        log.warning(f"Erro ao carregar trade_history.json: {e}")
    
    # Calcular estatisticas por estrategia
    strategy_stats = {}
    for t in trades:
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
        'strategy_stats': strategy_stats
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
        log.warning(f"Erro ao ler logs/bot.log: {e}")

    return jsonify({'logs': logs})


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
