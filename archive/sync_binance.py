"""
SINCRONIZACAO COMPLETA COM BINANCE
==================================
Sincroniza estado local com dados reais da exchange
"""
import sys
import os
import json
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ccxt
from core.config import API_KEY, SECRET_KEY, USE_TESTNET

print("=" * 80)
print("  SINCRONIZANDO COM BINANCE...")
print("=" * 80)

# Conectar
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
    print("Modo: TESTNET\n")

exchange.load_time_difference()

# Buscar dados reais
balance = exchange.fetch_balance()
usdt_balance = float(balance['total'].get('USDT', 0))
usdt_free = float(balance['free'].get('USDT', 0))

print(f"Balance Total: ${usdt_balance:,.2f}")
print(f"Balance Free: ${usdt_free:,.2f}")

# Buscar posicoes
positions = exchange.fetch_positions()
open_positions = [p for p in positions if abs(float(p.get('contracts', 0) or 0)) > 0]

print(f"\nPosicoes abertas: {len(open_positions)}")

# Carregar estado existente
try:
    with open('state/trader_state.json', 'r') as f:
        old_state = json.load(f)
except:
    old_state = {'params': {}, 'stats': {}}

# Criar novo estado
new_positions = {}
total_unrealized_pnl = 0

for p in open_positions:
    # Normalizar simbolo (remover :USDT do final)
    raw_sym = p['symbol']
    sym = raw_sym.replace(':USDT', '')
    
    side = p['side']
    contracts = abs(float(p.get('contracts', 0) or 0))
    entry = float(p.get('entryPrice', 0) or 0)
    mark = float(p.get('markPrice', 0) or 0)
    pnl = float(p.get('unrealizedPnl', 0) or 0)
    leverage = int(p.get('leverage', 1) or 1)
    liq_price = float(p.get('liquidationPrice', 0) or 0)
    notional = float(p.get('notional', 0) or 0)
    
    total_unrealized_pnl += pnl
    
    # Calcular SL e TP baseado em ATR (2x e 3x)
    atr_mult = 0.02  # ~2% aproximado
    if side == 'long':
        sl = entry * (1 - atr_mult * 2)
        tp = entry * (1 + atr_mult * 3)
    else:
        sl = entry * (1 + atr_mult * 2)
        tp = entry * (1 - atr_mult * 3)
    
    new_positions[sym] = {
        'symbol': sym,
        'side': side,
        'entry_price': entry,
        'current_price': mark,
        'quantity': contracts,
        'notional': abs(notional),
        'stop_loss': round(sl, 6),
        'take_profit': round(tp, 6),
        'unrealized_pnl': pnl,
        'pnl_pct': (pnl / abs(notional) * 100) if notional != 0 else 0,
        'leverage': leverage,
        'liquidation_price': liq_price,
        'entry_time': datetime.now().isoformat(),
        'reason_entry': 'Synced from Binance',
        'strategy': old_state.get('params', {}).get('strategy', 'stoch_extreme')
    }
    
    pnl_color = '+' if pnl >= 0 else ''
    print(f"\n  {sym}:")
    print(f"    {side.upper()} {contracts} @ ${entry:,.6f}")
    print(f"    Mark: ${mark:,.6f} | PnL: {pnl_color}${pnl:,.2f}")
    print(f"    Leverage: {leverage}x | Liq: ${liq_price:,.6f}")

print(f"\n  Total PnL nao realizado: ${total_unrealized_pnl:,.2f}")

# Salvar novo estado
new_state = {
    'balance': usdt_balance,
    'initial_balance': usdt_balance - total_unrealized_pnl,
    'high_water_mark': usdt_balance,
    'is_halted': False,
    'positions': new_positions,
    'stats': {
        'total_trades': old_state.get('stats', {}).get('total_trades', 0),
        'wins': old_state.get('stats', {}).get('wins', 0),
        'losses': old_state.get('stats', {}).get('losses', 0),
        'win_rate': old_state.get('stats', {}).get('win_rate', 0),
        'profit_factor': old_state.get('stats', {}).get('profit_factor', 0),
        'total_pnl': total_unrealized_pnl,
        'avg_win': old_state.get('stats', {}).get('avg_win', 0),
        'avg_loss': old_state.get('stats', {}).get('avg_loss', 0),
        'balance': usdt_balance,
        'return_pct': (total_unrealized_pnl / (usdt_balance - total_unrealized_pnl) * 100) if usdt_balance != total_unrealized_pnl else 0
    },
    'params': old_state.get('params', {
        'strategy': 'stoch_extreme',
        'rsi_oversold': 20,
        'rsi_overbought': 80,
        'sl_atr_mult': 2.0,
        'tp_atr_mult': 3.0,
        'risk_per_trade': 0.01
    }),
    'timestamp': datetime.now().isoformat()
}

with open('state/trader_state.json', 'w') as f:
    json.dump(new_state, f, indent=2)

print("\n" + "=" * 80)
print("  SINCRONIZACAO COMPLETA!")
print("=" * 80)
print(f"""
  Balance: ${usdt_balance:,.2f}
  Posicoes: {len(new_positions)}
  PnL Total: ${total_unrealized_pnl:,.2f}
  
  Estado salvo em: state/trader_state.json
""")
