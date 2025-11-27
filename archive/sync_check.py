"""
VERIFICAR SINCRONIZACAO COM BINANCE
===================================
Detecta discrepancias entre estado local e exchange
"""
import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ccxt
from core.config import API_KEY, SECRET_KEY, USE_TESTNET

print("=" * 80)
print("  VERIFICACAO DE SINCRONIZACAO COM BINANCE")
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
    print("Modo: TESTNET")

exchange.load_time_difference()

# 1. DADOS DA BINANCE
print("\n" + "=" * 80)
print("  1. DADOS REAIS DA BINANCE")
print("=" * 80)

# Balance
balance = exchange.fetch_balance()
usdt_balance = float(balance['total'].get('USDT', 0))
usdt_free = float(balance['free'].get('USDT', 0))
print(f"\n  Balance USDT: ${usdt_balance:,.2f}")
print(f"  Free USDT: ${usdt_free:,.2f}")

# Posicoes reais
positions = exchange.fetch_positions()
open_positions = [p for p in positions if abs(float(p.get('contracts', 0))) > 0]

print(f"\n  Posicoes abertas na Binance: {len(open_positions)}")

binance_positions = {}
for p in open_positions:
    sym = p['symbol']
    side = p['side']
    contracts = abs(float(p.get('contracts', 0) or 0))
    entry = float(p.get('entryPrice', 0) or 0)
    mark = float(p.get('markPrice', 0) or 0)
    pnl = float(p.get('unrealizedPnl', 0) or 0)
    leverage = int(p.get('leverage', 1) or 1)
    liq_price = float(p.get('liquidationPrice', 0) or 0)
    
    binance_positions[sym] = {
        'side': side,
        'contracts': contracts,
        'entry_price': entry,
        'mark_price': mark,
        'unrealized_pnl': pnl,
        'leverage': leverage,
        'liquidation_price': liq_price
    }
    
    print(f"\n  {sym}:")
    print(f"    Side: {side}")
    print(f"    Contracts: {contracts}")
    print(f"    Entry: ${entry:,.6f}")
    print(f"    Mark: ${mark:,.6f}")
    print(f"    PnL: ${pnl:,.2f}")
    print(f"    Leverage: {leverage}x")
    if liq_price > 0:
        print(f"    Liquidation: ${liq_price:,.6f}")

# 2. DADOS LOCAIS
print("\n" + "=" * 80)
print("  2. DADOS LOCAIS (trader_state.json)")
print("=" * 80)

local_positions = {}
try:
    with open('state/trader_state.json', 'r') as f:
        state = json.load(f)
    
    local_balance = state.get('balance', 0)
    print(f"\n  Balance local: ${local_balance:,.2f}")
    
    for sym, pos in state.get('positions', {}).items():
        local_positions[sym] = pos
        print(f"\n  {sym}:")
        print(f"    Side: {pos.get('side')}")
        print(f"    Quantity: {pos.get('quantity')}")
        print(f"    Entry: ${pos.get('entry_price'):,.6f}")
except Exception as e:
    print(f"  Erro: {e}")

# 3. COMPARAR
print("\n" + "=" * 80)
print("  3. DISCREPANCIAS DETECTADAS")
print("=" * 80)

problems = []

# Balance
if abs(usdt_balance - local_balance) > 1:
    problems.append({
        'type': 'BALANCE',
        'issue': f'Balance diferente: Binance=${usdt_balance:.2f} vs Local=${local_balance:.2f}',
        'fix': 'Atualizar balance local'
    })

# Posicoes que existem na Binance mas nao localmente
for sym in binance_positions:
    if sym not in local_positions:
        problems.append({
            'type': 'MISSING_LOCAL',
            'issue': f'{sym} existe na Binance mas nao no estado local',
            'fix': 'Adicionar posicao ao estado local'
        })

# Posicoes que existem localmente mas nao na Binance
for sym in local_positions:
    if sym not in binance_positions:
        problems.append({
            'type': 'GHOST_POSITION',
            'issue': f'{sym} existe localmente mas NAO na Binance',
            'fix': 'Remover posicao fantasma do estado local'
        })

# Posicoes com dados diferentes
for sym in binance_positions:
    if sym in local_positions:
        binance = binance_positions[sym]
        local = local_positions[sym]
        
        # Verificar side
        binance_side = 'long' if binance['side'] == 'long' else 'short'
        if binance_side != local.get('side'):
            problems.append({
                'type': 'SIDE_MISMATCH',
                'issue': f'{sym} side diferente: Binance={binance_side} vs Local={local.get("side")}',
                'fix': 'Corrigir side'
            })
        
        # Verificar quantidade
        if abs(binance['contracts'] - local.get('quantity', 0)) > 0.0001:
            problems.append({
                'type': 'QTY_MISMATCH',
                'issue': f'{sym} qty diferente: Binance={binance["contracts"]} vs Local={local.get("quantity")}',
                'fix': 'Corrigir quantidade'
            })

if not problems:
    print("\n  [OK] Nenhuma discrepancia detectada!")
else:
    print(f"\n  {len(problems)} problemas encontrados:\n")
    for i, p in enumerate(problems, 1):
        print(f"  [{i}] {p['type']}: {p['issue']}")
        print(f"      Fix: {p['fix']}")

# 4. CORRIGIR AUTOMATICAMENTE
print("\n" + "=" * 80)
print("  4. CORRIGINDO AUTOMATICAMENTE...")
print("=" * 80)

# Criar novo estado sincronizado
new_state = {
    'balance': usdt_balance,
    'initial_balance': usdt_balance,
    'high_water_mark': usdt_balance,
    'is_halted': False,
    'positions': {},
    'stats': {
        'total_trades': state.get('stats', {}).get('total_trades', 0),
        'wins': state.get('stats', {}).get('wins', 0),
        'losses': state.get('stats', {}).get('losses', 0),
        'win_rate': state.get('stats', {}).get('win_rate', 0),
        'profit_factor': state.get('stats', {}).get('profit_factor', 0),
        'total_pnl': state.get('stats', {}).get('total_pnl', 0),
        'avg_win': state.get('stats', {}).get('avg_win', 0),
        'avg_loss': state.get('stats', {}).get('avg_loss', 0),
        'balance': usdt_balance,
        'return_pct': 0
    },
    'params': state.get('params', {}),
    'timestamp': __import__('datetime').datetime.now().isoformat()
}

# Adicionar posicoes reais da Binance
for sym, pos in binance_positions.items():
    new_state['positions'][sym] = {
        'symbol': sym,
        'side': 'long' if pos['side'] == 'long' else 'short',
        'entry_price': pos['entry_price'],
        'quantity': pos['contracts'],
        'stop_loss': 0,  # Precisa calcular
        'take_profit': 0,  # Precisa calcular
        'entry_time': __import__('datetime').datetime.now().isoformat(),
        'reason_entry': 'Synced from Binance',
        'strategy': state.get('params', {}).get('strategy', 'unknown'),
        'leverage': pos['leverage'],
        'mark_price': pos['mark_price'],
        'unrealized_pnl': pos['unrealized_pnl'],
        'liquidation_price': pos['liquidation_price']
    }

# Salvar
with open('state/trader_state.json', 'w') as f:
    json.dump(new_state, f, indent=2)

print("\n  [OK] Estado sincronizado com Binance!")
print(f"  Balance: ${usdt_balance:,.2f}")
print(f"  Posicoes: {len(binance_positions)}")

for sym, pos in binance_positions.items():
    print(f"    {sym}: {pos['side']} {pos['contracts']} @ ${pos['entry_price']:,.4f} | PnL: ${pos['unrealized_pnl']:,.2f}")

print("\n" + "=" * 80)
