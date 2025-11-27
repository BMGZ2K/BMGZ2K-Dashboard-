"""
Backtest - Testar estrategia com parametros otimizados
Usa mesma logica do auto_optimizer para consistencia
"""
import sys
import os
import time
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import ccxt

from core.config import API_KEY, SECRET_KEY, USE_TESTNET, PRIMARY_TIMEFRAME


def fetch_data(exchange, symbol: str, timeframe: str, days: int = 60) -> pd.DataFrame:
    """Baixar dados historicos."""
    print(f"Baixando {symbol} {timeframe} ({days} dias)...")
    
    all_data = []
    end_time = int(time.time() * 1000)
    start_time = end_time - (days * 24 * 60 * 60 * 1000)
    current_end = end_time
    
    while current_end > start_time:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=1000, params={'endTime': current_end})
            if not ohlcv:
                break
            all_data.extend(ohlcv)
            current_end = ohlcv[0][0] - 1
            time.sleep(0.1)
        except Exception as e:
            print(f"Erro: {e}")
            break
    
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df = df.drop_duplicates().sort_values('timestamp').reset_index(drop=True)
    return df


def calculate_indicators(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Calcular indicadores."""
    df = df.copy()
    
    # RSI
    period = params.get('rsi_period', 14)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    
    # ATR
    atr_period = params.get('atr_period', 14)
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift()),
        abs(df['low'] - df['close'].shift())
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(atr_period).mean()
    
    # EMAs
    df['ema_fast'] = df['close'].ewm(span=params.get('ema_fast', 9)).mean()
    df['ema_slow'] = df['close'].ewm(span=params.get('ema_slow', 21)).mean()
    
    # Bollinger
    bb_period = params.get('bb_period', 20)
    df['bb_mid'] = df['close'].rolling(bb_period).mean()
    bb_std = df['close'].rolling(bb_period).std()
    df['bb_upper'] = df['bb_mid'] + bb_std * 2
    df['bb_lower'] = df['bb_mid'] - bb_std * 2
    
    # ADX simplificado
    plus_dm = df['high'].diff().clip(lower=0)
    minus_dm = (-df['low'].diff()).clip(lower=0)
    plus_di = 100 * plus_dm.ewm(span=14).mean() / (df['atr'] + 1e-10)
    minus_di = 100 * minus_dm.ewm(span=14).mean() / (df['atr'] + 1e-10)
    df['adx'] = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100).ewm(span=14).mean()
    
    return df


def run_backtest(df: pd.DataFrame, params: dict, initial_balance: float = 10000) -> dict:
    """Executar backtest com mesma logica do optimizer."""
    df = calculate_indicators(df, params)
    
    strategy = params.get('strategy', 'rsi')
    rsi_oversold = params.get('rsi_oversold', 30)
    rsi_overbought = params.get('rsi_overbought', 70)
    sl_mult = params.get('sl_mult', 2.0)
    tp_mult = params.get('tp_mult', 3.0)
    risk_pct = params.get('risk_pct', 0.01)
    adx_threshold = params.get('adx_threshold', 20)
    
    balance = initial_balance
    position = None
    trades = []
    equity_curve = [balance]
    
    for i in range(50, len(df)):
        row = df.iloc[i]
        price = row['close']
        atr = row['atr']
        
        if pd.isna(atr) or atr == 0:
            continue
        
        # Gerenciar posicao
        if position:
            exit_price = None
            reason = None
            
            if position['side'] == 'long':
                if row['low'] <= position['sl']:
                    exit_price = position['sl']
                    reason = 'SL'
                elif row['high'] >= position['tp']:
                    exit_price = position['tp']
                    reason = 'TP'
            else:
                if row['high'] >= position['sl']:
                    exit_price = position['sl']
                    reason = 'SL'
                elif row['low'] <= position['tp']:
                    exit_price = position['tp']
                    reason = 'TP'
            
            if exit_price:
                if position['side'] == 'long':
                    pnl = (exit_price - position['entry']) * position['qty']
                else:
                    pnl = (position['entry'] - exit_price) * position['qty']
                
                comm = (position['entry'] + exit_price) * position['qty'] * 0.0004
                balance += pnl - comm
                trades.append({'pnl': pnl - comm, 'reason': reason})
                position = None
            
            equity_curve.append(balance)
            continue
        
        # Sinais de entrada
        signal = None
        rsi = row['rsi']
        adx = row['adx']
        ema_f = row['ema_fast']
        ema_s = row['ema_slow']
        bb_upper = row['bb_upper']
        bb_lower = row['bb_lower']
        
        if pd.isna(rsi):
            continue
        
        if strategy == 'rsi':
            if rsi < rsi_oversold:
                signal = 'long'
            elif rsi > rsi_overbought:
                signal = 'short'
                
        elif strategy == 'trend':
            if adx > adx_threshold:
                if ema_f > ema_s and rsi > 50:
                    signal = 'long'
                elif ema_f < ema_s and rsi < 50:
                    signal = 'short'
                    
        elif strategy == 'bb':
            if price < bb_lower and rsi < 35:
                signal = 'long'
            elif price > bb_upper and rsi > 65:
                signal = 'short'
                
        elif strategy == 'combined':
            if rsi < rsi_oversold and ema_f > ema_s:
                signal = 'long'
            elif rsi > rsi_overbought and ema_f < ema_s:
                signal = 'short'
        
        if signal:
            if signal == 'long':
                sl = price - atr * sl_mult
                tp = price + atr * tp_mult
            else:
                sl = price + atr * sl_mult
                tp = price - atr * tp_mult
            
            stop_dist = abs(price - sl)
            qty = (balance * risk_pct) / stop_dist if stop_dist > 0 else 0
            
            if qty * price > 10:
                position = {
                    'side': signal,
                    'entry': price,
                    'qty': qty,
                    'sl': sl,
                    'tp': tp
                }
        
        equity_curve.append(balance)
    
    # Metricas
    if not trades:
        return {'trades': 0, 'win_rate': 0, 'profit_factor': 0, 'return_pct': 0, 'max_drawdown': 0, 'final_balance': balance}
    
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] < 0]
    
    gross_profit = sum(t['pnl'] for t in wins) if wins else 0
    gross_loss = abs(sum(t['pnl'] for t in losses)) if losses else 0
    
    # Drawdown
    peak = equity_curve[0]
    max_dd = 0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)
    
    return {
        'trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': len(wins) / len(trades) * 100,
        'profit_factor': gross_profit / gross_loss if gross_loss > 0 else float('inf'),
        'avg_win': gross_profit / len(wins) if wins else 0,
        'avg_loss': gross_loss / len(losses) if losses else 0,
        'return_pct': (balance - initial_balance) / initial_balance * 100,
        'max_drawdown': max_dd * 100,
        'final_balance': balance
    }


def main():
    parser = argparse.ArgumentParser(description='Backtest')
    parser.add_argument('--symbol', default='BTC/USDT', help='Par de trading')
    parser.add_argument('--timeframe', default=None, help='Timeframe')
    parser.add_argument('--days', type=int, default=60, help='Dias de dados')
    parser.add_argument('--balance', type=float, default=10000, help='Balance inicial')
    args = parser.parse_args()
    
    print("=" * 60)
    print("  BACKTEST")
    print("=" * 60)
    
    # Conectar
    exchange = ccxt.binance({
        'apiKey': API_KEY,
        'secret': SECRET_KEY,
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })
    if USE_TESTNET:
        exchange.set_sandbox_mode(True)
        print("Modo: TESTNET\n")
    
    # Carregar parametros
    params = {}
    timeframe = args.timeframe or PRIMARY_TIMEFRAME
    
    if os.path.exists('state/optimized_params.json'):
        try:
            with open('state/optimized_params.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                params = data.get('params', {})
                # Usar timeframe do arquivo se nao especificado
                if not args.timeframe and 'optimization' in data:
                    timeframe = data['optimization'].get('timeframe', timeframe)
                print(f"Parametros carregados: {data.get('best_strategy', 'unknown')}")
        except Exception as e:
            print(f"Aviso: {e}")
    
    if not params:
        params = {
            'strategy': 'trend',
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'sl_mult': 2.0,
            'tp_mult': 3.0,
            'risk_pct': 0.01,
            'adx_threshold': 25
        }
    
    print(f"Estrategia: {params.get('strategy', 'unknown')}")
    print(f"Timeframe: {timeframe}")
    print(f"SL: {params.get('sl_mult')}x ATR | TP: {params.get('tp_mult')}x ATR")
    print()
    
    # Baixar dados
    df = fetch_data(exchange, args.symbol, timeframe, args.days)
    print(f"Total: {len(df)} candles\n")
    
    # Buy & Hold
    bh = (df.iloc[-1]['close'] - df.iloc[0]['close']) / df.iloc[0]['close'] * 100
    print(f"Buy & Hold: {bh:.1f}%\n")
    
    # Executar backtest
    print("Executando backtest...")
    results = run_backtest(df, params, args.balance)
    
    # Resultados
    print("\n" + "=" * 60)
    print("  RESULTADOS")
    print("=" * 60)
    print(f"Trades:        {results['trades']}")
    print(f"Wins:          {results['wins']}")
    print(f"Losses:        {results['losses']}")
    print(f"Win Rate:      {results['win_rate']:.1f}%")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    print(f"Avg Win:       ${results['avg_win']:.2f}")
    print(f"Avg Loss:      ${results['avg_loss']:.2f}")
    print(f"Max Drawdown:  {results['max_drawdown']:.1f}%")
    print(f"Retorno:       {results['return_pct']:.1f}%")
    print(f"Saldo Final:   ${results['final_balance']:.2f}")
    
    alpha = results['return_pct'] - bh
    print(f"\nBuy & Hold:    {bh:.1f}%")
    print(f"Alpha:         {alpha:.1f}%")
    
    # Validacao
    print("\n" + "-" * 60)
    if results['profit_factor'] >= 1.5 and alpha > 10:
        print("EXCELENTE - PF >= 1.5 e Alpha > 10%")
    elif results['profit_factor'] >= 1.0 and alpha > 0:
        print("APROVADO - Estrategia lucrativa com alpha positivo")
    elif alpha > 0:
        print("PARCIAL - Alpha positivo mas PF < 1.0")
    else:
        print("REPROVADO - Verificar parametros")
    
    # Salvar resultados
    os.makedirs('results', exist_ok=True)
    with open('results/last_backtest.json', 'w', encoding='utf-8') as f:
        json.dump({
            'symbol': args.symbol,
            'timeframe': timeframe,
            'days': args.days,
            'params': params,
            'results': results,
            'buy_hold': bh,
            'alpha': alpha
        }, f, indent=2)


if __name__ == "__main__":
    main()
