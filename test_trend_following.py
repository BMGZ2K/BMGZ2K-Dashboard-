"""
Script de teste para TREND FOLLOWING STRATEGY

Demonstra uso da estrategia e realiza backtest basico.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from core.trend_following_strategy import TrendFollowingStrategy, backtest_trend_following

def generate_sample_data(days=90, timeframe='1h'):
    """Gerar dados de exemplo para teste."""
    # Gerar timestamps
    if timeframe == '1h':
        periods = days * 24
        freq = '1H'
    elif timeframe == '4h':
        periods = days * 6
        freq = '4H'
    else:
        periods = days
        freq = 'D'

    dates = pd.date_range(end=datetime.now(), periods=periods, freq=freq)

    # Simular precos com trend
    base_price = 50000
    trend = np.linspace(0, 10000, periods)  # Trend ascendente
    noise = np.random.randn(periods) * 500

    # Ciclos de trend
    cycle = np.sin(np.linspace(0, 4*np.pi, periods)) * 3000

    close = base_price + trend + cycle + noise

    # OHLC
    high = close + np.abs(np.random.randn(periods) * 200)
    low = close - np.abs(np.random.randn(periods) * 200)
    open_price = close + np.random.randn(periods) * 100

    # Volume
    volume = np.random.uniform(100, 1000, periods)

    df = pd.DataFrame({
        'timestamp': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })

    return df


def test_basic_usage():
    """Teste 1: Uso basico da estrategia."""
    print("=" * 80)
    print("TESTE 1: USO BASICO")
    print("=" * 80)
    print()

    # Gerar dados
    df = generate_sample_data(days=30, timeframe='1h')
    df_htf = generate_sample_data(days=30, timeframe='4h')

    # Criar estrategia com parametros default (otimizados)
    strategy = TrendFollowingStrategy()

    # Preparar dados
    df = strategy.calculate_indicators(df)

    # Gerar sinal na ultima vela
    signal = strategy.generate_signal(df, df_htf)

    print(f"Sinal Gerado:")
    print(f"  Direcao: {signal.direction.upper()}")
    print(f"  Forca: {signal.strength:.1f}/10")
    print(f"  Confianca: {signal.confidence.upper()}")
    print(f"  Preco Entrada: ${signal.entry_price:,.2f}")

    if signal.direction != 'none':
        print(f"  Stop Loss: ${signal.stop_loss:,.2f}")
        print(f"  Take Profit: ${signal.take_profit:,.2f}")
        print(f"  Trailing Stop: {signal.trailing_stop_pct*100:.0f}%")

        # Calcular R:R
        if signal.direction == 'long':
            risk = signal.entry_price - signal.stop_loss
            reward = signal.take_profit - signal.entry_price
        else:
            risk = signal.stop_loss - signal.entry_price
            reward = signal.entry_price - signal.take_profit

        rr_ratio = reward / risk if risk > 0 else 0
        print(f"  Risk/Reward: 1:{rr_ratio:.2f}")
        print()
        print(f"  Razao: {signal.reason}")
    else:
        print(f"  Razao: {signal.reason}")

    print()


def test_parameter_variations():
    """Teste 2: Testar variacoes de parametros."""
    print("=" * 80)
    print("TESTE 2: VARIACOES DE PARAMETROS")
    print("=" * 80)
    print()

    # Gerar dados
    df = generate_sample_data(days=90, timeframe='1h')

    # Configuracoes para testar
    configs = [
        {
            'name': 'DEFAULT (12/26, ADX 30)',
            'params': {}  # Usa defaults
        },
        {
            'name': 'FAST (9/21, ADX 25)',
            'params': {
                'ema_fast': 9,
                'ema_slow': 21,
                'adx_min': 25,
                'trailing_stop_pct': 0.15  # 15% (mais apertado)
            }
        },
        {
            'name': 'SLOW (20/50, ADX 35)',
            'params': {
                'ema_fast': 20,
                'ema_slow': 50,
                'adx_min': 35,
                'trailing_stop_pct': 0.25  # 25% (mais largo)
            }
        },
        {
            'name': 'AGGRESSIVE (12/26, ADX 25, RR 3:1)',
            'params': {
                'adx_min': 25,
                'sl_atr_mult': 2.0,
                'tp_atr_mult': 6.0,
                'trailing_stop_pct': 0.15
            }
        },
        {
            'name': 'CONSERVATIVE (12/26, ADX 40, RR 2:1)',
            'params': {
                'adx_min': 40,
                'sl_atr_mult': 3.0,
                'tp_atr_mult': 6.0,
                'trailing_stop_pct': 0.25
            }
        }
    ]

    print("Comparando configuracoes diferentes:\n")

    for config in configs:
        print(f"{config['name']}:")

        # Backtest
        result = backtest_trend_following(
            df=df.copy(),
            params=config['params'],
            initial_capital=10000
        )

        print(f"  Retorno Total: {result['total_return_pct']:+.2f}%")
        print(f"  Total de Trades: {result['total_trades']}")
        print(f"  Win Rate: {result['win_rate']*100:.1f}%")
        print(f"  Profit Factor: {result['profit_factor']:.2f}")
        print(f"  Max Drawdown: {result['max_drawdown_pct']:.2f}%")

        if result['avg_win'] > 0 and result['avg_loss'] > 0:
            rr = result['avg_win'] / result['avg_loss']
            print(f"  Avg Win/Loss: ${result['avg_win']:.2f} / ${result['avg_loss']:.2f} (R:R {rr:.2f})")

        print()

    print()


def test_realtime_simulation():
    """Teste 3: Simulacao de uso em tempo real."""
    print("=" * 80)
    print("TESTE 3: SIMULACAO REALTIME")
    print("=" * 80)
    print()

    # Gerar dados
    df = generate_sample_data(days=30, timeframe='1h')
    df_htf = generate_sample_data(days=30, timeframe='4h')

    strategy = TrendFollowingStrategy()

    print("Simulando 10 ultimas velas:\n")

    position = None

    for i in range(-10, 0):
        df_slice = df.iloc[:len(df)+i]

        # Preparar dados
        df_slice = strategy.calculate_indicators(df_slice)

        # Gerar sinal
        signal = strategy.generate_signal(df_slice, df_htf, position)

        timestamp = df_slice['timestamp'].values[-1]
        price = df_slice['close'].values[-1]

        print(f"Vela {i+10}/10 - {pd.Timestamp(timestamp).strftime('%Y-%m-%d %H:%M')} - ${price:,.0f}")

        # Abrir posicao
        if position is None and signal.direction in ['long', 'short']:
            if signal.strength >= 6:
                position = {
                    'side': signal.direction,
                    'entry': signal.entry_price,
                    'sl': signal.stop_loss,
                    'tp': signal.take_profit,
                    'amt': 1 if signal.direction == 'long' else -1
                }
                print(f"  >>> ABRIR {signal.direction.upper()} @ ${signal.entry_price:,.0f}")
                print(f"      SL: ${signal.stop_loss:,.0f} | TP: ${signal.take_profit:,.0f}")
                print(f"      Razao: {signal.reason}")

        # Fechar posicao
        elif position is not None:
            # Verificar SL/TP
            closed = False

            if position['side'] == 'long':
                if price <= position['sl']:
                    pnl = price - position['entry']
                    print(f"  <<< FECHAR LONG (SL) @ ${price:,.0f} | PnL: ${pnl:,.0f}")
                    position = None
                    closed = True
                elif price >= position['tp']:
                    pnl = price - position['entry']
                    print(f"  <<< FECHAR LONG (TP) @ ${price:,.0f} | PnL: ${pnl:,.0f}")
                    position = None
                    closed = True
            else:
                if price >= position['sl']:
                    pnl = position['entry'] - price
                    print(f"  <<< FECHAR SHORT (SL) @ ${price:,.0f} | PnL: ${pnl:,.0f}")
                    position = None
                    closed = True
                elif price <= position['tp']:
                    pnl = position['entry'] - price
                    print(f"  <<< FECHAR SHORT (TP) @ ${price:,.0f} | PnL: ${pnl:,.0f}")
                    position = None
                    closed = True

            # Verificar trailing
            if not closed and signal.direction != 'none':
                if (position['side'] == 'long' and signal.direction == 'short') or \
                   (position['side'] == 'short' and signal.direction == 'long'):
                    if position['side'] == 'long':
                        pnl = price - position['entry']
                    else:
                        pnl = position['entry'] - price
                    print(f"  <<< FECHAR {position['side'].upper()} (TRAILING) @ ${price:,.0f} | PnL: ${pnl:,.0f}")
                    print(f"      Razao: {signal.reason}")
                    position = None

            # Status da posicao
            if position is not None:
                if position['side'] == 'long':
                    pnl = price - position['entry']
                else:
                    pnl = position['entry'] - price
                print(f"  --- Em {position['side'].upper()} | PnL: ${pnl:,.0f}")

        print()

    print()


def display_strategy_info():
    """Exibir informacoes sobre a estrategia."""
    print()
    print("=" * 80)
    print("TREND FOLLOWING STRATEGY - Informacoes")
    print("=" * 80)
    print()

    print("FUNDAMENTOS DA ESTRATEGIA:")
    print()
    print("1. ENTRADA:")
    print("   - EMA rapida cruza acima/abaixo da EMA lenta")
    print("   - ADX confirma trend forte (>30)")
    print("   - Preco alinhado com EMA 200 (filtro de trend geral)")
    print("   - [OPCIONAL] Higher timeframe confirma direcao")
    print("   - [OPCIONAL] Volume acima da media")
    print()

    print("2. SAIDA:")
    print("   - Stop Loss fixo: 2.5 ATR")
    print("   - Take Profit fixo: 6.0 ATR (R:R ~2.4:1)")
    print("   - Trailing Stop: 20% do pico/fundo (ativa apos 1.5R de lucro)")
    print()

    print("3. FILTROS:")
    print("   - RSI extremos (evita overbought/oversold)")
    print("   - DI+/DI- (confirma direcao do trend)")
    print("   - ADX pico (evita reversoes em ADX >50)")
    print()

    print("PARAMETROS OTIMIZADOS (baseados em pesquisa):")
    print()
    print("  EMA Fast: 12")
    print("    - Balanco entre reatividade e reducao de ruido")
    print("    - Ref: https://altfins.com/knowledge-base/ema-12-50-crossovers/")
    print()
    print("  EMA Slow: 26")
    print("    - Filtra ruido, capta trends de medio prazo")
    print()
    print("  ADX Threshold: 30")
    print("    - Ideal para crypto devido a alta volatilidade")
    print("    - Ref: https://www.mindmathmoney.com/articles/adx-indicator-trading-strategy")
    print()
    print("  Trailing Stop: 20%")
    print("    - Maximiza lucros em trends fortes")
    print("    - Ref: https://www.altrady.com/crypto-trading/technical-analysis/risk-management-trend-following-strategies")
    print()

    print("PERFORMANCE ESPERADA:")
    print()
    print("  Win Rate: 35-45%")
    print("    - Normal para trend following (poucas entradas, mas de qualidade)")
    print()
    print("  Risk/Reward: >2.5:1")
    print("    - Compensa win rate mais baixo")
    print("    - Ganhos grandes compensam perdas pequenas")
    print()
    print("  Melhor em: Mercados trending (bull ou bear claro)")
    print("  Pior em: Mercados laterais (choppy, baixa direcionalidade)")
    print()
    print("=" * 80)
    print()


if __name__ == '__main__':
    # Exibir informacoes
    display_strategy_info()

    # Rodar testes
    test_basic_usage()
    test_parameter_variations()
    test_realtime_simulation()

    print("=" * 80)
    print("TESTES CONCLUIDOS")
    print("=" * 80)
    print()
    print("Proximos passos:")
    print()
    print("1. Testar com dados reais da Binance:")
    print("   from core.data import fetch_ohlcv")
    print("   df = fetch_ohlcv('BTCUSDT', '1h', days=90)")
    print("   df_htf = fetch_ohlcv('BTCUSDT', '4h', days=90)")
    print()
    print("2. Integrar com portfolio_wfo.py para validacao Walk-Forward")
    print()
    print("3. Adicionar ao auto_evolve.py para otimizacao genetica")
    print()
    print("4. Deploy no bot.py para trading ao vivo")
    print()
