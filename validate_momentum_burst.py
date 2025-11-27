"""
VALIDAÇÃO E OTIMIZAÇÃO - Momentum Burst Strategy
Testa a estratégia com dados reais e otimiza parâmetros
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ccxt
from momentum_burst_strategy import (
    MomentumBurstStrategy,
    backtest_momentum_burst,
    optimize_parameters
)


def download_crypto_data(
    symbol: str = 'BTC/USDT',
    timeframe: str = '1h',
    days: int = 180
) -> pd.DataFrame:
    """
    Download dados reais de crypto da Binance.

    Args:
        symbol: Par de trading
        timeframe: Timeframe dos candles
        days: Dias de histórico

    Returns:
        DataFrame com dados OHLCV
    """
    print(f"\nBaixando dados de {symbol} ({timeframe})...")

    try:
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })

        # Calculate time range
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = end_time - (days * 24 * 60 * 60 * 1000)

        all_data = []
        current_time = start_time

        while current_time < end_time:
            try:
                ohlcv = exchange.fetch_ohlcv(
                    symbol,
                    timeframe,
                    since=current_time,
                    limit=1000
                )

                if not ohlcv:
                    break

                all_data.extend(ohlcv)
                current_time = ohlcv[-1][0] + 1

                print(f"Baixados {len(all_data)} candles...", end='\r')

            except Exception as e:
                print(f"\nErro ao baixar dados: {e}")
                break

        if not all_data:
            print("Nenhum dado baixado!")
            return pd.DataFrame()

        df = pd.DataFrame(
            all_data,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)

        print(f"\nDados baixados: {len(df)} candles")
        print(f"Período: {df['timestamp'].min()} a {df['timestamp'].max()}")

        return df

    except Exception as e:
        print(f"Erro ao conectar exchange: {e}")
        return pd.DataFrame()


def comprehensive_backtest(
    df: pd.DataFrame,
    strategy_params: dict = None
) -> dict:
    """
    Realiza backtest completo com análise detalhada.

    Args:
        df: DataFrame com dados
        strategy_params: Parâmetros da estratégia

    Returns:
        Dicionário com resultados completos
    """
    print("\n" + "=" * 80)
    print("EXECUTANDO BACKTEST COMPLETO")
    print("=" * 80)

    results = backtest_momentum_burst(
        df,
        params=strategy_params,
        initial_capital=10000.0,
        position_size=0.95,
        fees=0.0006
    )

    # Additional analysis
    if results['total_trades'] > 0:
        trades = results['trades']

        # Calculate additional metrics
        returns = trades['pnl_pct'].values
        wins = trades[trades['pnl'] > 0]
        losses = trades[trades['pnl'] < 0]

        # Sharpe Ratio (simplified)
        if len(returns) > 1:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe = 0

        # Max Drawdown
        equity = results['equity_curve']['equity'].values
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak * 100
        max_dd = np.min(drawdown)

        # Consecutive wins/losses
        streak = []
        current_streak = 0
        for _, trade in trades.iterrows():
            if trade['pnl'] > 0:
                current_streak = current_streak + 1 if current_streak > 0 else 1
            else:
                current_streak = current_streak - 1 if current_streak < 0 else -1
            streak.append(current_streak)

        max_win_streak = max([s for s in streak if s > 0], default=0)
        max_loss_streak = abs(min([s for s in streak if s < 0], default=0))

        # Average trade duration
        if 'exit_time' in trades.columns and 'entry_time' in trades.columns:
            trades['duration'] = (trades['exit_time'] - trades['entry_time']).dt.total_seconds() / 3600
            avg_duration = trades['duration'].mean()
        else:
            avg_duration = 0

        # Risk-Reward Ratio
        rr_ratio = abs(results['avg_win'] / results['avg_loss']) if results['avg_loss'] != 0 else 0

        results.update({
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'max_win_streak': max_win_streak,
            'max_loss_streak': max_loss_streak,
            'avg_trade_duration_hours': avg_duration,
            'risk_reward_ratio': rr_ratio,
            'total_wins': len(wins),
            'total_losses': len(losses)
        })

    return results


def print_detailed_results(results: dict):
    """Imprime resultados detalhados do backtest."""
    print("\n" + "=" * 80)
    print("RESULTADOS DO BACKTEST")
    print("=" * 80)

    print(f"\nPERFORMANCE GERAL:")
    print(f"  Total de Trades: {results['total_trades']}")
    print(f"  Trades Vencedores: {results.get('total_wins', 0)}")
    print(f"  Trades Perdedores: {results.get('total_losses', 0)}")
    print(f"  Win Rate: {results['win_rate']:.2f}%")
    print(f"  Profit Factor: {results['profit_factor']:.2f}")
    print(f"  Risk-Reward Ratio: {results.get('risk_reward_ratio', 0):.2f}")

    print(f"\nRETORNOS:")
    print(f"  Retorno Total: {results['total_return']:.2f}%")
    print(f"  Capital Inicial: $10,000.00")
    print(f"  Capital Final: ${results['final_capital']:.2f}")
    print(f"  Lucro: ${results['final_capital'] - 10000:.2f}")

    print(f"\nMÉTRICAS DE TRADES:")
    print(f"  Média de Ganho: {results['avg_win']:.2f}%")
    print(f"  Média de Perda: {results['avg_loss']:.2f}%")
    print(f"  Duração Média: {results.get('avg_trade_duration_hours', 0):.1f}h")

    print(f"\nRISCO:")
    print(f"  Max Drawdown: {results.get('max_drawdown', 0):.2f}%")
    print(f"  Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
    print(f"  Max Win Streak: {results.get('max_win_streak', 0)}")
    print(f"  Max Loss Streak: {results.get('max_loss_streak', 0)}")

    # Quality Assessment
    print(f"\n" + "=" * 80)
    print("AVALIAÇÃO DE QUALIDADE")
    print("=" * 80)

    score = 0
    max_score = 0

    # Profit Factor > 2.0 (OBJETIVO PRINCIPAL)
    max_score += 30
    if results['profit_factor'] >= 2.5:
        score += 30
        print("  Profit Factor: EXCELENTE (>2.5) +30 pts")
    elif results['profit_factor'] >= 2.0:
        score += 25
        print("  Profit Factor: ÓTIMO (>2.0) +25 pts")
    elif results['profit_factor'] >= 1.5:
        score += 15
        print("  Profit Factor: BOM (>1.5) +15 pts")
    else:
        print("  Profit Factor: INSUFICIENTE (<1.5) +0 pts")

    # Win Rate > 50%
    max_score += 20
    if results['win_rate'] >= 60:
        score += 20
        print("  Win Rate: EXCELENTE (>60%) +20 pts")
    elif results['win_rate'] >= 50:
        score += 15
        print("  Win Rate: BOM (>50%) +15 pts")
    else:
        print("  Win Rate: BAIXO (<50%) +0 pts")

    # Total Return
    max_score += 20
    if results['total_return'] >= 100:
        score += 20
        print("  Retorno Total: EXCELENTE (>100%) +20 pts")
    elif results['total_return'] >= 50:
        score += 15
        print("  Retorno Total: BOM (>50%) +15 pts")
    elif results['total_return'] >= 20:
        score += 10
        print("  Retorno Total: MODERADO (>20%) +10 pts")
    else:
        print("  Retorno Total: BAIXO (<20%) +0 pts")

    # Max Drawdown
    max_score += 15
    max_dd = abs(results.get('max_drawdown', 100))
    if max_dd < 10:
        score += 15
        print("  Max Drawdown: EXCELENTE (<10%) +15 pts")
    elif max_dd < 20:
        score += 10
        print("  Max Drawdown: BOM (<20%) +10 pts")
    elif max_dd < 30:
        score += 5
        print("  Max Drawdown: MODERADO (<30%) +5 pts")
    else:
        print("  Max Drawdown: ALTO (>30%) +0 pts")

    # Number of Trades (liquidez)
    max_score += 15
    if results['total_trades'] >= 50:
        score += 15
        print("  Quantidade de Trades: EXCELENTE (>50) +15 pts")
    elif results['total_trades'] >= 30:
        score += 10
        print("  Quantidade de Trades: BOM (>30) +10 pts")
    elif results['total_trades'] >= 20:
        score += 5
        print("  Quantidade de Trades: MODERADO (>20) +5 pts")
    else:
        print("  Quantidade de Trades: BAIXO (<20) +0 pts")

    final_score = (score / max_score) * 100

    print(f"\n" + "=" * 80)
    print(f"SCORE FINAL: {final_score:.1f}/100")
    print("=" * 80)

    if final_score >= 80:
        print("AVALIAÇÃO: EXCELENTE - Estratégia pronta para produção!")
    elif final_score >= 60:
        print("AVALIAÇÃO: BOM - Estratégia sólida com possíveis melhorias")
    elif final_score >= 40:
        print("AVALIAÇÃO: MODERADO - Requer otimização adicional")
    else:
        print("AVALIAÇÃO: INSUFICIENTE - Precisa de ajustes significativos")

    print("=" * 80)


def main():
    """Função principal de validação."""
    print("=" * 80)
    print("MOMENTUM BURST STRATEGY - VALIDAÇÃO COM DADOS REAIS")
    print("=" * 80)

    # 1. Download dados
    df = download_crypto_data(
        symbol='BTC/USDT',
        timeframe='1h',
        days=180
    )

    if df.empty:
        print("Erro ao baixar dados. Abortando.")
        return

    # 2. Test com parâmetros padrão
    print("\n" + "=" * 80)
    print("TESTE 1: PARÂMETROS PADRÃO")
    print("=" * 80)

    results_default = comprehensive_backtest(df)
    print_detailed_results(results_default)

    # 3. Optimization
    print("\n" + "=" * 80)
    print("TESTE 2: OTIMIZAÇÃO DE PARÂMETROS")
    print("=" * 80)

    param_grid = {
        'atr_multiplier_entry': [2.0, 2.5, 3.0],
        'adx_threshold': [20, 25, 30],
        'volume_multiplier': [1.3, 1.5, 2.0],
        'sl_atr_multiplier': [1.0, 1.2, 1.5],
        'tp_atr_multiplier': [3.0, 3.5, 4.0, 4.5]
    }

    print("\nBuscando melhores parâmetros...")
    print("Grid de busca:")
    for key, values in param_grid.items():
        print(f"  {key}: {values}")

    optimization = optimize_parameters(df, param_grid)

    if optimization['best_params']:
        print("\n" + "=" * 80)
        print("MELHORES PARÂMETROS ENCONTRADOS:")
        print("=" * 80)
        for key, value in optimization['best_params'].items():
            print(f"  {key}: {value}")

        print(f"\nProfit Factor Otimizado: {optimization['best_profit_factor']:.2f}")

        if optimization['best_results']:
            print_detailed_results(optimization['best_results'])

            # Save optimized parameters
            print("\nSalvando parâmetros otimizados...")
            params_df = pd.DataFrame([optimization['best_params']])
            params_df.to_csv('momentum_burst_optimized_params.csv', index=False)
            print("Parâmetros salvos em: momentum_burst_optimized_params.csv")

    else:
        print("\nNenhuma combinação atingiu os critérios mínimos:")
        print("  - Profit Factor > 2.0")
        print("  - Win Rate > 50%")
        print("  - Trades > 20")

    # 4. Test múltiplos ativos
    print("\n" + "=" * 80)
    print("TESTE 3: VALIDAÇÃO EM MÚLTIPLOS ATIVOS")
    print("=" * 80)

    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    multi_results = []

    for symbol in symbols:
        print(f"\nTestando {symbol}...")
        df_symbol = download_crypto_data(symbol, '1h', 90)

        if not df_symbol.empty:
            results = comprehensive_backtest(df_symbol, optimization.get('best_params'))
            multi_results.append({
                'symbol': symbol,
                'trades': results['total_trades'],
                'win_rate': results['win_rate'],
                'profit_factor': results['profit_factor'],
                'return': results['total_return']
            })

    if multi_results:
        print("\n" + "=" * 80)
        print("RESULTADOS MULTI-ATIVO")
        print("=" * 80)

        multi_df = pd.DataFrame(multi_results)
        print(multi_df.to_string(index=False))

        print(f"\nMÉDIAS:")
        print(f"  Win Rate médio: {multi_df['win_rate'].mean():.2f}%")
        print(f"  Profit Factor médio: {multi_df['profit_factor'].mean():.2f}")
        print(f"  Retorno médio: {multi_df['return'].mean():.2f}%")

    print("\n" + "=" * 80)
    print("VALIDAÇÃO COMPLETA!")
    print("=" * 80)


if __name__ == '__main__':
    main()
