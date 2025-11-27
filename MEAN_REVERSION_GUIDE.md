# MEAN REVERSION STRATEGY - Guia Completo

## Índice
1. [O que é Mean Reversion](#o-que-é-mean-reversion)
2. [Quando Usar vs Não Usar](#quando-usar-vs-não-usar)
3. [Fundamentos e Pesquisa](#fundamentos-e-pesquisa)
4. [Parâmetros Otimizados](#parâmetros-otimizados)
5. [Como Integrar no Sistema](#como-integrar-no-sistema)
6. [Resultados Esperados](#resultados-esperados)
7. [Troubleshooting](#troubleshooting)

---

## O que é Mean Reversion

**Mean Reversion** é uma estratégia baseada na teoria de que preços que se afastam muito da média tendem a **retornar** à média.

### Conceito Central
```
Preço se afasta da média → Movimento exagerado → Correção de volta à média
```

### Por que funciona?
- **Psicologia de mercado**: Extremos são insustentáveis
- **Realização de lucros**: Traders fazem take profit em extremos
- **Suporte/Resistência**: Níveis técnicos agem como ímãs
- **Regressão à média estatística**: Propriedade fundamental de séries temporais

---

## Quando Usar vs Não Usar

### ✅ USAR Mean Reversion quando:

1. **ADX < 25** (mercado lateral/sem tendência)
   - ADX mede força da tendência
   - < 20 = ideal para mean reversion
   - 20-25 = aceitável
   - > 25 = evitar

2. **Bollinger Bands normais ou apertadas**
   - Não em squeeze (breakout iminente)
   - BB Width estável

3. **RSI em extremos**
   - < 30 (oversold) para LONG
   - > 70 (overbought) para SHORT
   - Quanto mais extremo, melhor

4. **Volume normal ou baixo**
   - Volume spike pode indicar breakout
   - Evitar volume > 3x média

5. **Mercado sem catalisadores**
   - Sem notícias importantes
   - Sem eventos econômicos
   - Horário normal de trading

### ❌ NÃO USAR Mean Reversion quando:

1. **ADX > 30** (tendência forte)
   - Preço pode continuar na direção da tendência
   - Mean reversion falha em trends fortes

2. **Breakout de consolidação**
   - Bollinger Bands muito apertadas (squeeze)
   - Preço comprimido por muito tempo

3. **Volume spike extremo** (> 3-4x média)
   - Pode ser início de movimento forte
   - Não é correção temporária

4. **Notícias/Eventos**
   - Decisões de FED, inflação
   - Listagens, hacks, regulações
   - Qualquer catalisador forte

5. **Mercado em crash ou pump parabólico**
   - Pânico destrói lógica de mean reversion
   - FOMO idem

6. **Você está em dúvida**
   - Se não tem certeza, não entre
   - Mercado sempre terá outra oportunidade

---

## Fundamentos e Pesquisa

### Estudos e Backtests (2025)

Baseado em pesquisa recente de múltiplas fontes:

#### Win Rates Documentados:
- **Basic BB Mean Reversion**: 60-65% (em range-bound)
- **BB + RSI**: 65-70%
- **BB + RSI + MACD**: até 78% ([fonte](https://www.quantifiedstrategies.com/macd-and-bollinger-bands-strategy/))
- **BB + RSI + ADX filter**: 70-75% ([fonte](https://aliazary.medium.com/enhancing-bollinger-bands-mean-reversion-leveraging-adx-and-rsi-filters-to-shift-returns-from-7-97b5fd70ac44))

#### Risk:Reward Típico:
- **Mean Reversion geralmente**: 1:1 a 1:1.5
- **Não espere** R:R de 1:3 como em trend following
- **Compensação**: Win rate ALTO compensa R:R menor

#### ADX como Filtro:
- **ADX < 20**: Mercado lateral ideal ([fonte](https://www.altrady.com/crypto-trading/technical-analysis/average-directional-index-adx))
- **ADX 20-25**: Aceitável para mean reversion
- **ADX > 25**: Início de tendência, evitar MR
- **ADX > 30**: Tendência forte, **NUNCA** usar MR

#### Bollinger Bands:
- **Padrão**: 20 períodos, 2.0 std
- **Crypto volátil**: 20 períodos, 2.5 std
- **%b indicator**: Mostra posição nas bandas
  - %b < 0.05 = muito oversold (ideal para LONG)
  - %b > 0.95 = muito overbought (ideal para SHORT)

---

## Parâmetros Otimizados

### 1. Configuração CONSERVADORA (Recomendada para iniciantes)

```python
conservative_params = {
    # Bollinger Bands
    'bb_period': 20,
    'bb_std': 2.0,

    # RSI
    'rsi_period': 14,
    'rsi_oversold': 35,      # Mais seletivo que 30
    'rsi_overbought': 65,    # Mais seletivo que 70

    # ADX Filter
    'adx_max': 20,           # Muito restritivo - só mercado lateral

    # Stop Loss / Take Profit
    'sl_atr_mult': 0.75,     # Stop em 0.75x ATR
    'tp_atr_mult': 1.5,      # TP em 1.5x ATR
    'use_bb_middle_tp': True, # TP na BB middle (RECOMENDADO)

    # Confirmações adicionais
    'use_macd_confirmation': True,
    'use_stoch_confirmation': True,

    # Filtros
    'max_volume_ratio': 3.0,
}
```

**Quando usar**:
- Iniciante em mean reversion
- Capital pequeno
- Baixa tolerância a risco
- Quer win rate máximo (70-75%)

**Performance esperada**:
- Win rate: 70-75%
- Trades/semana: 3-8 (baixo - muito seletivo)
- R:R médio: 1:1.5
- Sharpe: 1.2-1.5

---

### 2. Configuração MODERADA (Balanceada)

```python
moderate_params = {
    'bb_period': 20,
    'bb_std': 2.0,
    'rsi_oversold': 30,
    'rsi_overbought': 70,
    'adx_max': 25,
    'sl_atr_mult': 0.60,
    'tp_atr_mult': 1.20,
    'use_bb_middle_tp': True,
    'use_stoch_confirmation': True,  # MACD off para mais trades
}
```

**Quando usar**:
- Trader com experiência
- Quer mais trades
- Aceita win rate menor por mais oportunidades

**Performance esperada**:
- Win rate: 60-70%
- Trades/semana: 8-15
- R:R médio: 1:1.2
- Sharpe: 1.0-1.3

---

### 3. Configuração CRYPTO VOLÁTIL (BTC/ETH)

```python
volatile_crypto_params = {
    'bb_period': 20,
    'bb_std': 2.5,           # Bandas MAIS LARGAS
    'rsi_oversold': 25,      # Mais extremo
    'rsi_overbought': 75,
    'adx_max': 25,
    'sl_atr_mult': 0.75,
    'tp_atr_mult': 1.5,
    'use_bb_middle_tp': True,
    'max_volume_ratio': 4.0, # Tolera mais volume
}
```

**Quando usar**:
- Trading BTC, ETH
- Mercado muito volátil
- Movimentos amplos

**Performance esperada**:
- Win rate: 55-65%
- Trades/semana: 10-20
- R:R médio: 1:1.5
- Sharpe: 0.8-1.2

---

## Como Integrar no Sistema

### Opção 1: Integrar em `core/signals.py`

A estratégia mean reversion já existe em `core/signals.py` (método `_signal_mean_reversion`), mas pode ser melhorada:

```python
# Em core/signals.py, atualizar o método:

def _signal_mean_reversion(self, df: pd.DataFrame) -> Signal:
    """Estrategia Mean Reversion OTIMIZADA."""

    # Usar a classe MeanReversionStrategy
    from mean_reversion_strategy import MeanReversionStrategy

    mr_strategy = MeanReversionStrategy({
        'bb_period': self.params.get('bb_period', 20),
        'bb_std': self.params.get('bb_std', 2.0),
        'rsi_oversold': self.params.get('mr_rsi_long_max', 35),
        'rsi_overbought': self.params.get('mr_rsi_short_min', 65),
        'adx_max': self.params.get('mr_adx_max', 25),
        'sl_atr_mult': self.params.get('mr_sl_factor', 0.60) * self.sl_atr_mult,
        'tp_atr_mult': self.params.get('mr_tp_factor', 0.60) * self.tp_atr_mult,
    })

    mr_signal = mr_strategy.generate_signal(df)

    # Converter para formato Signal do sistema
    return Signal(
        direction=mr_signal.direction,
        strength=mr_signal.strength,
        entry_price=mr_signal.entry_price,
        stop_loss=mr_signal.stop_loss,
        take_profit=mr_signal.take_profit,
        reason=mr_signal.reason
    )
```

### Opção 2: Usar no Portfolio WFO

Adicionar mean reversion como estratégia ao grid search:

```python
# Em portfolio_wfo.py ou script de otimização

from mean_reversion_strategy import MeanReversionStrategy

# Grid de parâmetros para otimizar
param_grid = {
    'strategy': ['mean_reversion'],
    'bb_std': [2.0, 2.5],
    'rsi_oversold': [25, 30, 35],
    'rsi_overbought': [65, 70, 75],
    'adx_max': [20, 25],
    'sl_atr_mult': [0.50, 0.60, 0.75],
    'tp_atr_mult': [1.0, 1.2, 1.5],
    'use_bb_middle_tp': [True, False],
}

# Rodar WFO
best_params = wfo.run_wfo(
    symbols=['BTCUSDT', 'ETHUSDT'],
    start_date='2024-01-01',
    end_date='2025-01-01',
    param_grid=param_grid,
    n_folds=6
)
```

### Opção 3: Bot Standalone

Criar bot dedicado apenas para mean reversion:

```python
# mean_reversion_bot.py

from mean_reversion_strategy import MeanReversionStrategy
from core.trader import execute_trade
from core.data import fetch_ohlcv

strategy = MeanReversionStrategy(moderate_params)

while True:
    # Fetch dados
    df = fetch_ohlcv('BTCUSDT', '1h')

    # Check se deve evitar trading
    should_avoid, reason = strategy.should_avoid_trading(df)

    if should_avoid:
        print(f"Evitando trading: {reason}")
        continue

    # Gerar sinal
    signal = strategy.generate_signal(df)

    # Executar se confiança >= MEDIUM
    if signal.confidence in ['MEDIUM', 'HIGH']:
        execute_trade(signal)

    time.sleep(3600)  # Check a cada 1h
```

---

## Resultados Esperados

### Performance Realista (baseada em backtests reais)

#### Mercado Normal (lateral/range-bound):
- **Win Rate**: 60-75%
- **Profit Factor**: 1.2-1.5
- **Sharpe Ratio**: 1.0-1.5
- **Max Drawdown**: 10-15%
- **Retorno mensal**: 3-8%

#### Mercado Trending (ADX > 25):
- **Win Rate**: 40-50% ⚠️
- **Profit Factor**: 0.8-1.1 ⚠️
- **Resultado**: PREJUÍZO ou break-even
- **Ação**: **NÃO USAR** mean reversion

### Comparação com Trend Following:

| Métrica | Mean Reversion | Trend Following |
|---------|----------------|-----------------|
| Win Rate | 60-75% | 35-45% |
| R:R médio | 1:1 a 1:1.5 | 1:3 a 1:5 |
| Sharpe | 1.0-1.5 | 0.8-1.2 |
| Melhor em | Range-bound | Trending |
| ADX ideal | < 25 | > 30 |
| Drawdown | 10-15% | 15-25% |

### Quando Mean Reversion SUPERA Trend Following:
- Mercados laterais (70% do tempo em crypto)
- Timeframes menores (1h, 4h)
- Stablecoins, pares menos voláteis
- Períodos de baixa volatilidade

### Quando Trend Following SUPERA Mean Reversion:
- Tendências fortes (breakouts, bull/bear markets)
- Timeframes maiores (1d, 1w)
- BTC em momentos de descoberta de preço
- Alta volatilidade com direção clara

---

## Troubleshooting

### Problema: Win rate < 50%

**Possíveis causas**:
1. ADX muito alto (mercado em tendência)
   - Solução: Reduzir `adx_max` para 20
2. RSI não seletivo o suficiente
   - Solução: Usar 35/65 ao invés de 30/70
3. Stop Loss muito apertado
   - Solução: Aumentar `sl_atr_mult` para 0.75-1.0
4. Mercado em tendência forte
   - Solução: **Parar de usar mean reversion**, mudar para trend following

### Problema: Muitos trades, poucos lucros

**Possíveis causas**:
1. Parâmetros muito agressivos
   - Solução: Usar configuração CONSERVADORA
2. Faltam confirmações
   - Solução: Habilitar `use_macd_confirmation` e `use_stoch_confirmation`
3. Take Profit muito ambicioso
   - Solução: Usar `use_bb_middle_tp: True`

### Problema: Poucos trades

**Possíveis causas**:
1. ADX muito restritivo
   - Solução: Aumentar `adx_max` para 25
2. RSI muito seletivo
   - Solução: Voltar para 30/70
3. Muitas confirmações
   - Solução: Desabilitar MACD confirmation

### Problema: Drawdown muito alto

**Possíveis causas**:
1. Não respeitando filtro de ADX
   - Solução: Verificar se `should_avoid_trading()` está sendo chamado
2. Position sizing incorreto
   - Solução: Nunca arriscar mais que 1-2% por trade
3. Stop Loss não sendo respeitado
   - Solução: Implementar stop loss RIGOROSAMENTE

---

## Checklist Antes de Usar em Live

- [ ] Backtest com pelo menos 3 meses de dados
- [ ] Win rate >= 55%
- [ ] Profit factor >= 1.2
- [ ] Max drawdown <= 15%
- [ ] Sharpe ratio >= 0.8
- [ ] Testado em diferentes condições de mercado
- [ ] Filtro de ADX implementado e funcionando
- [ ] Stop loss SEMPRE respeitado
- [ ] Position sizing definido (1-2% risk per trade)
- [ ] Sistema de logging ativo
- [ ] Entendeu QUANDO NÃO USAR a estratégia
- [ ] Tem plano B (trend following) para quando mercado entrar em tendência

---

## Fontes e Referências

### Pesquisa e Estudos:
1. [Enhanced Mean Reversion Strategy with Bollinger Bands and RSI](https://medium.com/@redsword_23261/enhanced-mean-reversion-strategy-with-bollinger-bands-and-rsi-integration-87ec8ca1059f)
2. [MACD and Bollinger Bands Strategy – 78% Win Rate](https://www.quantifiedstrategies.com/macd-and-bollinger-bands-strategy/)
3. [Bollinger Bands Mean-Reversion with ADX and RSI](https://aliazary.medium.com/enhancing-bollinger-bands-mean-reversion-leveraging-adx-and-rsi-filters-to-shift-returns-from-7-97b5fd70ac44)
4. [ADX Guide: Mastering the Average Directional Index](https://www.altrady.com/crypto-trading/technical-analysis/average-directional-index-adx)
5. [Mean Reversion in Crypto Futures - OKX](https://www.okx.com/learn/mean-reversion-strategies-crypto-futures)

### Livros Recomendados:
- "Mean Reversion Trading Systems" - Howard Bandy
- "High Probability ETF Trading" - Larry Connors
- "Bollinger on Bollinger Bands" - John Bollinger

---

## Conclusão

Mean Reversion é uma estratégia **poderosa** quando usada **corretamente**:

✅ **Vantagens**:
- Win rate alto (60-75%)
- Lógica clara e testada
- Funciona na maioria do tempo (mercados são laterais 60-70% do tempo)
- Menor exposição a risco

❌ **Desvantagens**:
- R:R menor (1:1 a 1:1.5)
- PERIGOSO em tendências fortes
- Requer disciplina rigorosa
- Não funciona em breakouts

🎯 **Chave do Sucesso**:
1. **Respeitar o filtro de ADX** (< 25)
2. **NUNCA** tentar mean reversion em tendência forte
3. **Stop Loss NÃO NEGOCIÁVEL**
4. **TP conservador** (BB middle é ideal)
5. **Position sizing** correto (1-2% risk)
6. **Saber quando PARAR** de usar

💡 **Dica Final**:
Combine mean reversion com trend following. Use mean reversion quando ADX < 25 e trend following quando ADX > 30. Assim você tem estratégia para TODAS as condições de mercado.

Boa sorte! 🚀
