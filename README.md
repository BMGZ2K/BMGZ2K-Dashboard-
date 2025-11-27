# Sistema de Trading Automatizado - Binance Futures

Sistema completo de trading algoritmico com Walk-Forward Optimization (WFO), evolucao genetica de estrategias e execucao automatizada.

## Estrutura do Projeto

```
/
├── core/                    # MODULOS PRINCIPAIS
│   ├── __init__.py         # Exports publicos
│   ├── signals.py          # Gerador de sinais e indicadores
│   ├── binance_fees.py     # Taxas dinamicas da Binance API
│   ├── scoring.py          # Sistema de pontuacao de sinais
│   ├── evolution.py        # Storage de estrategias validadas
│   ├── config.py           # Configuracoes (API, simbolos, parametros)
│   ├── data.py             # Busca de dados OHLCV
│   ├── metrics.py          # Metricas de performance
│   ├── risk.py             # Gerenciamento de risco
│   ├── strategy.py         # Engine de estrategias
│   ├── trader.py           # Execucao de trades
│   └── indicators.py       # Indicadores tecnicos (legacy)
│
├── tests/                   # TESTES
│   ├── __init__.py
│   └── test_system.py      # Suite completa de testes
│
├── templates/               # TEMPLATES HTML
│   └── dashboard.html      # Interface do dashboard
│
├── state/                   # ESTADO PERSISTENTE
│   ├── trader_state.json   # Estado atual do bot
│   ├── trade_history.json  # Historico de trades
│   └── validated_strategies.json
│
├── logs/                    # LOGS DO SISTEMA
├── results/                 # RESULTADOS DE BACKTEST
├── archive/                 # ARQUIVOS ANTIGOS (nao usar)
│
├── portfolio_wfo.py        # BACKTESTER PRINCIPAL (WFO)
├── auto_evolve.py          # SISTEMA DE EVOLUCAO
├── bot.py                  # BOT DE TRADING LIVE
├── dashboard.py            # DASHBOARD WEB
│
├── run_all.bat             # Iniciar sistema completo
├── run_evolve.bat          # Iniciar evolucao
├── run_dashboard.bat       # Iniciar dashboard
├── run_bot.bat             # Iniciar bot
└── requirements.txt        # Dependencias
```

---

## Arquivos Principais

### 1. portfolio_wfo.py - Backtester com Walk-Forward Optimization

**Funcao:** Backtester principal com validacao Walk-Forward.

**Classes:**
- `PortfolioBacktester` - Engine principal
- `PortfolioBacktestResult` - Resultado do backtest
- `BacktestTrade` - Representacao de um trade

**Uso:**
```python
from portfolio_wfo import PortfolioBacktester

wfo = PortfolioBacktester()

# Buscar dados
data = wfo.fetch_data(['BTCUSDT', 'ETHUSDT'], '1h', start, end)

# Backtest simples
result = wfo.run_backtest(data, params)

# Walk-Forward Optimization
best = wfo.run_wfo(symbols, start, end, param_grid, n_folds=6)
```

**Metricas retornadas:**
- `total_return_pct` - Retorno total %
- `annual_return_pct` - Retorno anualizado
- `sharpe_ratio` - Sharpe ratio
- `sortino_ratio` - Sortino ratio
- `max_drawdown_pct` - Drawdown maximo
- `profit_factor` - Profit factor
- `win_rate` - Taxa de acerto
- `total_trades` - Total de trades
- `equity_curve` - Curva de equity
- `trades` - Lista de trades

---

### 2. core/signals.py - Gerador de Sinais

**Funcao:** Gera sinais de entrada/saida com indicadores tecnicos.

**Indicadores implementados (precisao validada vs pandas_ta):**
| Indicador | Funcao | Metodo |
|-----------|--------|--------|
| RSI | `calculate_rsi()` | Wilder's smoothing |
| ATR | `calculate_atr()` | Wilder's smoothing |
| EMA | `calculate_ema()` | Standard EWM |
| ADX | `calculate_adx()` | Wilder's smoothing |
| Bollinger | `calculate_bollinger()` | Population std (ddof=0) |
| Stochastic | `calculate_stochastic()` | smooth_k=3 |
| MACD | `calculate_macd()` | Standard |

**Estrategias disponveis:**
1. `stoch_extreme` - Stochastic oversold/overbought + EMA cross
2. `rsi_extremes` - RSI oversold/overbought
3. `trend_following` - EMA cross + ADX filter
4. `mean_reversion` - Bollinger Bands bounce
5. `momentum_burst` - Movimentos fortes

**Uso:**
```python
from core.signals import SignalGenerator, Signal

gen = SignalGenerator({
    'strategy': 'stoch_extreme',
    'ema_fast': 9,
    'ema_slow': 21,
    'sl_atr_mult': 2.0,
    'tp_atr_mult': 3.0,
})

# Preparar dados (pre-calcula indicadores)
df = gen.prepare_data(df_ohlcv)

# Gerar sinal
signal = gen.generate_signal(df, precomputed=True)

# signal.direction: 'long', 'short', 'none'
# signal.strength: 0-10
# signal.entry_price, signal.stop_loss, signal.take_profit
```

---

### 3. core/binance_fees.py - Taxas Dinamicas

**Funcao:** Busca taxas em tempo real da Binance API.

**Dados obtidos:**
- Funding rate (a cada 8h)
- Maker/Taker fees
- Mark price / Index price
- Leverage brackets
- Liquidation price

**Uso:**
```python
from core.binance_fees import get_binance_fees

fees = get_binance_fees(use_testnet=False)

# Todas as taxas de um simbolo
info = fees.get_all_fees_for_symbol('BTCUSDT')
# info['maker_fee'], info['taker_fee'], info['funding_rate']

# Calcular preco de liquidacao
liq_price = fees.calculate_liquidation_price(
    symbol='BTCUSDT',
    side='long',
    entry_price=50000,
    quantity=0.1,
    wallet_balance=1000,
    leverage=5
)
```

**Cache:**
- Exchange info: 1 hora
- Funding rates: 1 minuto
- Avg funding: 6 horas

---

### 4. core/scoring.py - Sistema de Pontuacao

**Funcao:** Pontua sinais para priorizar execucao.

**Criterios:**
- Risk/Reward ratio
- Forca do sinal (0-10)
- Win rate historico
- Volatilidade (ATR)
- Contexto de mercado

**Uso:**
```python
from core.scoring import ScoringSystem

scoring = ScoringSystem()
score = scoring.calculate_score(
    symbol='BTCUSDT',
    side='long',
    strength=7.5,
    entry_price=50000,
    stop_loss=49000,
    take_profit=52000,
    atr=500,
    historical_winrate=0.55
)
# score.score: 0-15 (quanto maior, melhor)
# score.breakdown: detalhes por criterio
```

---

### 5. core/evolution.py - Storage de Estrategias

**Funcao:** Armazena e gerencia estrategias validadas por WFO.

**Uso:**
```python
from core.evolution import get_storage, ValidatedStrategy

storage = get_storage()

# Salvar estrategia validada
storage.save_strategy(strategy)

# Buscar melhor estrategia
best = storage.get_best_strategy()

# Listar todas
all_strategies = storage.get_all_strategies()

# Exportar para dashboard
data = storage.export_for_dashboard()
```

---

### 6. auto_evolve.py - Sistema de Evolucao

**Funcao:** Evolui estrategias automaticamente usando algoritmo genetico.

**Processo:**
1. Carregar baseline (melhor estrategia atual)
2. Gerar mutacoes (variacoes de parametros)
3. Testar cada variacao via WFO
4. Se melhor que baseline: salvar como novo baseline
5. Deploy automatico no bot

**Uso:**
```bash
python auto_evolve.py           # Evolucao completa
python auto_evolve.py --quick   # Evolucao rapida (menos folds)
```

---

### 7. bot.py - Bot de Trading Live

**Funcao:** Executa trades em tempo real na Binance.

**Features:**
- Gerenciamento de posicoes
- Stop Loss / Take Profit automaticos
- Risk management
- Logging completo

**Uso:**
```bash
python bot.py
```

---

### 8. dashboard.py - Dashboard Web

**Funcao:** Interface web para monitoramento em tempo real.

**Endpoints API:**
| Endpoint | Descricao |
|----------|-----------|
| `/api/status` | Status geral (balance, PnL, etc) |
| `/api/positions` | Posicoes abertas |
| `/api/strategies` | Estrategias validadas |
| `/api/strategies/<id>` | Detalhes de uma estrategia |
| `/api/trades` | Historico de trades |
| `/api/metrics` | Metricas de performance |
| `/api/logs` | Logs do sistema |

**Uso:**
```bash
python dashboard.py
# Acesse http://localhost:5000
```

---

## Comandos Rapidos

```bash
# Rodar testes
python tests/test_system.py         # Todos os testes
python tests/test_system.py --quick # Testes rapidos

# Iniciar componentes
python auto_evolve.py       # Evolucao de estrategias
python dashboard.py         # Dashboard (http://localhost:5000)
python bot.py               # Bot de trading

# Ou usar scripts .bat
run_all.bat                 # Tudo junto
run_evolve.bat              # So evolucao
run_dashboard.bat           # So dashboard
run_bot.bat                 # So bot
```

---

## Configuracao (.env)

```env
BINANCE_API_KEY=sua_api_key
BINANCE_SECRET_KEY=sua_secret_key
```

---

## Performance

| Metrica | Valor |
|---------|-------|
| Backtest speed | ~5,500-6,500 candles/s |
| Precisao indicadores | < 0.01 vs pandas_ta |
| Consistencia PnL | < 0.01% discrepancia |
| Testes | 19/19 passando |

---

## Fluxo de Desenvolvimento

1. **Modificar estrategia:** Editar `core/signals.py`
2. **Testar:** `python tests/test_system.py`
3. **Validar WFO:** Usar `portfolio_wfo.py`
4. **Se aprovado:** Adicionar ao `auto_evolve.py`
5. **Deploy:** Bot usa automaticamente a melhor estrategia

---

## Proximos Passos para Melhoria

### Performance
- [ ] Implementar cache de indicadores por simbolo
- [ ] Paralelizar backtest por simbolo
- [ ] Usar numba para calculos criticos

### Estrategias
- [ ] Adicionar mais estrategias (VWAP, Order Flow)
- [ ] Implementar ML para classificacao de mercado
- [ ] Filtros de volatilidade/tendencia

### Risk Management
- [ ] Position sizing dinamico (Kelly Criterion)
- [ ] Correlacao entre posicoes
- [ ] Max drawdown protection

### Dashboard
- [ ] Graficos de equity curve
- [ ] Alertas por Telegram/Discord
- [ ] Controle de start/stop do bot

### Backtest
- [ ] Slippage dinamico
- [ ] Order book simulation
- [ ] Latency simulation
