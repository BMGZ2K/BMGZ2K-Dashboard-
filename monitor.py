"""
Real-Time Trading Monitor
=========================
Dashboard para monitoramento de performance em tempo real.

Funcionalidades:
- Visualizacao de posicoes abertas
- Metricas de performance (Sharpe, Sortino, Win Rate, etc.)
- Alertas de risco
- Validacao contra thresholds WFO
- Status de saude do sistema

Uso:
    python monitor.py           # Execucao unica
    python monitor.py --live    # Monitoramento continuo
    python monitor.py --export  # Exportar metricas para JSON
"""
import os
import sys
import json
import time
import hashlib
import hmac
import urllib.parse
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import (
    API_KEY, SECRET_KEY, USE_TESTNET,
    WFO_VALIDATED_PARAMS, VALIDATION_THRESHOLDS,
    get_validated_params
)
from core.utils import load_json_safe


class TradingMonitor:
    """Monitor de trading em tempo real."""

    def __init__(self):
        self.base_url = 'https://testnet.binancefuture.com' if USE_TESTNET else 'https://fapi.binance.com'
        self.headers = {'X-MBX-APIKEY': API_KEY}
        self.params = get_validated_params()

        # Carregar historico de trades
        self.trade_history = self._load_trade_history()

        # Thresholds para alertas
        self.alert_thresholds = {
            'max_drawdown_pct': 15.0,      # Alerta se DD > 15%
            'max_position_loss_pct': 10.0,  # Alerta se posicao perdendo > 10%
            'min_win_rate': 40.0,           # Alerta se WR < 40%
            'max_margin_usage': 80.0,       # Alerta se margem > 80%
            'min_profit_factor': 1.0,       # Alerta se PF < 1.0
        }

    def _sign(self, params: Dict) -> str:
        """Criar assinatura HMAC."""
        query_string = urllib.parse.urlencode(params)
        return hmac.new(
            SECRET_KEY.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    def _api_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Fazer request autenticado."""
        params = params or {}
        params['timestamp'] = int(time.time() * 1000)
        params['recvWindow'] = 5000
        params['signature'] = self._sign(params)

        try:
            r = requests.get(
                f'{self.base_url}{endpoint}',
                params=params,
                headers=self.headers,
                timeout=15
            )
            if r.status_code == 200:
                return r.json()
            else:
                print(f"API Error: {r.status_code} - {r.text}")
                return None
        except Exception as e:
            print(f"Request Error: {e}")
            return None

    def _load_trade_history(self) -> List[Dict]:
        """Carregar historico de trades do arquivo de estado."""
        try:
            state = load_json_safe('state/trader_state.json')
            return state.get('trade_history', [])
        except Exception:
            return []

    def get_account_info(self) -> Dict:
        """Obter informacoes da conta."""
        data = self._api_request('/fapi/v2/account')
        if not data:
            return {}

        return {
            'total_wallet': float(data.get('totalWalletBalance', 0)),
            'unrealized_pnl': float(data.get('totalUnrealizedProfit', 0)),
            'available': float(data.get('availableBalance', 0)),
            'margin_balance': float(data.get('totalMarginBalance', 0)),
            'position_margin': float(data.get('totalPositionInitialMargin', 0)),
            'positions': data.get('positions', [])
        }

    def get_open_positions(self) -> List[Dict]:
        """Obter posicoes abertas."""
        account = self.get_account_info()
        positions = account.get('positions', [])

        open_positions = []
        for p in positions:
            qty = float(p.get('positionAmt', 0))
            if qty == 0:
                continue

            entry = float(p.get('entryPrice', 0))
            mark = float(p.get('markPrice', 0))
            pnl = float(p.get('unrealizedProfit', 0))
            margin = float(p.get('initialMargin', 0))
            leverage = int(p.get('leverage', 1))

            open_positions.append({
                'symbol': p.get('symbol', 'N/A'),
                'side': 'LONG' if qty > 0 else 'SHORT',
                'quantity': abs(qty),
                'entry_price': entry,
                'mark_price': mark,
                'pnl': pnl,
                'roe_pct': (pnl / margin * 100) if margin > 0 else 0,
                'margin': margin,
                'leverage': leverage
            })

        return sorted(open_positions, key=lambda x: x['pnl'], reverse=True)

    def calculate_live_metrics(self) -> Dict:
        """Calcular metricas de performance em tempo real."""
        account = self.get_account_info()
        positions = self.get_open_positions()

        # Metricas basicas da conta
        total_wallet = account.get('total_wallet', 0)
        unrealized_pnl = account.get('unrealized_pnl', 0)
        margin_balance = account.get('margin_balance', 0)
        position_margin = account.get('position_margin', 0)

        # Margin usage
        margin_usage = (position_margin / margin_balance * 100) if margin_balance > 0 else 0

        # Performance das posicoes abertas
        if positions:
            winners = [p for p in positions if p['pnl'] > 0]
            losers = [p for p in positions if p['pnl'] < 0]

            total_pnl = sum(p['pnl'] for p in positions)
            gross_profit = sum(p['pnl'] for p in winners) if winners else 0
            gross_loss = abs(sum(p['pnl'] for p in losers)) if losers else 0

            position_win_rate = len(winners) / len(positions) * 100 if positions else 0
            position_profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

            avg_win = gross_profit / len(winners) if winners else 0
            avg_loss = gross_loss / len(losers) if losers else 0
        else:
            total_pnl = 0
            position_win_rate = 0
            position_profit_factor = 0
            avg_win = 0
            avg_loss = 0

        # Metricas do historico de trades
        history_metrics = self._calculate_history_metrics()

        # Drawdown atual (aproximado)
        current_dd_pct = (unrealized_pnl / total_wallet * 100) if total_wallet > 0 and unrealized_pnl < 0 else 0

        return {
            # Conta
            'total_wallet': total_wallet,
            'unrealized_pnl': unrealized_pnl,
            'unrealized_pnl_pct': (unrealized_pnl / total_wallet * 100) if total_wallet > 0 else 0,
            'margin_balance': margin_balance,
            'margin_usage_pct': margin_usage,

            # Posicoes abertas
            'open_positions': len(positions),
            'position_pnl': total_pnl,
            'position_win_rate': position_win_rate,
            'position_profit_factor': position_profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,

            # Historico
            'total_trades': history_metrics.get('total_trades', 0),
            'historical_win_rate': history_metrics.get('win_rate', 0),
            'historical_profit_factor': history_metrics.get('profit_factor', 0),
            'total_realized_pnl': history_metrics.get('total_pnl', 0),
            'sharpe_ratio': history_metrics.get('sharpe_ratio', 0),
            'sortino_ratio': history_metrics.get('sortino_ratio', 0),
            'max_drawdown_pct': history_metrics.get('max_drawdown_pct', 0),

            # Risk
            'current_drawdown_pct': abs(current_dd_pct),

            # Timestamp
            'timestamp': datetime.now().isoformat()
        }

    def _calculate_history_metrics(self) -> Dict:
        """Calcular metricas do historico de trades."""
        if not self.trade_history:
            return {}

        trades = self.trade_history
        total_trades = len(trades)

        if total_trades == 0:
            return {'total_trades': 0}

        # PnLs
        pnls = [t.get('pnl', 0) for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        # Metricas basicas
        win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
        total_pnl = sum(pnls)
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Sharpe e Sortino (simplificados)
        import numpy as np
        if len(pnls) > 1:
            returns = np.array(pnls)
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            downside_returns = returns[returns < 0]
            downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0

            # Anualizar (assumindo ~100 trades por ano)
            sharpe = (mean_return / std_return * np.sqrt(100)) if std_return > 0 else 0
            sortino = (mean_return / downside_std * np.sqrt(100)) if downside_std > 0 else 0
        else:
            sharpe = 0
            sortino = 0

        # Max Drawdown
        cumsum = np.cumsum(pnls)
        peak = np.maximum.accumulate(cumsum)
        drawdown = peak - cumsum
        max_dd = np.max(drawdown) if len(drawdown) > 0 else 0
        max_dd_pct = (max_dd / (peak.max() if peak.max() > 0 else 1)) * 100

        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_pnl': total_pnl,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown_pct': max_dd_pct
        }

    def check_alerts(self, metrics: Dict) -> List[str]:
        """Verificar alertas de risco."""
        alerts = []

        # Drawdown alto
        if metrics.get('current_drawdown_pct', 0) > self.alert_thresholds['max_drawdown_pct']:
            alerts.append(f"ALERTA: Drawdown atual {metrics['current_drawdown_pct']:.1f}% > {self.alert_thresholds['max_drawdown_pct']}%")

        # Margem alta
        if metrics.get('margin_usage_pct', 0) > self.alert_thresholds['max_margin_usage']:
            alerts.append(f"ALERTA: Uso de margem {metrics['margin_usage_pct']:.1f}% > {self.alert_thresholds['max_margin_usage']}%")

        # Win rate baixo (se tiver trades suficientes)
        if metrics.get('total_trades', 0) >= 10:
            if metrics.get('historical_win_rate', 0) < self.alert_thresholds['min_win_rate']:
                alerts.append(f"ALERTA: Win Rate {metrics['historical_win_rate']:.1f}% < {self.alert_thresholds['min_win_rate']}%")

            # Profit factor baixo
            pf = metrics.get('historical_profit_factor', 0)
            if pf < self.alert_thresholds['min_profit_factor'] and pf != float('inf'):
                alerts.append(f"ALERTA: Profit Factor {pf:.2f} < {self.alert_thresholds['min_profit_factor']}")

        return alerts

    def validate_against_wfo(self, metrics: Dict) -> Dict:
        """Validar metricas contra thresholds WFO."""
        validations = []
        passed = True

        # Sharpe
        min_sharpe = VALIDATION_THRESHOLDS.get('min_sharpe', 1.0)
        sharpe = metrics.get('sharpe_ratio', 0)
        if sharpe < min_sharpe:
            validations.append(f"Sharpe {sharpe:.2f} < {min_sharpe}")
            passed = False
        else:
            validations.append(f"Sharpe {sharpe:.2f} >= {min_sharpe} [OK]")

        # Win Rate
        min_wr = VALIDATION_THRESHOLDS.get('min_win_rate', 0.40) * 100
        wr = metrics.get('historical_win_rate', 0)
        if wr < min_wr and metrics.get('total_trades', 0) >= 10:
            validations.append(f"Win Rate {wr:.1f}% < {min_wr:.0f}%")
            passed = False
        elif metrics.get('total_trades', 0) >= 10:
            validations.append(f"Win Rate {wr:.1f}% >= {min_wr:.0f}% [OK]")

        # Profit Factor
        min_pf = VALIDATION_THRESHOLDS.get('min_profit_factor', 1.3)
        pf = metrics.get('historical_profit_factor', 0)
        if pf < min_pf and pf != float('inf') and metrics.get('total_trades', 0) >= 10:
            validations.append(f"Profit Factor {pf:.2f} < {min_pf}")
            passed = False
        elif metrics.get('total_trades', 0) >= 10:
            validations.append(f"Profit Factor {pf:.2f} >= {min_pf} [OK]")

        # Max Drawdown
        max_dd = VALIDATION_THRESHOLDS.get('max_drawdown', 0.25) * 100
        dd = metrics.get('max_drawdown_pct', 0)
        if dd > max_dd:
            validations.append(f"Max DD {dd:.1f}% > {max_dd:.0f}%")
            passed = False
        else:
            validations.append(f"Max DD {dd:.1f}% <= {max_dd:.0f}% [OK]")

        return {
            'passed': passed,
            'validations': validations
        }

    def get_system_status(self) -> Dict:
        """Verificar status do sistema."""
        status = {
            'bot_running': False,
            'last_trade_time': None,
            'strategy': self.params.get('strategy', 'unknown'),
            'params_source': 'unknown'
        }

        # Verificar se bot esta rodando (checar arquivo de estado recente)
        try:
            state_path = 'state/trader_state.json'
            if os.path.exists(state_path):
                mtime = os.path.getmtime(state_path)
                age_minutes = (time.time() - mtime) / 60
                status['bot_running'] = age_minutes < 10  # Considera rodando se atualizado nos ultimos 10 min
                status['state_age_minutes'] = age_minutes
        except Exception:
            pass

        # Ultimo trade
        if self.trade_history:
            last_trade = self.trade_history[-1]
            status['last_trade_time'] = last_trade.get('exit_time', last_trade.get('entry_time'))

        # Fonte dos parametros
        if os.path.exists('state/current_best.json'):
            status['params_source'] = 'current_best.json (WFO Optimized)'
        elif os.path.exists('state/trader_state.json'):
            status['params_source'] = 'trader_state.json'
        else:
            status['params_source'] = 'config.py (Default)'

        return status

    def display_dashboard(self, metrics: Dict, positions: List[Dict], alerts: List[str],
                          validation: Dict, status: Dict):
        """Exibir dashboard no terminal."""
        # Limpar tela
        os.system('cls' if os.name == 'nt' else 'clear')

        print("=" * 80)
        print(f"{'TRADING MONITOR - REAL TIME DASHBOARD':^80}")
        print("=" * 80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Ambiente: {'TESTNET' if USE_TESTNET else 'MAINNET'}")
        print(f"Estrategia: {status.get('strategy', 'N/A')}")
        print(f"Bot Status: {'RUNNING' if status.get('bot_running') else 'STOPPED/UNKNOWN'}")

        # Resumo da conta
        print("\n" + "-" * 80)
        print("CONTA")
        print("-" * 80)
        print(f"  Wallet Balance:     ${metrics.get('total_wallet', 0):,.2f}")
        pnl = metrics.get('unrealized_pnl', 0)
        pnl_pct = metrics.get('unrealized_pnl_pct', 0)
        print(f"  Unrealized PnL:     ${pnl:+,.2f} ({pnl_pct:+.2f}%)")
        print(f"  Margin Usage:       {metrics.get('margin_usage_pct', 0):.1f}%")

        # Posicoes
        print("\n" + "-" * 80)
        print(f"POSICOES ABERTAS ({len(positions)})")
        print("-" * 80)

        if positions:
            print(f"{'Symbol':<12} {'Side':<6} {'Entry':<12} {'Mark':<12} {'PnL':<12} {'ROE%':<8}")
            print("-" * 62)

            for p in positions:
                pnl_str = f"${p['pnl']:+.2f}"
                roe_str = f"{p['roe_pct']:+.1f}%"
                print(f"{p['symbol']:<12} {p['side']:<6} ${p['entry_price']:<11.4f} ${p['mark_price']:<11.4f} {pnl_str:<12} {roe_str:<8}")

            print("-" * 62)
            total_pos_pnl = sum(p['pnl'] for p in positions)
            print(f"{'TOTAL:':<50} ${total_pos_pnl:+.2f}")
        else:
            print("  Nenhuma posicao aberta")

        # Metricas de performance
        print("\n" + "-" * 80)
        print("METRICAS DE PERFORMANCE")
        print("-" * 80)
        print(f"  Posicoes - Win Rate: {metrics.get('position_win_rate', 0):.1f}% | PF: {metrics.get('position_profit_factor', 0):.2f}")
        print(f"  Historico - Trades: {metrics.get('total_trades', 0)} | Win Rate: {metrics.get('historical_win_rate', 0):.1f}% | PF: {metrics.get('historical_profit_factor', 0):.2f}")
        print(f"  Sharpe: {metrics.get('sharpe_ratio', 0):.2f} | Sortino: {metrics.get('sortino_ratio', 0):.2f}")
        print(f"  Max Drawdown: {metrics.get('max_drawdown_pct', 0):.1f}% | Current DD: {metrics.get('current_drawdown_pct', 0):.1f}%")
        print(f"  Total Realized PnL: ${metrics.get('total_realized_pnl', 0):+,.2f}")

        # Validacao WFO
        print("\n" + "-" * 80)
        print(f"VALIDACAO WFO: {'PASSED' if validation.get('passed') else 'FAILED'}")
        print("-" * 80)
        for v in validation.get('validations', []):
            indicator = "[OK]" if "[OK]" in v else "[X]"
            print(f"  {v}")

        # Alertas
        if alerts:
            print("\n" + "-" * 80)
            print("ALERTAS")
            print("-" * 80)
            for alert in alerts:
                print(f"  {alert}")

        # Health check
        print("\n" + "-" * 80)
        print("HEALTH CHECK")
        print("-" * 80)

        health_issues = []

        # Verificar se bot esta rodando
        if not status.get('bot_running'):
            health_issues.append("Bot nao esta rodando (state desatualizado)")

        # Verificar margin
        if metrics.get('margin_usage_pct', 0) > 80:
            health_issues.append(f"Margem alta: {metrics['margin_usage_pct']:.1f}%")

        # Verificar drawdown
        if metrics.get('current_drawdown_pct', 0) > 10:
            health_issues.append(f"Drawdown elevado: {metrics['current_drawdown_pct']:.1f}%")

        # Verificar posicoes com grande perda
        for p in positions:
            if p['roe_pct'] < -10:
                health_issues.append(f"Posicao {p['symbol']} com perda elevada: {p['roe_pct']:.1f}%")

        if health_issues:
            for issue in health_issues:
                print(f"  [!] {issue}")
        else:
            print("  [OK] Sistema operando normalmente")

        print("\n" + "=" * 80)

    def export_metrics(self, filepath: str = 'state/monitor_metrics.json'):
        """Exportar metricas para arquivo JSON."""
        metrics = self.calculate_live_metrics()
        positions = self.get_open_positions()
        alerts = self.check_alerts(metrics)
        validation = self.validate_against_wfo(metrics)
        status = self.get_system_status()

        export_data = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'positions': positions,
            'alerts': alerts,
            'validation': validation,
            'status': status
        }

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

        print(f"Metricas exportadas para: {filepath}")
        return export_data

    def run_once(self):
        """Executar uma vez e exibir dashboard."""
        metrics = self.calculate_live_metrics()
        positions = self.get_open_positions()
        alerts = self.check_alerts(metrics)
        validation = self.validate_against_wfo(metrics)
        status = self.get_system_status()

        self.display_dashboard(metrics, positions, alerts, validation, status)

    def run_live(self, interval: int = 30):
        """Monitoramento continuo."""
        print(f"Iniciando monitoramento continuo (intervalo: {interval}s)")
        print("Pressione Ctrl+C para parar")

        try:
            while True:
                self.run_once()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nMonitoramento parado pelo usuario")


def main():
    parser = argparse.ArgumentParser(description='Trading Monitor')
    parser.add_argument('--live', action='store_true', help='Monitoramento continuo')
    parser.add_argument('--export', action='store_true', help='Exportar metricas para JSON')
    parser.add_argument('--interval', type=int, default=30, help='Intervalo de atualizacao (segundos)')

    args = parser.parse_args()

    monitor = TradingMonitor()

    if args.export:
        monitor.export_metrics()
    elif args.live:
        monitor.run_live(args.interval)
    else:
        monitor.run_once()


if __name__ == '__main__':
    main()
