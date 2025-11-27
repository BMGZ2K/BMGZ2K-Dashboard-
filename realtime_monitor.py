"""
Monitor de Performance em Tempo Real
=====================================
Sistema para:
1. Monitorar PnL em tempo real
2. Detectar problemas automaticamente
3. Sugerir otimizações baseadas em dados reais
4. Registrar histórico de performance

VERSÃO: 2.0 - Otimizações de I/O e estabilidade
- Arquivo de estado lido UMA vez por iteração (não por posição)
- Exception handling melhorado
- Logging adequado
- Cache de conexão
"""
import os
import sys
import json
import time
import ccxt
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from collections import deque
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from core.config import API_KEY, SECRET_KEY

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
log = logging.getLogger(__name__)


@dataclass
class PositionSnapshot:
    """Snapshot de uma posição."""
    symbol: str
    side: str
    entry_price: float
    current_price: float
    quantity: float
    unrealized_pnl: float
    pnl_pct: float
    distance_to_sl: float  # % até SL
    distance_to_tp: float  # % até TP
    duration_hours: float
    timestamp: str


@dataclass
class PerformanceSnapshot:
    """Snapshot de performance geral."""
    timestamp: str
    balance: float
    equity: float
    unrealized_pnl: float
    num_positions: int
    winning_positions: int
    losing_positions: int
    total_margin_used: float
    margin_usage_pct: float
    best_position: Optional[str]
    worst_position: Optional[str]
    positions: List[PositionSnapshot]


class RealtimeMonitor:
    """Monitor de performance em tempo real."""

    def __init__(self):
        self.exchange = self._create_exchange()
        # Usar deque para limitar memória automaticamente
        self.history: deque = deque(maxlen=100)
        self.alerts: List[Dict] = []
        self.start_balance: Optional[float] = None
        self.start_time: datetime = datetime.now()
        # Cache para reduzir I/O
        self._state_cache = {}
        self._state_cache_time = 0

    def _create_exchange(self):
        """Criar conexão com exchange (testnet)."""
        exchange = ccxt.binance({
            'apiKey': API_KEY,
            'secret': SECRET_KEY,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        exchange.set_sandbox_mode(True)
        return exchange

    def _load_trader_state(self) -> dict:
        """Carregar trader_state UMA vez por iteração (não por posição)."""
        now = time.time()
        # Cache por 5 segundos
        if now - self._state_cache_time < 5:
            return self._state_cache

        try:
            state_file = 'state/trader_state.json'
            if os.path.exists(state_file):
                with open(state_file, 'r', encoding='utf-8') as f:
                    self._state_cache = json.load(f)
                    self._state_cache_time = now
                    return self._state_cache
        except FileNotFoundError:
            log.debug("trader_state.json não encontrado")
        except json.JSONDecodeError as e:
            log.warning(f"JSON inválido em trader_state: {e}")
        except Exception as e:
            log.error(f"Erro ao carregar trader_state: {e}")

        return {}

    def fetch_current_state(self) -> PerformanceSnapshot:
        """Buscar estado atual da conta."""
        try:
            # Balance e posições
            balance = self.exchange.fetch_balance()
            positions = self.exchange.fetch_positions()

            # Filtrar posições ativas
            active_positions = [p for p in positions if float(p.get('contracts', 0)) > 0]

            # OTIMIZADO: Carregar estado UMA vez antes do loop
            trader_state = self._load_trader_state()
            saved_positions = trader_state.get('positions', {})

            # Calcular métricas
            total_balance = float(balance.get('USDT', {}).get('total', 0))
            free_balance = float(balance.get('USDT', {}).get('free', 0))

            if self.start_balance is None:
                self.start_balance = total_balance

            position_snapshots = []
            total_unrealized = 0
            total_margin = 0
            winning = 0
            losing = 0
            best_pnl_pct = -999
            worst_pnl_pct = 999
            best_symbol = None
            worst_symbol = None

            for pos in active_positions:
                symbol = pos['symbol']
                side = pos['side']
                entry = float(pos['entryPrice'])
                current = float(pos['markPrice'])
                qty = float(pos['contracts'])
                notional = float(pos.get('notional', qty * current))
                margin = float(pos.get('initialMargin', notional / 5))

                # Calcular PnL
                if side == 'long':
                    unrealized = (current - entry) * qty
                    pnl_pct = (current - entry) / entry * 100
                else:
                    unrealized = (entry - current) * qty
                    pnl_pct = (entry - current) / entry * 100

                total_unrealized += unrealized
                total_margin += margin

                if unrealized > 0:
                    winning += 1
                else:
                    losing += 1

                if pnl_pct > best_pnl_pct:
                    best_pnl_pct = pnl_pct
                    best_symbol = symbol
                if pnl_pct < worst_pnl_pct:
                    worst_pnl_pct = pnl_pct
                    worst_symbol = symbol

                # OTIMIZADO: Usar estado já carregado (não ler arquivo novamente)
                sl_distance = 0
                tp_distance = 0
                duration = 0
                pos_data = saved_positions.get(symbol, {})

                if pos_data:
                    # SL/TP distances
                    sl = pos_data.get('stop_loss', entry)
                    tp = pos_data.get('take_profit', entry)
                    if sl and current > 0:
                        sl_distance = abs(current - sl) / current * 100
                    if tp and current > 0:
                        tp_distance = abs(tp - current) / current * 100

                    # Duration
                    if 'entry_time' in pos_data:
                        try:
                            entry_time = datetime.fromisoformat(pos_data['entry_time'])
                            duration = (datetime.now() - entry_time).total_seconds() / 3600
                        except (ValueError, TypeError) as e:
                            log.debug(f"Erro parsing entry_time para {symbol}: {e}")

                position_snapshots.append(PositionSnapshot(
                    symbol=symbol,
                    side=side,
                    entry_price=entry,
                    current_price=current,
                    quantity=qty,
                    unrealized_pnl=unrealized,
                    pnl_pct=pnl_pct,
                    distance_to_sl=sl_distance,
                    distance_to_tp=tp_distance,
                    duration_hours=duration,
                    timestamp=datetime.now().isoformat()
                ))

            equity = total_balance + total_unrealized
            margin_usage = total_margin / total_balance * 100 if total_balance > 0 else 0

            snapshot = PerformanceSnapshot(
                timestamp=datetime.now().isoformat(),
                balance=total_balance,
                equity=equity,
                unrealized_pnl=total_unrealized,
                num_positions=len(active_positions),
                winning_positions=winning,
                losing_positions=losing,
                total_margin_used=total_margin,
                margin_usage_pct=margin_usage,
                best_position=best_symbol,
                worst_position=worst_symbol,
                positions=position_snapshots
            )

            return snapshot

        except Exception as e:
            print(f"Erro ao buscar estado: {e}")
            return None

    def check_alerts(self, snapshot: PerformanceSnapshot) -> List[Dict]:
        """Verificar e gerar alertas."""
        alerts = []

        # Alerta: Drawdown
        if self.start_balance and snapshot.equity < self.start_balance * 0.95:
            dd = (self.start_balance - snapshot.equity) / self.start_balance * 100
            alerts.append({
                'type': 'DRAWDOWN',
                'severity': 'WARNING' if dd < 10 else 'CRITICAL',
                'message': f'Drawdown de {dd:.1f}% desde o início',
                'timestamp': datetime.now().isoformat()
            })

        # Alerta: Posição próxima do SL
        for pos in snapshot.positions:
            if pos.distance_to_sl < 1.0:  # Menos de 1% do SL
                alerts.append({
                    'type': 'NEAR_STOP_LOSS',
                    'severity': 'WARNING',
                    'message': f'{pos.symbol} está a {pos.distance_to_sl:.2f}% do Stop Loss',
                    'timestamp': datetime.now().isoformat()
                })

        # Alerta: Posição perdedora por muito tempo
        for pos in snapshot.positions:
            if pos.pnl_pct < -5 and pos.duration_hours > 12:
                alerts.append({
                    'type': 'LONG_LOSER',
                    'severity': 'INFO',
                    'message': f'{pos.symbol} perdendo {abs(pos.pnl_pct):.1f}% há {pos.duration_hours:.0f}h',
                    'timestamp': datetime.now().isoformat()
                })

        # Alerta: Margem alta
        if snapshot.margin_usage_pct > 80:
            alerts.append({
                'type': 'HIGH_MARGIN',
                'severity': 'WARNING',
                'message': f'Uso de margem em {snapshot.margin_usage_pct:.1f}%',
                'timestamp': datetime.now().isoformat()
            })

        return alerts

    def generate_insights(self, snapshots: List[PerformanceSnapshot]) -> Dict:
        """Gerar insights baseados no histórico."""
        if len(snapshots) < 2:
            return {}

        # Tendência do PnL
        pnls = [s.unrealized_pnl for s in snapshots]
        equities = [s.equity for s in snapshots]

        pnl_trend = "SUBINDO" if pnls[-1] > pnls[0] else "DESCENDO"
        equity_change = equities[-1] - equities[0]
        equity_change_pct = equity_change / equities[0] * 100 if equities[0] > 0 else 0

        # Posições que mais contribuíram
        all_positions = {}
        for s in snapshots:
            for p in s.positions:
                if p.symbol not in all_positions:
                    all_positions[p.symbol] = []
                all_positions[p.symbol].append(p.pnl_pct)

        best_performers = []
        worst_performers = []
        for symbol, pnls in all_positions.items():
            avg_pnl = np.mean(pnls)
            if avg_pnl > 0:
                best_performers.append((symbol, avg_pnl))
            else:
                worst_performers.append((symbol, avg_pnl))

        best_performers.sort(key=lambda x: x[1], reverse=True)
        worst_performers.sort(key=lambda x: x[1])

        return {
            'pnl_trend': pnl_trend,
            'equity_change': equity_change,
            'equity_change_pct': equity_change_pct,
            'best_performers': best_performers[:3],
            'worst_performers': worst_performers[:3],
            'monitoring_duration_minutes': (datetime.now() - self.start_time).total_seconds() / 60
        }

    def save_snapshot(self, snapshot: PerformanceSnapshot):
        """Salvar snapshot no histórico."""
        self.history.append(snapshot)

        # Salvar em arquivo
        history_file = 'state/performance_history.json'
        try:
            history_data = []
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    history_data = json.load(f)

            # Converter para dict
            snapshot_dict = asdict(snapshot)
            snapshot_dict['positions'] = [asdict(p) for p in snapshot.positions]
            history_data.append(snapshot_dict)

            # Manter últimas 1000 entradas
            if len(history_data) > 1000:
                history_data = history_data[-1000:]

            with open(history_file, 'w') as f:
                json.dump(history_data, f)
        except Exception as e:
            print(f"Erro ao salvar histórico: {e}")

    def print_status(self, snapshot: PerformanceSnapshot, alerts: List[Dict], insights: Dict, iteration: int = 0):
        """Imprimir status formatado com cores e layout otimizado."""
        import platform

        # Cores ANSI
        GREEN = '\033[92m'
        RED = '\033[91m'
        YELLOW = '\033[93m'
        BLUE = '\033[94m'
        CYAN = '\033[96m'
        BOLD = '\033[1m'
        END = '\033[0m'

        # Limpar tela
        os.system('cls' if platform.system() == 'Windows' else 'clear')

        # Header
        print(f"\n{BOLD}{CYAN}{'═' * 75}{END}")
        print(f"{BOLD}{CYAN}  🔴 MONITOR EM TEMPO REAL - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{END}")
        if iteration > 0:
            elapsed = (datetime.now() - self.start_time).total_seconds() / 60
            print(f"{CYAN}  Iteração: {iteration} | Tempo: {elapsed:.1f}min{END}")
        print(f"{BOLD}{CYAN}{'═' * 75}{END}")

        # Métricas principais com cores
        print(f"\n{BOLD}┌─── CONTA {'─' * 63}┐{END}")
        print(f"│ {'BALANCE:':<18} ${snapshot.balance:>12,.2f}                              │")

        eq_color = GREEN if snapshot.equity >= snapshot.balance else RED
        print(f"│ {'EQUITY:':<18} {eq_color}${snapshot.equity:>12,.2f}{END}                              │")

        pnl_color = GREEN if snapshot.unrealized_pnl >= 0 else RED
        pnl_sign = '+' if snapshot.unrealized_pnl >= 0 else ''
        print(f"│ {'UNREALIZED PnL:':<18} {pnl_color}{pnl_sign}${snapshot.unrealized_pnl:>11,.2f}{END}                              │")

        margin_color = RED if snapshot.margin_usage_pct > 80 else YELLOW if snapshot.margin_usage_pct > 50 else GREEN
        print(f"│ {'MARGIN USAGE:':<18} {margin_color}{snapshot.margin_usage_pct:>11.1f}%{END}                               │")

        # Session PnL se disponível
        if self.start_balance:
            session_pnl = snapshot.equity - self.start_balance
            session_pnl_pct = (session_pnl / self.start_balance) * 100
            sess_color = GREEN if session_pnl >= 0 else RED
            sess_sign = '+' if session_pnl >= 0 else ''
            print(f"│ {'SESSION PnL:':<18} {sess_color}{sess_sign}${session_pnl:>11,.2f} ({sess_sign}{session_pnl_pct:.2f}%){END}                   │")

        print(f"{BOLD}└{'─' * 73}┘{END}")

        # Posições
        win_rate = (snapshot.winning_positions / snapshot.num_positions * 100) if snapshot.num_positions > 0 else 0
        avg_pnl = np.mean([p.pnl_pct for p in snapshot.positions]) if snapshot.positions else 0

        print(f"\n{BOLD}┌─── POSIÇÕES ({snapshot.num_positions}) │ WR: {win_rate:.0f}% ({snapshot.winning_positions}W/{snapshot.losing_positions}L) │ Avg: {avg_pnl:+.2f}% {'─' * 10}┐{END}")

        if snapshot.num_positions == 0:
            print(f"│ {YELLOW}Nenhuma posição aberta{END}                                                  │")
        else:
            print(f"│ {'Symbol':<12} {'Side':<8} {'PnL %':>10} {'Unrealized':>12} {'Dist SL':>10} {'Dist TP':>10} │")
            print(f"│ {'-' * 71} │")

            for pos in sorted(snapshot.positions, key=lambda x: x.pnl_pct, reverse=True):
                pnl_c = GREEN if pos.pnl_pct > 0 else RED
                pnl_sign = '+' if pos.pnl_pct > 0 else ''
                unr_sign = '+' if pos.unrealized_pnl > 0 else ''

                # Warning se perto do SL
                sl_str = f"{pos.distance_to_sl:>9.2f}%"
                if pos.distance_to_sl < 2.0:
                    sl_str = f"{RED}{BOLD}{pos.distance_to_sl:>9.2f}%{END}"
                elif pos.distance_to_sl < 5.0:
                    sl_str = f"{YELLOW}{pos.distance_to_sl:>9.2f}%{END}"

                print(f"│ {pos.symbol:<12} {pos.side:<8} {pnl_c}{pnl_sign}{pos.pnl_pct:>9.2f}%{END} {pnl_c}{unr_sign}${pos.unrealized_pnl:>10.2f}{END} {sl_str} {GREEN}{pos.distance_to_tp:>9.2f}%{END} │")

        print(f"{BOLD}└{'─' * 73}┘{END}")

        # Insights
        if insights:
            print(f"\n{BOLD}┌─── INSIGHTS {'─' * 60}┐{END}")
            trend_icon = "📈" if insights.get('pnl_trend') == 'SUBINDO' else "📉"
            trend_color = GREEN if insights.get('pnl_trend') == 'SUBINDO' else RED
            print(f"│ Tendência: {trend_color}{trend_icon} {insights.get('pnl_trend', 'N/A')}{END}                                        │")

            eq_change = insights.get('equity_change', 0)
            eq_pct = insights.get('equity_change_pct', 0)
            eq_c = GREEN if eq_change >= 0 else RED
            eq_s = '+' if eq_change >= 0 else ''
            print(f"│ Mudança Equity: {eq_c}{eq_s}${eq_change:,.2f} ({eq_s}{eq_pct:.2f}%){END}                           │")
            print(f"│ Monitorando há: {insights.get('monitoring_duration_minutes', 0):.1f} minutos                               │")

            if insights.get('best_performers'):
                best = ', '.join([f'{s}({GREEN}{p:+.1f}%{END})' for s, p in insights['best_performers']])
                print(f"│ Melhores: {best:<50} │")
            if insights.get('worst_performers'):
                worst = ', '.join([f'{s}({RED}{p:+.1f}%{END})' for s, p in insights['worst_performers']])
                print(f"│ Piores: {worst:<52} │")

            print(f"{BOLD}└{'─' * 73}┘{END}")

        # Alertas
        if alerts:
            print(f"\n{BOLD}┌─── ⚠️  ALERTAS ({len(alerts)}) {'─' * 53}┐{END}")
            sev_colors = {'CRITICAL': RED + BOLD, 'WARNING': YELLOW, 'INFO': BLUE}
            for alert in alerts:
                color = sev_colors.get(alert['severity'], '')
                print(f"│ {color}[{alert['severity']:<8}]{END} {alert['message']:<56} │")
            print(f"{BOLD}└{'─' * 73}┘{END}")

        print(f"\n{CYAN}Última atualização: {datetime.now().strftime('%H:%M:%S')} │ Próxima em 60s{END}")

    def run(self, interval_seconds: int = 60, duration_hours: float = 5):
        """Executar monitoramento contínuo."""
        import platform

        # Cores
        CYAN = '\033[96m'
        BOLD = '\033[1m'
        END = '\033[0m'

        os.system('cls' if platform.system() == 'Windows' else 'clear')
        print(f"\n{BOLD}{CYAN}{'═' * 75}{END}")
        print(f"{BOLD}{CYAN}  🚀 INICIANDO MONITORAMENTO EM TEMPO REAL{END}")
        print(f"{CYAN}  Duração: {duration_hours}h | Intervalo: {interval_seconds}s{END}")
        print(f"{CYAN}  Término previsto: {(datetime.now() + timedelta(hours=duration_hours)).strftime('%H:%M:%S')}{END}")
        print(f"{BOLD}{CYAN}{'═' * 75}{END}")
        time.sleep(2)

        end_time = datetime.now() + timedelta(hours=duration_hours)
        iteration = 0
        consecutive_errors = 0

        while datetime.now() < end_time:
            iteration += 1

            try:
                # Buscar estado atual
                snapshot = self.fetch_current_state()
                if snapshot is None:
                    consecutive_errors += 1
                    log.warning(f"Snapshot vazio, tentativa {consecutive_errors}/5")
                    if consecutive_errors >= 5:
                        log.error("Muitos erros consecutivos, aguardando 60s")
                        time.sleep(60)
                        consecutive_errors = 0
                    else:
                        time.sleep(30)
                    continue

                consecutive_errors = 0  # Reset no sucesso

                # Salvar snapshot
                self.save_snapshot(snapshot)

                # Verificar alertas
                alerts = self.check_alerts(snapshot)
                self.alerts.extend(alerts)

                # Gerar insights (a cada 5 iterações)
                insights = {}
                if iteration % 5 == 0 and len(self.history) > 1:
                    insights = self.generate_insights(list(self.history)[-20:])

                # Imprimir status com iteração
                self.print_status(snapshot, alerts, insights, iteration)

                # Salvar resumo
                self._save_summary(snapshot, insights)

            except ccxt.NetworkError as e:
                log.error(f"Erro de rede na iteração {iteration}: {e}")
                consecutive_errors += 1
            except Exception as e:
                log.error(f"Erro na iteração {iteration}: {e}")
                consecutive_errors += 1

            time.sleep(interval_seconds)

        print("\n" + "=" * 70)
        print("  MONITORAMENTO FINALIZADO")
        print("=" * 70)

        # Relatório final
        self._generate_final_report()

    def _save_summary(self, snapshot: PerformanceSnapshot, insights: Dict):
        """Salvar resumo atual."""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'balance': snapshot.balance,
            'equity': snapshot.equity,
            'unrealized_pnl': snapshot.unrealized_pnl,
            'num_positions': snapshot.num_positions,
            'winning': snapshot.winning_positions,
            'losing': snapshot.losing_positions,
            'margin_usage_pct': snapshot.margin_usage_pct,
            'insights': insights,
            'monitoring_start': self.start_time.isoformat(),
            'start_balance': self.start_balance
        }

        with open('state/realtime_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

    def _generate_final_report(self):
        """Gerar relatório final do monitoramento."""
        if not self.history:
            return

        first = self.history[0]
        last = self.history[-1]

        report = {
            'period': {
                'start': first.timestamp,
                'end': last.timestamp,
                'duration_hours': (datetime.now() - self.start_time).total_seconds() / 3600
            },
            'performance': {
                'start_equity': first.equity,
                'end_equity': last.equity,
                'equity_change': last.equity - first.equity,
                'equity_change_pct': (last.equity - first.equity) / first.equity * 100,
                'max_equity': max(s.equity for s in self.history),
                'min_equity': min(s.equity for s in self.history),
                'max_drawdown_pct': (max(s.equity for s in self.history) - min(s.equity for s in self.history)) / max(s.equity for s in self.history) * 100
            },
            'positions': {
                'total_unique': len(set(p.symbol for s in self.history for p in s.positions)),
                'avg_positions': np.mean([s.num_positions for s in self.history]),
                'max_positions': max(s.num_positions for s in self.history)
            },
            'alerts': {
                'total': len(self.alerts),
                'by_type': {}
            }
        }

        # Contar alertas por tipo
        for alert in self.alerts:
            t = alert['type']
            report['alerts']['by_type'][t] = report['alerts']['by_type'].get(t, 0) + 1

        with open('state/monitoring_report.json', 'w') as f:
            json.dump(report, f, indent=2)

        print("\n--- RELATÓRIO FINAL ---")
        print(f"Duração: {report['period']['duration_hours']:.1f} horas")
        print(f"Equity inicial: ${report['performance']['start_equity']:,.2f}")
        print(f"Equity final: ${report['performance']['end_equity']:,.2f}")
        print(f"Variação: ${report['performance']['equity_change']:,.2f} ({report['performance']['equity_change_pct']:.2f}%)")
        print(f"Max Drawdown: {report['performance']['max_drawdown_pct']:.1f}%")
        print(f"Total alertas: {report['alerts']['total']}")


def main():
    """Executar monitor."""
    monitor = RealtimeMonitor()
    monitor.run(interval_seconds=60, duration_hours=5)


if __name__ == '__main__':
    main()
