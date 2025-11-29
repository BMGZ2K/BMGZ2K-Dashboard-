"""
WFO Strategy Storage
====================
Sistema de armazenamento de estrategias validadas com todas as metricas.
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import asdict
import hashlib

from .metrics import StrategyMetrics, PerformanceMetrics, RobustnessMetrics


class StrategyStorage:
    """
    Sistema de armazenamento de estrategias otimizadas.

    Funcionalidades:
    - Salva estrategias com todas as metricas
    - Mantém historico de versoes
    - Ranking automatico por score
    - Backup automatico
    - Exportacao para producao
    """

    def __init__(self, storage_dir: str = "wfo/strategies"):
        """
        Inicializa o storage.

        Args:
            storage_dir: Diretorio para armazenar estrategias
        """
        self.storage_dir = Path(storage_dir)
        self.strategies_file = self.storage_dir / "strategies.json"
        self.history_dir = self.storage_dir / "history"
        self.backups_dir = self.storage_dir / "backups"

        # Cria diretorios
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(exist_ok=True)
        self.backups_dir.mkdir(exist_ok=True)

        # Carrega estrategias existentes
        self.strategies: Dict[str, Dict] = self._load_strategies()

    def _load_strategies(self) -> Dict[str, Dict]:
        """Carrega estrategias do arquivo."""
        if self.strategies_file.exists():
            try:
                with open(self.strategies_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_strategies(self):
        """Salva estrategias no arquivo."""
        # Backup antes de salvar
        if self.strategies_file.exists():
            backup_name = f"strategies_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            backup_path = self.backups_dir / backup_name
            shutil.copy(self.strategies_file, backup_path)

            # Limpa backups antigos (mantém 10)
            self._cleanup_backups()

        # Salva atomicamente
        temp_file = self.strategies_file.with_suffix('.tmp')
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(self.strategies, f, indent=2, ensure_ascii=False)

        # Move temp para final
        temp_file.replace(self.strategies_file)

    def _cleanup_backups(self, keep: int = 10):
        """Remove backups antigos."""
        backups = sorted(self.backups_dir.glob("strategies_backup_*.json"))
        for backup in backups[:-keep]:
            backup.unlink()

    def _generate_id(self, strategy_name: str, params: Dict) -> str:
        """Gera ID unico para estrategia."""
        # Hash baseado em nome + parametros principais
        param_str = json.dumps(params, sort_keys=True)
        content = f"{strategy_name}_{param_str}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def save_strategy(
        self,
        metrics: StrategyMetrics,
        force: bool = False
    ) -> str:
        """
        Salva uma estrategia com suas metricas.

        Args:
            metrics: Metricas completas da estrategia
            force: Forca salvamento mesmo se score menor

        Returns:
            ID da estrategia salva
        """
        strategy_id = self._generate_id(metrics.strategy_name, metrics.params)

        # Converte para dict
        strategy_data = metrics.to_dict()
        strategy_data['id'] = strategy_id
        strategy_data['saved_at'] = datetime.now().isoformat()

        # Verifica se ja existe
        existing = self.strategies.get(strategy_id)

        if existing and not force:
            # Só salva se score melhorou
            if metrics.composite_score <= existing.get('composite_score', 0):
                print(f"[Storage] Estrategia {strategy_id} nao salva - score nao melhorou")
                return strategy_id

        # Salva no historico se ja existia
        if existing:
            self._save_to_history(strategy_id, existing)

        # Salva nova versao
        self.strategies[strategy_id] = strategy_data
        self._save_strategies()

        print(f"[Storage] Estrategia {strategy_id} salva - Score: {metrics.composite_score:.2f}")
        return strategy_id

    def _save_to_history(self, strategy_id: str, data: Dict):
        """Salva versao anterior no historico."""
        history_file = self.history_dir / f"{strategy_id}_history.json"

        history = []
        if history_file.exists():
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            except (json.JSONDecodeError, IOError):
                history = []

        # Adiciona versao antiga
        data['archived_at'] = datetime.now().isoformat()
        history.append(data)

        # Mantém apenas 20 versoes
        history = history[-20:]

        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)

    def get_strategy(self, strategy_id: str) -> Optional[Dict]:
        """Retorna estrategia pelo ID."""
        return self.strategies.get(strategy_id)

    def get_best_strategy(self, min_score: float = 0.0) -> Optional[Dict]:
        """
        Retorna a melhor estrategia.

        Args:
            min_score: Score minimo requerido

        Returns:
            Melhor estrategia ou None
        """
        if not self.strategies:
            return None

        sorted_strategies = sorted(
            self.strategies.values(),
            key=lambda x: x.get('composite_score', 0),
            reverse=True
        )

        for strategy in sorted_strategies:
            if strategy.get('composite_score', 0) >= min_score:
                # Verifica se é robusta
                robustness = strategy.get('robustness', {})
                if robustness.get('is_robust', False):
                    return strategy

        # Se nenhuma robusta, retorna a melhor acima do score minimo
        for strategy in sorted_strategies:
            if strategy.get('composite_score', 0) >= min_score:
                return strategy

        return sorted_strategies[0] if sorted_strategies else None

    def get_top_strategies(
        self,
        n: int = 10,
        robust_only: bool = True,
        min_score: float = 0.0
    ) -> List[Dict]:
        """
        Retorna as N melhores estrategias.

        Args:
            n: Numero de estrategias
            robust_only: Apenas estrategias robustas
            min_score: Score minimo

        Returns:
            Lista das melhores estrategias
        """
        strategies = list(self.strategies.values())

        # Filtra por score minimo
        strategies = [s for s in strategies if s.get('composite_score', 0) >= min_score]

        # Filtra apenas robustas se requisitado
        if robust_only:
            strategies = [
                s for s in strategies
                if s.get('robustness', {}).get('is_robust', False)
            ]

        # Ordena por score
        strategies = sorted(
            strategies,
            key=lambda x: x.get('composite_score', 0),
            reverse=True
        )

        return strategies[:n]

    def get_strategy_history(self, strategy_id: str) -> List[Dict]:
        """Retorna historico de versoes de uma estrategia."""
        history_file = self.history_dir / f"{strategy_id}_history.json"

        if history_file.exists():
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return []
        return []

    def delete_strategy(self, strategy_id: str) -> bool:
        """Remove uma estrategia."""
        if strategy_id in self.strategies:
            # Salva no historico antes de deletar
            self._save_to_history(strategy_id, self.strategies[strategy_id])

            del self.strategies[strategy_id]
            self._save_strategies()
            return True
        return False

    def export_for_production(
        self,
        strategy_id: str = None,
        output_file: str = "config/best_strategy.json"
    ) -> bool:
        """
        Exporta estrategia para uso em producao.

        Args:
            strategy_id: ID da estrategia (ou None para melhor)
            output_file: Arquivo de saida

        Returns:
            True se exportado com sucesso
        """
        if strategy_id:
            strategy = self.get_strategy(strategy_id)
        else:
            strategy = self.get_best_strategy()

        if not strategy:
            print("[Storage] Nenhuma estrategia para exportar")
            return False

        # Extrai apenas parametros necessarios para producao
        production_config = {
            'strategy_id': strategy.get('id'),
            'strategy_name': strategy.get('strategy_name'),
            'params': strategy.get('params', {}),
            'symbols': strategy.get('symbols', []),
            'timeframe': strategy.get('timeframe'),
            'metrics_summary': {
                'composite_score': strategy.get('composite_score'),
                'sharpe_ratio': strategy.get('performance', {}).get('sharpe_ratio'),
                'profit_factor': strategy.get('performance', {}).get('profit_factor'),
                'win_rate': strategy.get('performance', {}).get('win_rate'),
                'max_drawdown': strategy.get('performance', {}).get('max_drawdown'),
                'is_robust': strategy.get('robustness', {}).get('is_robust')
            },
            'exported_at': datetime.now().isoformat()
        }

        # Garante diretorio existe
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(production_config, f, indent=2)

        print(f"[Storage] Estrategia exportada para {output_file}")
        return True

    def get_statistics(self) -> Dict:
        """Retorna estatisticas do storage."""
        if not self.strategies:
            return {
                'total_strategies': 0,
                'robust_strategies': 0,
                'avg_score': 0,
                'best_score': 0,
                'avg_sharpe': 0,
                'avg_profit_factor': 0
            }

        strategies = list(self.strategies.values())

        robust_count = sum(
            1 for s in strategies
            if s.get('robustness', {}).get('is_robust', False)
        )

        scores = [s.get('composite_score', 0) for s in strategies]
        sharpes = [s.get('performance', {}).get('sharpe_ratio', 0) for s in strategies]
        pfs = [s.get('performance', {}).get('profit_factor', 0) for s in strategies]

        return {
            'total_strategies': len(strategies),
            'robust_strategies': robust_count,
            'avg_score': sum(scores) / len(scores) if scores else 0,
            'best_score': max(scores) if scores else 0,
            'avg_sharpe': sum(sharpes) / len(sharpes) if sharpes else 0,
            'avg_profit_factor': sum(pfs) / len(pfs) if pfs else 0
        }

    def search_strategies(
        self,
        filters: Dict = None,
        sort_by: str = 'composite_score',
        ascending: bool = False
    ) -> List[Dict]:
        """
        Busca estrategias com filtros.

        Args:
            filters: Filtros a aplicar
                - min_sharpe: float
                - min_win_rate: float
                - min_profit_factor: float
                - max_drawdown: float
                - is_robust: bool
                - strategy_name: str
            sort_by: Campo para ordenacao
            ascending: Ordem ascendente

        Returns:
            Lista de estrategias filtradas
        """
        strategies = list(self.strategies.values())

        if filters:
            # Filtra por sharpe minimo
            if 'min_sharpe' in filters:
                strategies = [
                    s for s in strategies
                    if s.get('performance', {}).get('sharpe_ratio', 0) >= filters['min_sharpe']
                ]

            # Filtra por win rate minimo
            if 'min_win_rate' in filters:
                strategies = [
                    s for s in strategies
                    if s.get('performance', {}).get('win_rate', 0) >= filters['min_win_rate']
                ]

            # Filtra por profit factor minimo
            if 'min_profit_factor' in filters:
                strategies = [
                    s for s in strategies
                    if s.get('performance', {}).get('profit_factor', 0) >= filters['min_profit_factor']
                ]

            # Filtra por max drawdown maximo
            if 'max_drawdown' in filters:
                strategies = [
                    s for s in strategies
                    if s.get('performance', {}).get('max_drawdown', 1) <= filters['max_drawdown']
                ]

            # Filtra apenas robustas
            if filters.get('is_robust'):
                strategies = [
                    s for s in strategies
                    if s.get('robustness', {}).get('is_robust', False)
                ]

            # Filtra por nome
            if 'strategy_name' in filters:
                strategies = [
                    s for s in strategies
                    if filters['strategy_name'].lower() in s.get('strategy_name', '').lower()
                ]

        # Ordena
        def get_sort_key(s):
            if '.' in sort_by:
                parts = sort_by.split('.')
                val = s
                for part in parts:
                    val = val.get(part, {}) if isinstance(val, dict) else 0
                return val
            return s.get(sort_by, 0)

        strategies = sorted(strategies, key=get_sort_key, reverse=not ascending)

        return strategies

    def compare_strategies(self, strategy_ids: List[str]) -> Dict:
        """
        Compara multiplas estrategias.

        Args:
            strategy_ids: Lista de IDs para comparar

        Returns:
            Comparacao detalhada
        """
        strategies = [self.get_strategy(sid) for sid in strategy_ids]
        strategies = [s for s in strategies if s is not None]

        if len(strategies) < 2:
            return {'error': 'Necessita pelo menos 2 estrategias para comparar'}

        comparison = {
            'strategies': [],
            'winner': None,
            'metrics_comparison': {}
        }

        # Metricas para comparar
        metrics_to_compare = [
            ('composite_score', 'Score Composto', True),  # (key, label, higher_better)
            ('performance.sharpe_ratio', 'Sharpe Ratio', True),
            ('performance.sortino_ratio', 'Sortino Ratio', True),
            ('performance.profit_factor', 'Profit Factor', True),
            ('performance.win_rate', 'Win Rate', True),
            ('performance.total_return', 'Retorno Total', True),
            ('performance.max_drawdown', 'Max Drawdown', False),
            ('robustness.oos_is_ratio', 'OOS/IS Ratio', True),
            ('robustness.consistency_score', 'Consistencia', True),
        ]

        for s in strategies:
            comparison['strategies'].append({
                'id': s.get('id'),
                'name': s.get('strategy_name'),
                'score': s.get('composite_score')
            })

        # Compara cada metrica
        for key, label, higher_better in metrics_to_compare:
            values = []
            for s in strategies:
                if '.' in key:
                    parts = key.split('.')
                    val = s
                    for part in parts:
                        val = val.get(part, {}) if isinstance(val, dict) else 0
                else:
                    val = s.get(key, 0)
                values.append(val)

            best_idx = values.index(max(values) if higher_better else min(values))

            comparison['metrics_comparison'][label] = {
                'values': values,
                'best': strategies[best_idx].get('id')
            }

        # Determina vencedor geral (por score composto)
        scores = [s.get('composite_score', 0) for s in strategies]
        winner_idx = scores.index(max(scores))
        comparison['winner'] = strategies[winner_idx].get('id')

        return comparison

    def print_summary(self):
        """Imprime resumo do storage."""
        stats = self.get_statistics()

        print("\n" + "=" * 60)
        print("STRATEGY STORAGE SUMMARY")
        print("=" * 60)
        print(f"Total Strategies:     {stats['total_strategies']}")
        print(f"Robust Strategies:    {stats['robust_strategies']}")
        print(f"Average Score:        {stats['avg_score']:.2f}")
        print(f"Best Score:           {stats['best_score']:.2f}")
        print(f"Average Sharpe:       {stats['avg_sharpe']:.4f}")
        print(f"Average Profit Factor: {stats['avg_profit_factor']:.4f}")

        if self.strategies:
            print("\nTop 5 Strategies:")
            print("-" * 60)
            for i, s in enumerate(self.get_top_strategies(5, robust_only=False), 1):
                robust = "R" if s.get('robustness', {}).get('is_robust') else " "
                print(f"{i}. [{robust}] {s.get('id')} - {s.get('strategy_name')}")
                print(f"   Score: {s.get('composite_score', 0):.2f} | "
                      f"Sharpe: {s.get('performance', {}).get('sharpe_ratio', 0):.4f} | "
                      f"PF: {s.get('performance', {}).get('profit_factor', 0):.2f}")

        print("=" * 60 + "\n")


# ============================================================
# Funcoes auxiliares de conveniencia
# ============================================================

def get_storage(storage_dir: str = "wfo/strategies") -> StrategyStorage:
    """Retorna instancia do storage."""
    return StrategyStorage(storage_dir)


def save_best_strategy(
    metrics: StrategyMetrics,
    storage_dir: str = "wfo/strategies"
) -> str:
    """Salva estrategia se for a melhor."""
    storage = get_storage(storage_dir)
    return storage.save_strategy(metrics)


def get_production_params(storage_dir: str = "wfo/strategies") -> Optional[Dict]:
    """Retorna parametros da melhor estrategia para producao."""
    storage = get_storage(storage_dir)
    best = storage.get_best_strategy()

    if best:
        return best.get('params', {})
    return None


def export_best_for_production(
    output_file: str = "config/best_strategy.json",
    storage_dir: str = "wfo/strategies"
) -> bool:
    """Exporta melhor estrategia para producao."""
    storage = get_storage(storage_dir)
    return storage.export_for_production(output_file=output_file)
