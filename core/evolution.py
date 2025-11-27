"""
Sistema de Evolução e Armazenamento de Estratégias Validadas
============================================================
Gerencia estratégias validadas por WFO, armazena histórico
e fornece interface para dashboard em tempo real.
"""
import os
import json
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import hashlib


@dataclass
class FoldResult:
    """Resultado de um fold individual do WFO."""
    fold_number: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    profit_factor: float
    total_trades: int
    win_rate: float
    avg_trade_pnl: float


@dataclass
class ValidatedStrategy:
    """Estratégia validada por WFO com todos os detalhes."""
    id: str
    name: str
    strategy_type: str
    symbols: List[str]
    timeframe: str

    # Parâmetros
    params: Dict

    # Métricas agregadas
    avg_return_pct: float
    min_return_pct: float
    max_return_pct: float
    std_return_pct: float
    avg_drawdown_pct: float
    max_drawdown_pct: float
    avg_sharpe: float
    avg_sortino: float
    avg_profit_factor: float
    avg_win_rate: float
    total_trades: int

    # Detalhes dos folds
    num_folds: int
    fold_results: List[Dict]

    # Score e status
    wfo_score: float
    robustness_score: float
    is_active: bool

    # Timestamps
    created_at: str
    validated_at: str
    last_used: Optional[str]

    # Performance real (se disponível)
    real_trades: int = 0
    real_pnl: float = 0.0
    real_win_rate: float = 0.0


class StrategyStorage:
    """
    Armazenamento persistente de estratégias validadas.

    Arquivos:
    - state/validated_strategies.json: Lista de todas estratégias validadas
    - state/active_strategy.json: Estratégia atualmente ativa
    - state/baselines/: Histórico de baselines
    """

    def __init__(self, base_path: str = "state"):
        self.base_path = base_path
        self.strategies_file = os.path.join(base_path, "validated_strategies.json")
        self.active_file = os.path.join(base_path, "active_strategy.json")
        self.baselines_dir = os.path.join(base_path, "baselines")

        # Criar diretórios se não existirem
        os.makedirs(self.baselines_dir, exist_ok=True)

        # Cache de estratégias
        self._strategies: Dict[str, ValidatedStrategy] = {}
        self._load_strategies()

    def _load_strategies(self):
        """Carregar estratégias do arquivo."""
        if os.path.exists(self.strategies_file):
            try:
                with open(self.strategies_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for s in data.get('strategies', []):
                        try:
                            # Converter formato antigo para novo se necessário
                            normalized = self._normalize_strategy_data(s)
                            strategy = ValidatedStrategy(**normalized)
                            self._strategies[strategy.id] = strategy
                        except Exception as e:
                            print(f"Erro carregando estratégia {s.get('id', 'unknown')}: {e}")
            except Exception as e:
                print(f"Erro carregando arquivo de estratégias: {e}")

    def _normalize_strategy_data(self, data: Dict) -> Dict:
        """
        Normaliza dados de estratégia para o formato esperado.

        Lida com formatos antigos que usam 'metrics' como objeto aninhado.
        """
        result = data.copy()

        # Se tem 'metrics' como objeto, extrair campos individuais
        if 'metrics' in result and isinstance(result['metrics'], dict):
            metrics = result.pop('metrics')

            # Mapear campos de metrics para campos do dataclass
            field_mapping = {
                'return_pct': 'avg_return_pct',
                'avg_return': 'avg_return_pct',
                'sharpe_ratio': 'avg_sharpe',
                'avg_sharpe': 'avg_sharpe',
                'max_drawdown_pct': 'max_drawdown_pct',
                'max_drawdown': 'max_drawdown_pct',
                'win_rate': 'avg_win_rate',
                'avg_win_rate': 'avg_win_rate',
                'total_trades': 'total_trades',
                'profit_factor': 'avg_profit_factor',
                'avg_profit_factor': 'avg_profit_factor',
            }

            for old_key, new_key in field_mapping.items():
                if old_key in metrics and new_key not in result:
                    result[new_key] = metrics[old_key]

        # Garantir campos obrigatórios com valores padrão
        defaults = {
            'strategy_type': result.get('params', {}).get('strategy', 'unknown'),
            'symbols': ['BTCUSDT'],
            'timeframe': '1h',
            'params': {},
            'avg_return_pct': 0.0,
            'min_return_pct': 0.0,
            'max_return_pct': 0.0,
            'std_return_pct': 0.0,
            'avg_drawdown_pct': 0.0,
            'max_drawdown_pct': 0.0,
            'avg_sharpe': 0.0,
            'avg_sortino': 0.0,
            'avg_profit_factor': 0.0,
            'avg_win_rate': 0.0,
            'total_trades': 0,
            'num_folds': 1,
            'fold_results': [],
            'wfo_score': 0.0,
            'robustness_score': 0.0,
            'is_active': False,
            'created_at': datetime.now().isoformat(),
            'validated_at': datetime.now().isoformat(),
            'last_used': None,
            'real_trades': 0,
            'real_pnl': 0.0,
            'real_win_rate': 0.0,
        }

        for key, default_value in defaults.items():
            if key not in result:
                result[key] = default_value

        # Remover campos extras que não existem no dataclass
        valid_fields = {
            'id', 'name', 'strategy_type', 'symbols', 'timeframe', 'params',
            'avg_return_pct', 'min_return_pct', 'max_return_pct', 'std_return_pct',
            'avg_drawdown_pct', 'max_drawdown_pct', 'avg_sharpe', 'avg_sortino',
            'avg_profit_factor', 'avg_win_rate', 'total_trades', 'num_folds',
            'fold_results', 'wfo_score', 'robustness_score', 'is_active',
            'created_at', 'validated_at', 'last_used', 'real_trades',
            'real_pnl', 'real_win_rate'
        }

        # Remover campos que não pertencem ao dataclass
        for key in list(result.keys()):
            if key not in valid_fields:
                del result[key]

        return result

    def _save_strategies(self):
        """Salvar estratégias no arquivo."""
        data = {
            'updated_at': datetime.now().isoformat(),
            'count': len(self._strategies),
            'strategies': [asdict(s) for s in self._strategies.values()]
        }

        with open(self.strategies_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _sync_to_trader_state(self, strategy_id: str):
        """
        Sincronizar estratégia ativa com trader_state.json.

        Isso garante que o bot use a mesma estratégia que está
        marcada como ativa no sistema de evolução.

        Atualiza tanto os novos campos de sync quanto o params.strategy
        para manter compatibilidade com o bot.
        """
        strategy = self._strategies.get(strategy_id)
        if not strategy:
            return

        trader_state_file = os.path.join(self.base_path, "trader_state.json")

        try:
            # Carregar estado atual do trader
            trader_state = {}
            if os.path.exists(trader_state_file):
                with open(trader_state_file, 'r', encoding='utf-8') as f:
                    trader_state = json.load(f)

            # Atualizar campos de sincronização (novo formato)
            trader_state['active_strategy'] = strategy.strategy_type
            trader_state['strategy_id'] = strategy.id
            trader_state['strategy_name'] = strategy.name
            trader_state['strategy_synced_at'] = datetime.now().isoformat()

            # Também atualizar params para manter compatibilidade com o bot
            if 'params' not in trader_state:
                trader_state['params'] = {}

            # Atualizar params com os parâmetros da estratégia
            trader_state['params']['strategy'] = strategy.strategy_type

            # Mesclar parâmetros da estratégia validada
            for key, value in strategy.params.items():
                if key != 'strategy':  # Já definimos acima
                    trader_state['params'][key] = value

            # Salvar
            with open(trader_state_file, 'w', encoding='utf-8') as f:
                json.dump(trader_state, f, indent=2, ensure_ascii=False)

            print(f"trader_state sincronizado com estratégia {strategy.name}")

        except Exception as e:
            print(f"Erro sincronizando trader_state: {e}")
            import traceback
            traceback.print_exc()

    def generate_id(self, strategy_type: str, params: Dict) -> str:
        """Gerar ID único para estratégia baseado em seus parâmetros."""
        # Hash dos parâmetros para criar ID único
        params_str = json.dumps(params, sort_keys=True)
        hash_obj = hashlib.md5(f"{strategy_type}_{params_str}".encode())
        return hash_obj.hexdigest()[:12]

    def add_strategy(
        self,
        strategy_type: str,
        symbols: List[str],
        timeframe: str,
        params: Dict,
        fold_results: List[Dict],
        wfo_score: float
    ) -> ValidatedStrategy:
        """
        Adicionar nova estratégia validada.

        Args:
            strategy_type: Tipo da estratégia (ex: 'stoch_extreme')
            symbols: Lista de símbolos usados
            timeframe: Timeframe (ex: '1h')
            params: Parâmetros da estratégia
            fold_results: Lista de resultados de cada fold
            wfo_score: Score do WFO

        Returns:
            ValidatedStrategy criada
        """
        # Calcular métricas agregadas
        returns = [f.get('return_pct', 0) for f in fold_results]
        drawdowns = [f.get('max_drawdown_pct', 0) for f in fold_results]
        sharpes = [f.get('sharpe_ratio', 0) for f in fold_results]
        sortinos = [f.get('sortino_ratio', 0) for f in fold_results]
        profit_factors = [f.get('profit_factor', 0) for f in fold_results]
        win_rates = [f.get('win_rate', 0) for f in fold_results]
        trades = [f.get('total_trades', 0) for f in fold_results]

        import numpy as np

        avg_return = np.mean(returns) if returns else 0
        min_return = np.min(returns) if returns else 0
        max_return = np.max(returns) if returns else 0
        std_return = np.std(returns) if returns else 0

        # Robustez: quanto menor a variância e maior a consistência, melhor
        # Estratégia robusta tem todos os folds positivos com baixa variância
        positive_folds = sum(1 for r in returns if r > 0)
        consistency = positive_folds / len(returns) if returns else 0

        # Score de robustez (0-100)
        if std_return > 0:
            cv = abs(std_return / avg_return) if avg_return != 0 else float('inf')
            robustness = max(0, min(100, (1 - cv) * consistency * 100))
        else:
            robustness = consistency * 100

        strategy_id = self.generate_id(strategy_type, params)
        now = datetime.now().isoformat()

        strategy = ValidatedStrategy(
            id=strategy_id,
            name=f"{strategy_type}_{timeframe}_{len(symbols)}sym",
            strategy_type=strategy_type,
            symbols=symbols,
            timeframe=timeframe,
            params=params,
            avg_return_pct=round(float(avg_return), 2),
            min_return_pct=round(float(min_return), 2),
            max_return_pct=round(float(max_return), 2),
            std_return_pct=round(float(std_return), 2),
            avg_drawdown_pct=round(float(np.mean(drawdowns)), 2) if drawdowns else 0,
            max_drawdown_pct=round(float(np.max(drawdowns)), 2) if drawdowns else 0,
            avg_sharpe=round(float(np.mean(sharpes)), 2) if sharpes else 0,
            avg_sortino=round(float(np.mean(sortinos)), 2) if sortinos else 0,
            avg_profit_factor=round(float(np.mean(profit_factors)), 2) if profit_factors else 0,
            avg_win_rate=round(float(np.mean(win_rates)), 1) if win_rates else 0,
            total_trades=sum(trades),
            num_folds=len(fold_results),
            fold_results=fold_results,
            wfo_score=round(wfo_score, 2),
            robustness_score=round(robustness, 1),
            is_active=False,
            created_at=now,
            validated_at=now,
            last_used=None
        )

        # Adicionar ou atualizar
        self._strategies[strategy_id] = strategy
        self._save_strategies()

        return strategy

    def get_strategy(self, strategy_id: str) -> Optional[ValidatedStrategy]:
        """Obter estratégia por ID."""
        return self._strategies.get(strategy_id)

    def get_all_strategies(self) -> List[ValidatedStrategy]:
        """Obter todas as estratégias ordenadas por score."""
        return sorted(
            self._strategies.values(),
            key=lambda s: s.wfo_score,
            reverse=True
        )

    def get_active_strategies(self) -> List[ValidatedStrategy]:
        """Obter estratégias ativas."""
        return [s for s in self._strategies.values() if s.is_active]

    def set_active(self, strategy_id: str, active: bool = True, sync_trader_state: bool = True):
        """
        Marcar estratégia como ativa/inativa.

        IMPORTANTE: Quando ativando uma estratégia, todas as outras são
        automaticamente desativadas para manter consistência.

        Args:
            strategy_id: ID da estratégia
            active: True para ativar, False para desativar
            sync_trader_state: Se True, sincroniza com trader_state.json
        """
        if strategy_id not in self._strategies:
            return

        # Se ativando, primeiro desativar TODAS as outras estratégias
        if active:
            for sid, strategy in self._strategies.items():
                if sid != strategy_id and strategy.is_active:
                    strategy.is_active = False

        # Agora ativar/desativar a estratégia solicitada
        self._strategies[strategy_id].is_active = active
        if active:
            self._strategies[strategy_id].last_used = datetime.now().isoformat()

        self._save_strategies()

        # Sincronizar com trader_state.json para manter consistência global
        if active and sync_trader_state:
            self._sync_to_trader_state(strategy_id)

    def update_real_performance(
        self,
        strategy_id: str,
        trades: int,
        pnl: float,
        win_rate: float
    ):
        """Atualizar performance real da estratégia."""
        if strategy_id in self._strategies:
            s = self._strategies[strategy_id]
            s.real_trades = trades
            s.real_pnl = pnl
            s.real_win_rate = win_rate
            self._save_strategies()

    def get_best_strategy(self) -> Optional[ValidatedStrategy]:
        """Obter melhor estratégia por score."""
        strategies = self.get_all_strategies()
        return strategies[0] if strategies else None

    def export_for_dashboard(self) -> Dict:
        """Exportar dados formatados para dashboard."""
        strategies = self.get_all_strategies()
        active = self.get_active_strategies()

        return {
            'updated_at': datetime.now().isoformat(),
            'total_strategies': len(strategies),
            'active_count': len(active),
            'strategies': [
                {
                    'id': s.id,
                    'name': s.name,
                    'strategy_type': s.strategy_type,
                    'symbols': s.symbols,
                    'timeframe': s.timeframe,
                    'params': s.params,
                    'metrics': {
                        'avg_return': s.avg_return_pct,
                        'min_return': s.min_return_pct,
                        'max_return': s.max_return_pct,
                        'std_return': s.std_return_pct,
                        'avg_drawdown': s.avg_drawdown_pct,
                        'max_drawdown': s.max_drawdown_pct,
                        'avg_sharpe': s.avg_sharpe,
                        'avg_sortino': s.avg_sortino,
                        'avg_profit_factor': s.avg_profit_factor,
                        'avg_win_rate': s.avg_win_rate,
                        'total_trades': s.total_trades,
                    },
                    'wfo': {
                        'num_folds': s.num_folds,
                        'wfo_score': s.wfo_score,
                        'robustness': s.robustness_score,
                        'fold_results': s.fold_results,
                    },
                    'status': {
                        'is_active': s.is_active,
                        'created_at': s.created_at,
                        'validated_at': s.validated_at,
                        'last_used': s.last_used,
                    },
                    'real_performance': {
                        'trades': s.real_trades,
                        'pnl': s.real_pnl,
                        'win_rate': s.real_win_rate,
                    }
                }
                for s in strategies
            ],
            'active_strategies': [
                {'id': s.id, 'name': s.name, 'strategy_type': s.strategy_type}
                for s in active
            ]
        }


    def sync_from_current_best(self) -> Optional[str]:
        """
        Sincronizar current_best.json com o sistema de estratégias.

        NÃO cria novas estratégias - apenas marca a estratégia ativa existente
        com os parâmetros mais recentes. Para adicionar novas estratégias,
        use add_strategy() diretamente após validação WFO.

        Returns:
            ID da estratégia ativa ou None se não houver estratégias
        """
        # Se já temos estratégias ativas, apenas retornar o ID
        active = self.get_active_strategies()
        if active:
            return active[0].id

        # Se não há ativas mas há estratégias, ativar a melhor
        all_strategies = self.get_all_strategies()
        if all_strategies:
            best = all_strategies[0]  # Já ordenadas por wfo_score
            self.set_active(best.id, True)
            return best.id

        return None

    def get_sync_status(self) -> Dict:
        """
        Obter status de sincronização entre todas as fontes de dados.

        Retorna informações sobre consistência entre:
        - validated_strategies.json
        - current_best.json
        - trader_state.json

        Returns:
            Dict com status de sincronização
        """
        status = {
            'timestamp': datetime.now().isoformat(),
            'is_synced': True,
            'sources': {},
            'issues': []
        }

        # 1. Verificar validated_strategies
        active_strategies = self.get_active_strategies()
        status['sources']['validated_strategies'] = {
            'file': self.strategies_file,
            'exists': os.path.exists(self.strategies_file),
            'total_count': len(self._strategies),
            'active_count': len(active_strategies),
            'active_ids': [s.id for s in active_strategies],
            'active_names': [s.name for s in active_strategies]
        }

        if len(active_strategies) > 1:
            status['is_synced'] = False
            status['issues'].append(f"Múltiplas estratégias ativas ({len(active_strategies)})")

        # 2. Verificar current_best.json
        current_best_file = os.path.join(self.base_path, "current_best.json")
        if os.path.exists(current_best_file):
            try:
                with open(current_best_file, 'r', encoding='utf-8') as f:
                    best = json.load(f)

                # Extrair tipo de estratégia (pode estar em diferentes lugares)
                best_params = best.get('params', {})
                best_type = best_params.get('strategy', best.get('strategy', 'unknown'))

                # Se ainda não achou, tentar extrair do strategy_name
                if best_type == 'unknown':
                    strategy_name = best.get('strategy_name', '')
                    if 'combined' in strategy_name.lower():
                        best_type = 'combined'
                    elif 'stoch' in strategy_name.lower():
                        best_type = 'stoch_extreme'

                best_id = self.generate_id(best_type, best_params)
                status['sources']['current_best'] = {
                    'file': current_best_file,
                    'exists': True,
                    'strategy_type': best_type,
                    'strategy_name': best.get('strategy_name', best_type),
                    'expected_id': best_id,
                    'updated_at': best.get('activated_at', best.get('updated_at', 'unknown'))
                }

                # Verificar se current_best está ativo
                if best_id not in [s.id for s in active_strategies]:
                    status['is_synced'] = False
                    status['issues'].append(f"current_best ({best_type}) não está marcado como ativo")
            except Exception as e:
                status['sources']['current_best'] = {'exists': True, 'error': str(e)}
        else:
            status['sources']['current_best'] = {'file': current_best_file, 'exists': False}

        # 3. Verificar trader_state.json
        trader_state_file = os.path.join(self.base_path, "trader_state.json")
        if os.path.exists(trader_state_file):
            try:
                with open(trader_state_file, 'r', encoding='utf-8') as f:
                    trader = json.load(f)

                # Campos podem estar em diferentes lugares:
                # - active_strategy (novo formato de sync)
                # - params.strategy (formato original do bot)
                trader_strategy = trader.get('active_strategy')
                if not trader_strategy:
                    # Fallback para params.strategy
                    trader_strategy = trader.get('params', {}).get('strategy', 'none')

                trader_id = trader.get('strategy_id', '')
                synced_at = trader.get('strategy_synced_at', 'never')

                status['sources']['trader_state'] = {
                    'file': trader_state_file,
                    'exists': True,
                    'active_strategy': trader_strategy,
                    'strategy_id': trader_id,
                    'synced_at': synced_at,
                    'using_legacy_format': not trader.get('active_strategy')
                }

                # Verificar se trader_state corresponde ao ativo
                if active_strategies:
                    active_strategy = active_strategies[0]
                    # Comparar pelo tipo se não temos ID de sync
                    if trader_id:
                        if trader_id != active_strategy.id:
                            status['is_synced'] = False
                            status['issues'].append(
                                f"trader_state ID ({trader_id}) diferente de ativo ({active_strategy.id})"
                            )
                    else:
                        # Sem ID, comparar pelo tipo de estratégia
                        if trader_strategy != active_strategy.strategy_type:
                            status['is_synced'] = False
                            status['issues'].append(
                                f"trader_state ({trader_strategy}) diferente de ativo ({active_strategy.strategy_type})"
                            )
            except Exception as e:
                status['sources']['trader_state'] = {'exists': True, 'error': str(e)}
        else:
            status['sources']['trader_state'] = {'file': trader_state_file, 'exists': False}

        return status

    def force_sync_all(self) -> Dict:
        """
        Forçar sincronização de todas as fontes de dados.

        Ordem de prioridade:
        1. Ler current_best.json como fonte de verdade
        2. Atualizar validated_strategies.json
        3. Atualizar trader_state.json

        Returns:
            Dict com resultado da sincronização
        """
        result = {
            'success': False,
            'synced_strategy_id': None,
            'actions': []
        }

        try:
            # Sincronizar a partir do current_best
            strategy_id = self.sync_from_current_best()
            if strategy_id:
                result['success'] = True
                result['synced_strategy_id'] = strategy_id
                result['actions'].append(f"Estratégia {strategy_id} sincronizada de current_best")

                # Verificar resultado
                sync_status = self.get_sync_status()
                result['final_status'] = sync_status

        except Exception as e:
            result['error'] = str(e)

        return result


# Instância global para uso
_storage: Optional[StrategyStorage] = None


def get_storage(reload: bool = False) -> StrategyStorage:
    """Obter instância global do storage.

    Args:
        reload: Se True, força recarregamento do arquivo
    """
    global _storage
    if _storage is None:
        _storage = StrategyStorage()
    elif reload:
        _storage._load_strategies()
    return _storage
