"""
Gestão de Portfolio e Margem
============================
Gerencia alocação de margem entre posições,
rotação baseada em score, e otimização de capital.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import os

from .scoring import ScoringSystem, SignalScore


@dataclass
class PortfolioPosition:
    """Posição no portfolio."""
    symbol: str
    side: str
    entry_price: float
    quantity: float
    notional: float
    margin_used: float
    leverage: int
    score: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    unrealized_pnl: float = 0.0
    
    @property
    def margin_pct(self) -> float:
        """Percentual da margem total usada."""
        return self.margin_used


@dataclass
class PortfolioState:
    """Estado atual do portfolio."""
    total_balance: float
    available_margin: float
    used_margin: float
    margin_used_pct: float
    positions: List[PortfolioPosition]
    total_unrealized_pnl: float
    
    @property
    def can_open_new(self) -> bool:
        """Pode abrir nova posição?"""
        return self.margin_used_pct < 90  # Máximo 90%


class PortfolioManager:
    """
    Gerenciador de Portfolio com otimização de margem.
    
    Funcionalidades:
    - Alocação dinâmica de margem
    - Rotação de posições baseada em score
    - Limite de exposição por ativo
    - Diversificação automática
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.scoring = ScoringSystem()
        
        # Estado
        self.positions: Dict[str, PortfolioPosition] = {}
        self.pending_signals: List[SignalScore] = []
    
    def _default_config(self) -> Dict:
        return {
            # Limites de margem
            'max_margin_usage': 0.90,      # 90% máximo
            'min_margin_buffer': 0.10,     # 10% buffer mínimo
            'target_margin_usage': 0.80,   # 80% ideal
            
            # Limites por posição
            'max_position_pct': 0.25,      # 25% máximo por posição
            'min_position_pct': 0.05,      # 5% mínimo por posição
            
            # Diversificação
            'max_positions': 10,           # Máximo de posições
            'min_positions': 3,            # Mínimo para diversificar
            'max_correlation': 0.7,        # Máxima correlação permitida
            
            # Score
            'min_score_to_open': 5.0,
            'min_score_to_replace': 2.0,
            
            # Leverage
            'default_leverage': 5,
            'max_leverage': 20,
            'leverage_by_score': True,     # Maior score = maior leverage
        }
    
    def calculate_position_size(
        self,
        signal: SignalScore,
        available_margin: float,
        total_balance: float,
        current_positions: int = 0
    ) -> Tuple[float, int]:
        """
        Calcular tamanho da posição e leverage.
        
        Returns:
            (notional, leverage)
        """
        config = self.config
        
        # Margem máxima para esta posição
        max_pct = config['max_position_pct']
        min_pct = config['min_position_pct']
        
        # Ajustar pelo número de posições desejadas
        target_positions = config['max_positions']
        if current_positions < config['min_positions']:
            # Fase inicial: posições menores para diversificar
            position_pct = min_pct + (max_pct - min_pct) * 0.3
        else:
            # Ajustar baseado no score
            score_factor = signal.score / 10.0
            position_pct = min_pct + (max_pct - min_pct) * score_factor
        
        # Margem para esta posição
        position_margin = total_balance * position_pct
        position_margin = min(position_margin, available_margin * 0.5)  # Não usar mais que 50% do disponível
        
        # Leverage baseado no score e R:R
        if config['leverage_by_score']:
            base_lev = config['default_leverage']
            max_lev = config['max_leverage']
            
            # Score alto + R:R bom = mais leverage
            score_mult = signal.score / 7.0  # Score 7+ = multiplicador > 1
            rr_mult = min(signal.risk_reward / 2.0, 1.5)  # R:R 2+ = multiplicador > 1
            
            leverage = int(base_lev * score_mult * rr_mult)
            leverage = max(1, min(leverage, max_lev))
        else:
            leverage = config['default_leverage']
        
        # Notional = margin * leverage
        notional = position_margin * leverage
        
        return notional, leverage
    
    def should_replace_position(
        self,
        current_pos: PortfolioPosition,
        new_signal: SignalScore,
        current_price: float,
        atr: float
    ) -> bool:
        """Verificar se deve substituir posição por novo sinal."""
        # Calcular score atual da posição
        current_score = self.scoring.calculate_position_score(
            {
                'entry_price': current_pos.entry_price,
                'side': current_pos.side,
                'stop_loss': current_pos.stop_loss,
                'take_profit': current_pos.take_profit
            },
            current_price,
            atr
        )
        
        # Atualizar score da posição
        current_pos.score = current_score
        
        # Comparar scores
        min_diff = self.config['min_score_to_replace']
        return new_signal.score - current_score >= min_diff
    
    def get_positions_to_replace(
        self,
        new_signals: List[SignalScore],
        current_prices: Dict[str, float],
        atrs: Dict[str, float]
    ) -> List[Tuple[str, SignalScore]]:
        """
        Identificar posições que devem ser substituídas.
        
        Returns:
            Lista de (symbol_to_close, new_signal)
        """
        if not self.positions or not new_signals:
            return []
        
        replacements = []
        
        # Ordenar posições por score (menor primeiro)
        sorted_positions = sorted(
            self.positions.values(),
            key=lambda p: p.score
        )
        
        # Ordenar novos sinais por score (maior primeiro)
        sorted_signals = sorted(new_signals, key=lambda s: s.score, reverse=True)
        
        for signal in sorted_signals:
            # Não substituir se já tem posição no mesmo símbolo
            if signal.symbol in self.positions:
                continue
            
            # Verificar posições que podem ser substituídas
            for pos in sorted_positions:
                if pos.symbol in [r[0] for r in replacements]:
                    continue  # Já marcada para substituição
                
                current_price = current_prices.get(pos.symbol, pos.entry_price)
                atr = atrs.get(pos.symbol, current_price * 0.02)
                
                if self.should_replace_position(pos, signal, current_price, atr):
                    replacements.append((pos.symbol, signal))
                    break
        
        return replacements
    
    def optimize_allocation(
        self,
        total_balance: float,
        available_margin: float,
        signals: List[SignalScore]
    ) -> List[Dict]:
        """
        Otimizar alocação de margem entre sinais.
        
        Returns:
            Lista de alocações recomendadas
        """
        if not signals:
            return []
        
        # Filtrar sinais com score mínimo
        min_score = self.config['min_score_to_open']
        valid_signals = [s for s in signals if s.score >= min_score]
        
        if not valid_signals:
            return []
        
        # Ordenar por score
        sorted_signals = sorted(valid_signals, key=lambda s: s.score, reverse=True)
        
        # Limitar ao máximo de posições
        max_new = self.config['max_positions'] - len(self.positions)
        sorted_signals = sorted_signals[:max_new]
        
        # Calcular alocações
        allocations = []
        remaining_margin = available_margin
        
        for signal in sorted_signals:
            if remaining_margin < total_balance * self.config['min_position_pct']:
                break
            
            notional, leverage = self.calculate_position_size(
                signal,
                remaining_margin,
                total_balance,
                len(self.positions) + len(allocations)
            )
            
            margin_needed = notional / leverage
            
            if margin_needed <= remaining_margin:
                allocations.append({
                    'symbol': signal.symbol,
                    'side': signal.side,
                    'score': signal.score,
                    'notional': notional,
                    'margin': margin_needed,
                    'leverage': leverage,
                    'entry_price': signal.entry_price,
                    'stop_loss': signal.stop_loss,
                    'take_profit': signal.take_profit
                })
                remaining_margin -= margin_needed
        
        return allocations
    
    def get_portfolio_metrics(self) -> Dict:
        """Obter métricas do portfolio."""
        if not self.positions:
            return {
                'num_positions': 0,
                'total_notional': 0,
                'total_margin': 0,
                'avg_score': 0,
                'total_pnl': 0,
                'diversification': 0
            }
        
        positions = list(self.positions.values())
        
        return {
            'num_positions': len(positions),
            'total_notional': sum(p.notional for p in positions),
            'total_margin': sum(p.margin_used for p in positions),
            'avg_score': np.mean([p.score for p in positions]),
            'min_score': min(p.score for p in positions),
            'max_score': max(p.score for p in positions),
            'total_pnl': sum(p.unrealized_pnl for p in positions),
            'diversification': len(set(p.symbol.split('/')[0] for p in positions))
        }
    
    def suggest_rebalance(
        self,
        current_prices: Dict[str, float],
        atrs: Dict[str, float]
    ) -> List[Dict]:
        """
        Sugerir rebalanceamento do portfolio.
        
        Returns:
            Lista de ações sugeridas
        """
        suggestions = []
        
        if not self.positions:
            return suggestions
        
        for symbol, pos in self.positions.items():
            current_price = current_prices.get(symbol, pos.entry_price)
            atr = atrs.get(symbol, current_price * 0.02)
            
            # Atualizar score
            pos.score = self.scoring.calculate_position_score(
                {
                    'entry_price': pos.entry_price,
                    'side': pos.side,
                    'stop_loss': pos.stop_loss,
                    'take_profit': pos.take_profit
                },
                current_price,
                atr
            )
            
            # Calcular PnL
            if pos.side == 'long':
                pnl_pct = (current_price - pos.entry_price) / pos.entry_price
            else:
                pnl_pct = (pos.entry_price - current_price) / pos.entry_price
            
            pos.unrealized_pnl = pnl_pct * pos.notional
            
            # Sugestões baseadas em score
            if pos.score < 3:
                suggestions.append({
                    'action': 'CLOSE',
                    'symbol': symbol,
                    'reason': f'Score muito baixo ({pos.score:.1f})',
                    'urgency': 'high'
                })
            elif pos.score < 5:
                suggestions.append({
                    'action': 'REDUCE',
                    'symbol': symbol,
                    'reason': f'Score abaixo do ideal ({pos.score:.1f})',
                    'urgency': 'medium'
                })
        
        return suggestions
    
    def save_state(self, filepath: str):
        """Salvar estado do portfolio."""
        state = {
            'timestamp': datetime.now().isoformat(),
            'positions': {
                sym: {
                    'symbol': p.symbol,
                    'side': p.side,
                    'entry_price': p.entry_price,
                    'quantity': p.quantity,
                    'notional': p.notional,
                    'margin_used': p.margin_used,
                    'leverage': p.leverage,
                    'score': p.score,
                    'stop_loss': p.stop_loss,
                    'take_profit': p.take_profit,
                    'unrealized_pnl': p.unrealized_pnl
                }
                for sym, p in self.positions.items()
            },
            'metrics': self.get_portfolio_metrics()
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    def load_state(self, filepath: str):
        """Carregar estado do portfolio."""
        if not os.path.exists(filepath):
            return
        
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            for sym, data in state.get('positions', {}).items():
                self.positions[sym] = PortfolioPosition(
                    symbol=data['symbol'],
                    side=data['side'],
                    entry_price=data['entry_price'],
                    quantity=data['quantity'],
                    notional=data['notional'],
                    margin_used=data['margin_used'],
                    leverage=data['leverage'],
                    score=data['score'],
                    entry_time=datetime.fromisoformat(data.get('entry_time', datetime.now().isoformat())),
                    stop_loss=data['stop_loss'],
                    take_profit=data['take_profit'],
                    unrealized_pnl=data.get('unrealized_pnl', 0)
                )
        except Exception as e:
            print(f"Erro carregando portfolio: {e}")
