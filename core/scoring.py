"""
Sistema de Score para Posições
==============================
Calcula score de cada sinal/posição para:
- Decidir quais posições abrir
- Substituir posições fracas por fortes
- Otimizar uso de margem
"""
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class SignalScore:
    """Score completo de um sinal."""
    symbol: str
    side: str
    score: float
    components: Dict[str, float]
    
    # Métricas do sinal
    strength: float
    probability: float
    risk_reward: float
    volatility_adjusted: float
    
    # Preços
    entry_price: float
    stop_loss: float
    take_profit: float
    
    def __lt__(self, other):
        return self.score < other.score


class ScoringSystem:
    """
    Sistema de pontuação para sinais e posições.
    
    Score = (Strength * Probability * RiskReward * VolatilityFactor) / Risk
    
    Onde:
    - Strength: Força do sinal (0-10)
    - Probability: Probabilidade histórica de sucesso (0-1)
    - RiskReward: Ratio TP/SL
    - VolatilityFactor: Ajuste por volatilidade (ATR)
    - Risk: Risco da posição
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        
        # Cache de probabilidades históricas
        self.historical_probs: Dict[str, Dict[str, float]] = {}
    
    def _default_config(self) -> Dict:
        return {
            # Pesos dos componentes do score
            'weight_strength': 0.25,
            'weight_probability': 0.30,
            'weight_rr': 0.25,
            'weight_volatility': 0.20,
            
            # Limites
            'min_score_to_open': 5.0,
            'min_score_to_replace': 2.0,  # Diferença mínima para substituir
            
            # Ajustes
            'volatility_optimal': 0.02,  # ATR/price ideal (2%)
            'max_correlation_penalty': 0.3,  # Penalidade por correlação
        }
    
    def calculate_score(
        self,
        symbol: str,
        side: str,
        strength: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        atr: float,
        rsi: float = 50,
        volume_ratio: float = 1.0,
        historical_winrate: float = 0.5
    ) -> SignalScore:
        """
        Calcular score completo de um sinal.
        
        Args:
            symbol: Par de trading
            side: 'long' ou 'short'
            strength: Força do sinal (0-10)
            entry_price: Preço de entrada
            stop_loss: Stop loss
            take_profit: Take profit
            atr: Average True Range
            rsi: RSI atual
            volume_ratio: Volume atual / média
            historical_winrate: Win rate histórico do setup
            
        Returns:
            SignalScore com score calculado
        """
        # 1. Normalizar strength (0-10 -> 0-1)
        strength_norm = min(strength / 10.0, 1.0)
        
        # 2. Probabilidade baseada em histórico e indicadores
        probability = self._calculate_probability(
            side, rsi, volume_ratio, historical_winrate
        )
        
        # 3. Risk/Reward ratio
        if side == 'long':
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
        else:
            risk = abs(stop_loss - entry_price)
            reward = abs(entry_price - take_profit)
        
        risk_reward = (reward / risk) if risk > 0 else 0
        rr_score = min(risk_reward / 3.0, 1.0)  # Normalizar (3:1 = 1.0)
        
        # 4. Volatility factor (preferir volatilidade moderada)
        vol_ratio = (atr / entry_price) if entry_price > 0 else 0
        optimal_vol = self.config['volatility_optimal']
        
        if vol_ratio < optimal_vol * 0.5:
            vol_factor = vol_ratio / (optimal_vol * 0.5)  # Muito baixa
        elif vol_ratio > optimal_vol * 2:
            vol_factor = optimal_vol * 2 / vol_ratio  # Muito alta
        else:
            vol_factor = 1.0  # Ideal
        
        # 5. Calcular score final
        weights = self.config
        score = (
            strength_norm * weights['weight_strength'] +
            probability * weights['weight_probability'] +
            rr_score * weights['weight_rr'] +
            vol_factor * weights['weight_volatility']
        ) * 10  # Escalar para 0-10
        
        return SignalScore(
            symbol=symbol,
            side=side,
            score=round(score, 2),
            components={
                'strength': round(strength_norm, 3),
                'probability': round(probability, 3),
                'risk_reward': round(rr_score, 3),
                'volatility': round(vol_factor, 3)
            },
            strength=strength,
            probability=probability,
            risk_reward=risk_reward,
            volatility_adjusted=vol_factor,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
    
    def _calculate_probability(
        self,
        side: str,
        rsi: float,
        volume_ratio: float,
        historical_winrate: float
    ) -> float:
        """Calcular probabilidade de sucesso."""
        prob = historical_winrate
        
        # Ajustar por RSI
        if side == 'long':
            if rsi < 30:
                prob *= 1.2  # Oversold favorece long
            elif rsi > 70:
                prob *= 0.8  # Overbought desfavorece long
        else:
            if rsi > 70:
                prob *= 1.2  # Overbought favorece short
            elif rsi < 30:
                prob *= 0.8  # Oversold desfavorece short
        
        # Ajustar por volume
        if volume_ratio > 1.5:
            prob *= 1.1  # Alto volume = confirmação
        elif volume_ratio < 0.5:
            prob *= 0.9  # Baixo volume = menos confiável
        
        return min(max(prob, 0.1), 0.9)  # Limitar entre 10% e 90%
    
    def rank_signals(self, signals: List[SignalScore]) -> List[SignalScore]:
        """Ordenar sinais por score (maior primeiro)."""
        return sorted(signals, key=lambda x: x.score, reverse=True)
    
    def should_replace(
        self,
        current_position_score: float,
        new_signal_score: float
    ) -> bool:
        """Verificar se deve substituir posição por novo sinal."""
        min_diff = self.config['min_score_to_replace']
        return new_signal_score - current_position_score >= min_diff
    
    def calculate_position_score(
        self,
        position: Dict,
        current_price: float,
        atr: float
    ) -> float:
        """
        Calcular score atual de uma posição aberta.
        
        Considera:
        - Score original do sinal
        - PnL atual (momentum)
        - Distância para SL/TP
        - Tempo na posição
        """
        entry = position.get('entry_price', current_price)
        side = position.get('side', 'long')
        sl = position.get('stop_loss', 0)
        tp = position.get('take_profit', 0)
        
        # PnL atual
        if side == 'long':
            pnl_pct = (current_price - entry) / entry
        else:
            pnl_pct = (entry - current_price) / entry
        
        # Score base (simplificado para posições existentes)
        base_score = 5.0
        
        # Ajustar por PnL
        if pnl_pct > 0:
            base_score += min(pnl_pct * 100, 3)  # Máximo +3 pontos
        else:
            base_score += max(pnl_pct * 100, -3)  # Mínimo -3 pontos
        
        # Ajustar por proximidade do TP
        if tp > 0:
            if side == 'long':
                tp_distance = (tp - current_price) / current_price
            else:
                tp_distance = (current_price - tp) / current_price
            
            if tp_distance < 0.01:  # Muito perto do TP
                base_score += 2  # Manter para capturar lucro
        
        return max(0, min(10, base_score))
    
    def update_historical_prob(
        self,
        symbol: str,
        strategy: str,
        side: str,
        won: bool
    ):
        """Atualizar probabilidade histórica após trade."""
        key = f"{symbol}_{strategy}_{side}"
        
        if key not in self.historical_probs:
            self.historical_probs[key] = {'wins': 0, 'total': 0}
        
        self.historical_probs[key]['total'] += 1
        if won:
            self.historical_probs[key]['wins'] += 1
    
    def get_historical_winrate(
        self,
        symbol: str,
        strategy: str,
        side: str,
        default: float = 0.5
    ) -> float:
        """Obter win rate histórico de um setup."""
        key = f"{symbol}_{strategy}_{side}"
        
        if key in self.historical_probs:
            data = self.historical_probs[key]
            if data['total'] >= 10:  # Mínimo de trades para ser significativo
                return data['wins'] / data['total']
        
        return default
