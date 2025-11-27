"""
Performance Metrics Module
High-Performance Trading Analytics
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class TradeResult:
    """Single trade result."""
    entry_time: str
    exit_time: str
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    duration_minutes: float
    reason: str


class PerformanceMetrics:
    """Calculate comprehensive trading performance metrics."""
    
    def __init__(self, trades: List[Dict], initial_balance: float = 10000.0,
                 risk_free_rate: float = 0.0, periods_per_year: int = 252 * 24 * 12):
        """
        Initialize metrics calculator.
        
        Args:
            trades: List of trade dictionaries
            initial_balance: Starting balance
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year (default: 5min candles)
        """
        self.trades = trades
        self.initial_balance = initial_balance
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        
        self._process_trades()
    
    def _process_trades(self):
        """Process trades into returns series."""
        if not self.trades:
            self.returns = pd.Series([0.0])
            self.equity_curve = pd.Series([self.initial_balance])
            return
        
        pnls = [t.get('pnl', 0) for t in self.trades]
        self.equity_curve = pd.Series(
            np.cumsum([self.initial_balance] + pnls)
        )
        
        # Calculate returns
        self.returns = self.equity_curve.pct_change().dropna()
        if len(self.returns) == 0:
            self.returns = pd.Series([0.0])
    
    def calculate_all(self) -> Dict[str, float]:
        """Calculate all performance metrics."""
        return {
            # Returns
            'total_return': self.total_return(),
            'total_return_pct': self.total_return_pct(),
            'annualized_return': self.annualized_return(),
            
            # Risk-Adjusted Returns
            'sharpe_ratio': self.sharpe_ratio(),
            'sortino_ratio': self.sortino_ratio(),
            'calmar_ratio': self.calmar_ratio(),
            
            # Risk Metrics
            'max_drawdown': self.max_drawdown(),
            'max_drawdown_duration': self.max_drawdown_duration(),
            'volatility': self.volatility(),
            'downside_deviation': self.downside_deviation(),
            'var_95': self.var(0.95),
            'cvar_95': self.cvar(0.95),
            
            # Trade Statistics
            'total_trades': self.total_trades(),
            'win_rate': self.win_rate(),
            'profit_factor': self.profit_factor(),
            'expectancy': self.expectancy(),
            'avg_win': self.avg_win(),
            'avg_loss': self.avg_loss(),
            'win_loss_ratio': self.win_loss_ratio(),
            'largest_win': self.largest_win(),
            'largest_loss': self.largest_loss(),
            'avg_trade_duration': self.avg_trade_duration(),
            
            # Consistency
            'consecutive_wins': self.max_consecutive_wins(),
            'consecutive_losses': self.max_consecutive_losses(),
            'recovery_factor': self.recovery_factor(),
            
            # Final
            'final_balance': self.final_balance(),
        }
    
    def total_return(self) -> float:
        """Total absolute return."""
        return self.final_balance() - self.initial_balance
    
    def total_return_pct(self) -> float:
        """Total percentage return."""
        return (self.final_balance() / self.initial_balance - 1) * 100
    
    def final_balance(self) -> float:
        """Final account balance."""
        return self.equity_curve.iloc[-1] if len(self.equity_curve) > 0 else self.initial_balance
    
    def annualized_return(self) -> float:
        """Annualized return."""
        if len(self.returns) < 2:
            return 0.0
        total_return = self.final_balance() / self.initial_balance
        n_periods = len(self.returns)
        if n_periods == 0:
            return 0.0
        return (total_return ** (self.periods_per_year / n_periods) - 1) * 100
    
    def sharpe_ratio(self) -> float:
        """
        Sharpe Ratio = (Return - Risk Free Rate) / Volatility
        """
        if len(self.returns) < 2:
            return 0.0
        
        excess_returns = self.returns.mean() - (self.risk_free_rate / self.periods_per_year)
        std = self.returns.std()
        
        if std == 0:
            return 0.0
        
        return (excess_returns / std) * np.sqrt(self.periods_per_year)
    
    def sortino_ratio(self) -> float:
        """
        Sortino Ratio = (Return - Risk Free Rate) / Downside Deviation
        Only penalizes downside volatility.
        """
        if len(self.returns) < 2:
            return 0.0
        
        excess_returns = self.returns.mean() - (self.risk_free_rate / self.periods_per_year)
        downside = self.downside_deviation()
        
        if downside == 0:
            return 0.0
        
        return (excess_returns / downside) * np.sqrt(self.periods_per_year)
    
    def calmar_ratio(self) -> float:
        """
        Calmar Ratio = Annualized Return / Max Drawdown
        """
        max_dd = self.max_drawdown()
        if max_dd == 0:
            return 0.0
        return self.annualized_return() / (max_dd * 100)
    
    def max_drawdown(self) -> float:
        """Maximum drawdown as decimal."""
        if len(self.equity_curve) < 2:
            return 0.0
        
        peak = self.equity_curve.expanding(min_periods=1).max()
        drawdown = (self.equity_curve - peak) / peak
        return abs(drawdown.min())
    
    def max_drawdown_duration(self) -> int:
        """Maximum drawdown duration in periods."""
        if len(self.equity_curve) < 2:
            return 0
        
        peak = self.equity_curve.expanding(min_periods=1).max()
        in_drawdown = self.equity_curve < peak
        
        max_duration = 0
        current_duration = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        
        return max_duration
    
    def volatility(self) -> float:
        """Annualized volatility."""
        if len(self.returns) < 2:
            return 0.0
        return self.returns.std() * np.sqrt(self.periods_per_year)
    
    def downside_deviation(self) -> float:
        """Downside deviation (only negative returns)."""
        if len(self.returns) < 2:
            return 0.0
        negative_returns = self.returns[self.returns < 0]
        if len(negative_returns) == 0:
            return 0.0
        return negative_returns.std()
    
    def var(self, confidence: float = 0.95) -> float:
        """Value at Risk."""
        if len(self.returns) < 2:
            return 0.0
        return -np.percentile(self.returns, (1 - confidence) * 100)
    
    def cvar(self, confidence: float = 0.95) -> float:
        """Conditional Value at Risk (Expected Shortfall)."""
        if len(self.returns) < 2:
            return 0.0
        var = self.var(confidence)
        return -self.returns[self.returns <= -var].mean() if len(self.returns[self.returns <= -var]) > 0 else var
    
    def total_trades(self) -> int:
        """Total number of trades."""
        return len(self.trades)
    
    def win_rate(self) -> float:
        """Win rate as percentage."""
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.get('pnl', 0) > 0)
        return (wins / len(self.trades)) * 100
    
    def profit_factor(self) -> float:
        """Profit Factor = Gross Profit / Gross Loss."""
        if not self.trades:
            return 0.0
        
        gross_profit = sum(t['pnl'] for t in self.trades if t.get('pnl', 0) > 0)
        gross_loss = abs(sum(t['pnl'] for t in self.trades if t.get('pnl', 0) < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def expectancy(self) -> float:
        """
        Expected value per trade.
        E = (Win% * Avg Win) - (Loss% * Avg Loss)
        """
        if not self.trades:
            return 0.0
        
        wins = [t['pnl'] for t in self.trades if t.get('pnl', 0) > 0]
        losses = [abs(t['pnl']) for t in self.trades if t.get('pnl', 0) < 0]
        
        win_rate = len(wins) / len(self.trades)
        loss_rate = len(losses) / len(self.trades)
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        return (win_rate * avg_win) - (loss_rate * avg_loss)
    
    def avg_win(self) -> float:
        """Average winning trade."""
        wins = [t['pnl'] for t in self.trades if t.get('pnl', 0) > 0]
        return np.mean(wins) if wins else 0.0
    
    def avg_loss(self) -> float:
        """Average losing trade (positive number)."""
        losses = [abs(t['pnl']) for t in self.trades if t.get('pnl', 0) < 0]
        return np.mean(losses) if losses else 0.0
    
    def win_loss_ratio(self) -> float:
        """Average Win / Average Loss."""
        avg_l = self.avg_loss()
        if avg_l == 0:
            return 0.0
        return self.avg_win() / avg_l
    
    def largest_win(self) -> float:
        """Largest winning trade."""
        wins = [t['pnl'] for t in self.trades if t.get('pnl', 0) > 0]
        return max(wins) if wins else 0.0
    
    def largest_loss(self) -> float:
        """Largest losing trade (positive number)."""
        losses = [abs(t['pnl']) for t in self.trades if t.get('pnl', 0) < 0]
        return max(losses) if losses else 0.0
    
    def avg_trade_duration(self) -> float:
        """Average trade duration in minutes."""
        durations = [t.get('duration_minutes', 0) for t in self.trades]
        return np.mean(durations) if durations else 0.0
    
    def max_consecutive_wins(self) -> int:
        """Maximum consecutive winning trades."""
        return self._max_consecutive(lambda t: t.get('pnl', 0) > 0)
    
    def max_consecutive_losses(self) -> int:
        """Maximum consecutive losing trades."""
        return self._max_consecutive(lambda t: t.get('pnl', 0) < 0)
    
    def _max_consecutive(self, condition) -> int:
        """Helper for consecutive count."""
        if not self.trades:
            return 0
        
        max_count = 0
        current_count = 0
        
        for trade in self.trades:
            if condition(trade):
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        
        return max_count
    
    def recovery_factor(self) -> float:
        """Total Return / Max Drawdown."""
        max_dd = self.max_drawdown()
        if max_dd == 0:
            return 0.0
        return self.total_return() / (max_dd * self.initial_balance)
    
    def kelly_criterion(self) -> float:
        """
        Kelly Criterion for optimal position sizing.
        K = W - (1-W)/R where W=win rate, R=win/loss ratio
        """
        w = self.win_rate() / 100
        r = self.win_loss_ratio()
        
        if r == 0:
            return 0.0
        
        kelly = w - ((1 - w) / r)
        return max(0, min(kelly, 0.25))  # Cap at 25%


def compare_strategies(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Compare multiple strategy results.
    
    Args:
        results: Dict of strategy_name -> metrics dict
    
    Returns:
        DataFrame with comparison
    """
    df = pd.DataFrame(results).T
    
    # Rank by key metrics
    rank_cols = ['sharpe_ratio', 'sortino_ratio', 'profit_factor', 'win_rate']
    
    for col in rank_cols:
        if col in df.columns:
            df[f'{col}_rank'] = df[col].rank(ascending=False)
    
    return df.sort_values('sharpe_ratio', ascending=False)


def validate_strategy(metrics: Dict[str, float], thresholds: Dict[str, float]) -> Dict[str, Any]:
    """
    Validate strategy against thresholds.
    
    Returns:
        Dict with 'passed' (bool) and 'failures' (list of failed criteria)
    """
    failures = []
    
    checks = [
        ('sharpe_ratio', 'min_sharpe', '>='),
        ('sortino_ratio', 'min_sortino', '>='),
        ('calmar_ratio', 'min_calmar', '>='),
        ('profit_factor', 'min_profit_factor', '>='),
        ('win_rate', 'min_win_rate', '>='),
        ('max_drawdown', 'max_drawdown', '<='),
        ('total_trades', 'min_trades', '>='),
        ('expectancy', 'min_expectancy', '>='),
    ]
    
    for metric_key, threshold_key, op in checks:
        if metric_key in metrics and threshold_key in thresholds:
            metric_val = metrics[metric_key]
            threshold_val = thresholds[threshold_key]
            
            if op == '>=' and metric_val < threshold_val:
                failures.append(f"{metric_key}: {metric_val:.2f} < {threshold_val:.2f}")
            elif op == '<=' and metric_val > threshold_val:
                failures.append(f"{metric_key}: {metric_val:.2f} > {threshold_val:.2f}")
    
    return {
        'passed': len(failures) == 0,
        'failures': failures,
        'score': _calculate_composite_score(metrics)
    }


def _calculate_composite_score(metrics: Dict[str, float]) -> float:
    """Calculate composite score for ranking strategies."""
    weights = {
        'sharpe_ratio': 0.25,
        'sortino_ratio': 0.20,
        'profit_factor': 0.15,
        'win_rate': 0.10,
        'expectancy': 0.15,
        'calmar_ratio': 0.10,
        'recovery_factor': 0.05,
    }
    
    score = 0.0
    for metric, weight in weights.items():
        if metric in metrics:
            val = metrics[metric]
            # Normalize each metric
            if metric == 'sharpe_ratio':
                normalized = min(val / 3.0, 1.0)  # Cap at 3.0
            elif metric == 'sortino_ratio':
                normalized = min(val / 4.0, 1.0)
            elif metric == 'profit_factor':
                normalized = min((val - 1) / 2.0, 1.0)  # 1-3 range
            elif metric == 'win_rate':
                normalized = val / 100  # Already percentage
            elif metric == 'expectancy':
                normalized = min(val / 100, 1.0)  # Per $100 risked
            elif metric == 'calmar_ratio':
                normalized = min(val / 2.0, 1.0)
            elif metric == 'recovery_factor':
                normalized = min(val / 5.0, 1.0)
            else:
                normalized = 0.5
            
            score += normalized * weight
    
    return score * 100  # Return as 0-100 score
