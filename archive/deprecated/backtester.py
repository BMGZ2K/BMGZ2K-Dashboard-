"""
Backtester Module with Walk-Forward Optimization (WFO)
Professional-grade backtesting with anti-overfitting measures
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json

from .indicators import calculate_indicators
from .strategy import StrategyEngine, Signal
from .risk import RiskManager
from .metrics import PerformanceMetrics, validate_strategy


@dataclass
class BacktestResult:
    """Backtest result container."""
    trades: List[Dict]
    metrics: Dict[str, float]
    equity_curve: List[float]
    params: Dict[str, Any]
    period: str
    is_oos: bool = False  # Out-of-sample flag


class Backtester:
    """
    High-performance backtester with realistic execution simulation.
    """
    
    def __init__(
        self,
        params: Dict[str, Any],
        initial_balance: float = 10000.0,
        commission: float = 0.0004,  # 0.04% taker fee
        slippage: float = 0.001,     # 0.1% slippage
        leverage: int = 10
    ):
        self.params = params
        self.initial_balance = initial_balance
        self.commission = commission
        self.slippage = slippage
        self.leverage = leverage
        
        self.strategy = StrategyEngine(params)
        self.risk_manager = RiskManager(
            risk_per_trade=params.get('risk_per_trade', 0.02),
            leverage_cap=leverage
        )
        
        self._reset()
    
    def _reset(self):
        """Reset backtester state."""
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.trades = []
        self.equity_curve = [self.initial_balance]
        self.position = None
        self.trade_count = 0
    
    def run(self, df: pd.DataFrame, df_htf: Optional[pd.DataFrame] = None) -> BacktestResult:
        """
        Run backtest on historical data.
        
        Args:
            df: Primary timeframe OHLCV data
            df_htf: Higher timeframe data (optional)
        
        Returns:
            BacktestResult with trades and metrics
        """
        self._reset()
        
        if len(df) < 200:
            return self._empty_result()
        
        # Pre-calculate all indicators
        inds_df = calculate_indicators(df.copy(), self.params)
        df_with_inds = inds_df.get('df', df)
        
        # Warmup period
        warmup = 200
        
        for i in range(warmup, len(df_with_inds)):
            # Get current candle and history
            current = df_with_inds.iloc[i]
            history = df_with_inds.iloc[:i+1]
            
            timestamp = current.get('timestamp', i)
            price = current['close']
            high = current['high']
            low = current['low']
            
            # Update equity
            if self.position:
                self._update_position_pnl(price)
            
            # Check for exits first
            if self.position:
                exit_reason = self._check_exit(current, history)
                if exit_reason:
                    self._close_position(price, timestamp, exit_reason)
            
            # Check for entries if no position
            if not self.position:
                signal = self._generate_signal(history, df_htf)
                if signal.direction in ['long', 'short']:
                    self._open_position(signal, price, timestamp)
            
            # Record equity
            self.equity_curve.append(self.equity)
        
        # Close any remaining position
        if self.position:
            last_price = df_with_inds.iloc[-1]['close']
            self._close_position(last_price, df_with_inds.iloc[-1].get('timestamp', len(df)), 'END_OF_DATA')
        
        # Calculate metrics
        metrics_calc = PerformanceMetrics(self.trades, self.initial_balance)
        metrics = metrics_calc.calculate_all()
        
        return BacktestResult(
            trades=self.trades,
            metrics=metrics,
            equity_curve=self.equity_curve,
            params=self.params,
            period=f"{df.index[0]} to {df.index[-1]}" if hasattr(df, 'index') else 'Unknown'
        )
    
    def _generate_signal(self, history: pd.DataFrame, df_htf: Optional[pd.DataFrame]) -> Signal:
        """Generate trading signal."""
        return self.strategy.generate_signal(
            df=history,
            df_htf=df_htf,
            position=None,
            global_sentiment=0.5,
            funding_rate=0.0
        )
    
    def _open_position(self, signal: Signal, price: float, timestamp):
        """Open a new position."""
        # Apply slippage
        if signal.direction == 'long':
            entry_price = price * (1 + self.slippage)
        else:
            entry_price = price * (1 - self.slippage)
        
        # Calculate position size
        quantity, risk_amount = self.risk_manager.calculate_position_size(
            balance=self.balance,
            entry_price=entry_price,
            stop_loss_price=signal.stop_loss,
            signal_score=signal.score
        )
        
        if quantity <= 0:
            return
        
        # Commission
        commission_cost = entry_price * quantity * self.commission
        
        self.position = {
            'side': signal.direction,
            'entry_price': entry_price,
            'quantity': quantity,
            'stop_loss': signal.stop_loss,
            'take_profit': signal.take_profit,
            'entry_time': timestamp,
            'commission': commission_cost,
            'max_price': entry_price,
            'min_price': entry_price,
            'unrealized_pnl': 0
        }
        
        self.balance -= commission_cost
        self.trade_count += 1
    
    def _close_position(self, price: float, timestamp, reason: str):
        """Close current position."""
        if not self.position:
            return
        
        # Apply slippage
        if self.position['side'] == 'long':
            exit_price = price * (1 - self.slippage)
        else:
            exit_price = price * (1 + self.slippage)
        
        # Calculate PnL
        entry = self.position['entry_price']
        qty = self.position['quantity']
        
        if self.position['side'] == 'long':
            gross_pnl = (exit_price - entry) * qty
        else:
            gross_pnl = (entry - exit_price) * qty
        
        # Commission
        exit_commission = exit_price * qty * self.commission
        net_pnl = gross_pnl - exit_commission - self.position['commission']
        
        # Update balance
        self.balance += net_pnl
        self.equity = self.balance
        
        # Record trade
        self.trades.append({
            'entry_time': self.position['entry_time'],
            'exit_time': timestamp,
            'side': self.position['side'],
            'entry_price': entry,
            'exit_price': exit_price,
            'quantity': qty,
            'gross_pnl': gross_pnl,
            'commission': self.position['commission'] + exit_commission,
            'pnl': net_pnl,
            'pnl_pct': (net_pnl / (entry * qty)) * 100,
            'reason': reason
        })
        
        self.position = None
    
    def _check_exit(self, current: pd.Series, history: pd.DataFrame) -> Optional[str]:
        """Check if position should be closed."""
        if not self.position:
            return None
        
        price = current['close']
        high = current['high']
        low = current['low']
        
        side = self.position['side']
        entry = self.position['entry_price']
        sl = self.position['stop_loss']
        tp = self.position['take_profit']
        
        # Update max/min price for trailing
        if side == 'long':
            self.position['max_price'] = max(self.position['max_price'], high)
        else:
            self.position['min_price'] = min(self.position['min_price'], low)
        
        # Stop Loss
        if side == 'long' and low <= sl:
            return 'STOP_LOSS'
        if side == 'short' and high >= sl:
            return 'STOP_LOSS'
        
        # Take Profit
        if side == 'long' and high >= tp:
            return 'TAKE_PROFIT'
        if side == 'short' and low <= tp:
            return 'TAKE_PROFIT'
        
        # Dynamic trailing stop check
        atr = current.get('atr', 0)
        if atr > 0:
            pos_dict = {
                'side': side,
                'entry': entry,
                'trail_stop': self.position.get('trail_stop', 0),
                'max_price': self.position['max_price'],
                'min_price': self.position['min_price']
            }
            
            new_trail, should_exit = self.risk_manager.update_trailing_stop(
                pos_dict, price, atr, self.params
            )
            
            self.position['trail_stop'] = new_trail
            
            if should_exit:
                return 'TRAILING_STOP'
        
        # Signal-based exit
        signal = self.strategy.generate_signal(history, None, pos_dict)
        if signal.direction != 'neutral' and signal.score >= 6:
            if (side == 'long' and signal.direction == 'short') or \
               (side == 'short' and signal.direction == 'long'):
                return 'SIGNAL_REVERSAL'
        
        return None
    
    def _update_position_pnl(self, price: float):
        """Update unrealized PnL."""
        if not self.position:
            return
        
        entry = self.position['entry_price']
        qty = self.position['quantity']
        
        if self.position['side'] == 'long':
            unrealized = (price - entry) * qty
        else:
            unrealized = (entry - price) * qty
        
        self.position['unrealized_pnl'] = unrealized
        self.equity = self.balance + unrealized
    
    def _empty_result(self) -> BacktestResult:
        """Return empty result for insufficient data."""
        return BacktestResult(
            trades=[],
            metrics={'total_trades': 0, 'total_return': 0},
            equity_curve=[self.initial_balance],
            params=self.params,
            period='Insufficient data'
        )


class WalkForwardOptimizer:
    """
    Walk-Forward Optimization (WFO) for robust strategy validation.
    Prevents overfitting by testing on out-of-sample data.
    """
    
    def __init__(
        self,
        param_grid: Dict[str, List],
        initial_balance: float = 10000.0,
        num_folds: int = 5,
        train_ratio: float = 0.7,
        optimization_metric: str = 'sharpe_ratio',
        min_trades_per_fold: int = 30
    ):
        self.param_grid = param_grid
        self.initial_balance = initial_balance
        self.num_folds = num_folds
        self.train_ratio = train_ratio
        self.optimization_metric = optimization_metric
        self.min_trades_per_fold = min_trades_per_fold
        
        self.results = []
        self.best_params = None
        self.best_score = float('-inf')
    
    def run(
        self,
        df: pd.DataFrame,
        df_htf: Optional[pd.DataFrame] = None,
        base_params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Run walk-forward optimization.
        
        Args:
            df: Full historical data
            df_htf: Higher timeframe data
            base_params: Base parameters to extend
        
        Returns:
            Dict with best params, OOS results, and validation metrics
        """
        if base_params is None:
            base_params = {}
        
        total_len = len(df)
        fold_size = total_len // self.num_folds
        
        # Generate parameter combinations
        param_combinations = self._generate_combinations()
        
        print(f"Running WFO with {len(param_combinations)} param combinations over {self.num_folds} folds")
        
        all_oos_results = []
        
        for params in param_combinations:
            full_params = {**base_params, **params}
            
            fold_results = []
            fold_valid = True
            
            for fold in range(self.num_folds):
                # Split data
                start_idx = fold * fold_size
                end_idx = start_idx + fold_size
                
                if end_idx > total_len:
                    break
                
                fold_data = df.iloc[start_idx:end_idx].copy()
                
                train_size = int(len(fold_data) * self.train_ratio)
                train_data = fold_data.iloc[:train_size]
                test_data = fold_data.iloc[train_size:]
                
                # Run in-sample (training)
                bt_train = Backtester(full_params, self.initial_balance)
                train_result = bt_train.run(train_data)
                
                # Skip if training fails minimum criteria
                if train_result.metrics.get('total_trades', 0) < self.min_trades_per_fold // 2:
                    fold_valid = False
                    break
                
                if train_result.metrics.get('profit_factor', 0) < 1.0:
                    fold_valid = False
                    break
                
                # Run out-of-sample (testing)
                bt_test = Backtester(full_params, self.initial_balance)
                test_result = bt_test.run(test_data)
                test_result.is_oos = True
                
                fold_results.append({
                    'fold': fold,
                    'train': train_result.metrics,
                    'test': test_result.metrics
                })
            
            if not fold_valid or not fold_results:
                continue
            
            # Calculate average OOS performance
            oos_metrics = [r['test'] for r in fold_results]
            avg_metric = np.mean([m.get(self.optimization_metric, 0) for m in oos_metrics])
            avg_pf = np.mean([m.get('profit_factor', 0) for m in oos_metrics])
            avg_wr = np.mean([m.get('win_rate', 0) for m in oos_metrics])
            avg_dd = np.mean([m.get('max_drawdown', 1) for m in oos_metrics])
            
            # Robustness check: All folds must be profitable
            all_profitable = all(m.get('total_return', 0) > 0 for m in oos_metrics)
            
            if not all_profitable:
                continue
            
            # Update best
            if avg_metric > self.best_score:
                self.best_score = avg_metric
                self.best_params = full_params
                
                all_oos_results.append({
                    'params': params,
                    'avg_metric': avg_metric,
                    'avg_profit_factor': avg_pf,
                    'avg_win_rate': avg_wr,
                    'avg_drawdown': avg_dd,
                    'fold_results': fold_results
                })
        
        # Final validation on holdout set (last 20% of data)
        final_validation = None
        if self.best_params:
            holdout_start = int(len(df) * 0.8)
            holdout_data = df.iloc[holdout_start:]
            
            bt_final = Backtester(self.best_params, self.initial_balance)
            final_result = bt_final.run(holdout_data)
            
            final_validation = {
                'metrics': final_result.metrics,
                'trades': len(final_result.trades),
                'passed': self._validate_final(final_result.metrics)
            }
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'optimization_metric': self.optimization_metric,
            'all_results': sorted(all_oos_results, key=lambda x: x['avg_metric'], reverse=True)[:10],
            'final_validation': final_validation,
            'num_combinations_tested': len(param_combinations),
            'num_folds': self.num_folds
        }
    
    def _generate_combinations(self) -> List[Dict]:
        """Generate all parameter combinations."""
        import itertools
        
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        # Shuffle for better early stopping
        np.random.shuffle(combinations)
        
        return combinations
    
    def _validate_final(self, metrics: Dict) -> bool:
        """Validate final results against thresholds."""
        thresholds = {
            'min_sharpe': 0.5,
            'min_profit_factor': 1.1,
            'min_win_rate': 35,
            'max_drawdown': 0.30
        }
        
        if metrics.get('sharpe_ratio', 0) < thresholds['min_sharpe']:
            return False
        if metrics.get('profit_factor', 0) < thresholds['min_profit_factor']:
            return False
        if metrics.get('win_rate', 0) < thresholds['min_win_rate']:
            return False
        if metrics.get('max_drawdown', 1) > thresholds['max_drawdown']:
            return False
        
        return True


def run_quick_backtest(
    df: pd.DataFrame,
    params: Dict[str, Any],
    initial_balance: float = 10000.0
) -> Dict[str, Any]:
    """
    Run a quick backtest with given parameters.
    
    Returns:
        Dict with metrics and trade summary
    """
    bt = Backtester(params, initial_balance)
    result = bt.run(df)
    
    return {
        'metrics': result.metrics,
        'total_trades': len(result.trades),
        'equity_curve': result.equity_curve,
        'winning_trades': sum(1 for t in result.trades if t.get('pnl', 0) > 0),
        'losing_trades': sum(1 for t in result.trades if t.get('pnl', 0) < 0),
        'params': params
    }
