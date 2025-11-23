class RiskManager:
    def __init__(self, risk_per_trade=0.01, leverage=10, atr_multiplier=2.0):
        self.risk_per_trade = risk_per_trade
        self.leverage = leverage
        self.atr_multiplier = atr_multiplier

    def calculate_stop_loss_price(self, entry_price, atr, side):
        """
        Calculates Stop Loss price based on ATR.
        """
        if side == 'BUY':
            return entry_price - (atr * self.atr_multiplier)
        elif side == 'SELL':
            return entry_price + (atr * self.atr_multiplier)
        return None

    def calculate_take_profit_price(self, entry_price, atr, side, risk_reward_ratio=None, tp_atr_multiplier=None):
        """
        Calculates Take Profit price. 
        Supports either fixed Risk/Reward OR Dynamic ATR Multiplier.
        """
        # Default to Risk/Reward if not specified
        if risk_reward_ratio is None and tp_atr_multiplier is None:
            risk_reward_ratio = 2.0
            
        if tp_atr_multiplier:
            # Dynamic TP based on Volatility
            reward = atr * tp_atr_multiplier
        else:
            # Fixed Risk/Reward
            sl_price = self.calculate_stop_loss_price(entry_price, atr, side)
            risk = abs(entry_price - sl_price)
            reward = risk * risk_reward_ratio
        
        if side == 'BUY':
            return entry_price + reward
        elif side == 'SELL':
            return entry_price - reward
        return None

    def calculate_quantity(self, balance, entry_price, stop_loss_price):
        """
        Calculates position size (quantity) based on risk percentage.
        Quantity = (Balance * Risk%) / |Entry - SL|
        """
        if entry_price == 0 or stop_loss_price == 0:
            return 0.0
            
        risk_amount = balance * self.risk_per_trade
        price_diff = abs(entry_price - stop_loss_price)
        
        if price_diff == 0:
            return 0.0
            
        quantity = risk_amount / price_diff
        
        # Adjust for leverage (ensure we don't exceed max leverage)
        # Max Position Value = Balance * Leverage
        max_position_value = balance * self.leverage
        position_value = quantity * entry_price
        
        if position_value > max_position_value:
            quantity = max_position_value / entry_price
            
        return quantity

    def recommend_leverage(self, volatility_pct):
        """
        Recommends leverage based on volatility.
        Rule: Target Risk 20% / Volatility %
        """
        if volatility_pct <= 0: return 10
        rec_leverage = int(20 / volatility_pct)
        return max(1, min(rec_leverage, 20)) # Cap between 1x and 20x
