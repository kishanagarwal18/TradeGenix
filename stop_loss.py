import numpy as np

def get_dynamic_thresholds(data, sentiment_score, current_price=None):
    """
    Calculate stop-loss/take-profit with better parameter handling
    Args:
        data: DataFrame with historical prices
        sentiment_score: Numeric sentiment value (-1 to 1)
        current_price: Optional current price (defaults to last close)
    Returns:
        tuple: (stop_loss, take_profit, confidence)
    """
    try:
        # Get current price if not provided
        if current_price is None:
            current_price = data['Close'].iloc[-1]
        
        # Calculate volatility
        returns = data['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized
        
        # Base values
        base_sl = current_price * 0.95
        base_tp = current_price * 1.05
        
        # Adjust based on sentiment
        if sentiment_score < -0.3:  # Bearish
            stop_loss = min(base_sl, current_price * 0.98)
            take_profit = current_price * 1.03
            sentiment_factor = 0.6  # Lower confidence
        elif sentiment_score > 0.3:  # Bullish
            stop_loss = current_price * 0.97
            take_profit = current_price * 1.08
            sentiment_factor = 0.8  # Higher confidence
        else:  # Neutral
            stop_loss = base_sl
            take_profit = base_tp
            sentiment_factor = 0.7
            
        # Adjust for volatility
        if volatility > 0.25:  # High volatility
            stop_loss = current_price * 0.96
            take_profit = current_price * 1.06
            volatility_factor = 0.6
        else:
            volatility_factor = 0.8
            
        # Calculate confidence score (0-1)
        confidence = min(0.95, (sentiment_factor * 0.6 + volatility_factor * 0.4))
        
        return stop_loss, take_profit, confidence
        
    except Exception as e:
        print(f"Error in threshold calculation: {e}")
        # Return conservative defaults
        return current_price * 0.95, current_price * 1.05, 0.5