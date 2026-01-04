import numpy as np

def get_rl_action(df, sentiment_score, future_predictions=None):
    """
    Enhanced RL agent that considers future predictions
    Returns action and confidence score separately
    """
    if future_predictions is None:
        future_predictions = []
    
    # Simple trend analysis
    recent_prices = df['Close'][-10:].values
    recent_trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
    
    # Future trend analysis if available
    if len(future_predictions) > 1:
        future_trend = np.polyfit(range(len(future_predictions)), future_predictions, 1)[0]
        combined_trend = (recent_trend * 0.4) + (future_trend * 0.6)
    else:
        combined_trend = recent_trend
    
    # Decision making with confidence score
    if sentiment_score > 0.2 and combined_trend > 0:
        confidence = min(0.9, sentiment_score * 2 + combined_trend * 100 / recent_prices[-1])
        return "Strong Buy", confidence
    elif sentiment_score < -0.2 and combined_trend < 0:
        confidence = min(0.9, abs(sentiment_score) * 2 + abs(combined_trend) * 100 / recent_prices[-1])
        return "Strong Sell", confidence
    elif combined_trend > 0:
        confidence = 0.7  # Medium confidence for trend-based decisions
        return "Buy", confidence
    elif combined_trend < 0:
        confidence = 0.7
        return "Sell", confidence
    else:
        confidence = 0.5  # Low confidence for neutral
        return "Hold", confidence