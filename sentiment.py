# sentiment.py (updated)
from textblob import TextBlob
import yfinance as yf

def analyze_sentiment(symbol):
    try:
        # Clean the symbol (remove .NS if present)
        clean_symbol = symbol.split('.')[0]
        
        ticker = yf.Ticker(clean_symbol)
        news = ticker.news
        
        if not news:
            print(f"No news articles found for {clean_symbol}")
            return 0.0
            
        total_polarity = 0
        count = 0
        
        for article in news[:5]:  # Analyze first 5 headlines
            title = article.get('title', '')
            if not title:
                continue
                
            analysis = TextBlob(title)
            total_polarity += analysis.sentiment.polarity
            count += 1

        if count == 0:
            print("No valid headlines found")
            return 0.0
            
        return round(total_polarity / count, 2)
        
    except Exception as e:
        print(f"Sentiment analysis failed: {str(e)}")
        return 0.0