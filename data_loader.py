# In data_loader.py
from alpha_vantage.timeseries import TimeSeries

def load_stock_data(symbol, days):
    try:
        ts = TimeSeries(key='B3V81JX3JI16JZST', output_format='pandas')
        df, _ = ts.get_daily(symbol=symbol.replace('.NS', ''), outputsize='full')
        df = df.sort_index(ascending=True)
        df = df.tail(days)
        df = df.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume'
        })
        return df
    except Exception as e:
        return None