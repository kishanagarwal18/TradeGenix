import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

def predict_future_prices(df, future_days=30):
    """
    Predict future stock prices using LSTM
    Args:
        df: DataFrame with historical data
        future_days: Number of days to predict ahead
    Returns:
        tuple: (historical_predictions, future_predictions, dates, metrics)
    """
    # Prepare data
    prices = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)
    
    # Create sequences
    X, y = [], []
    sequence_length = 60
    
    for i in range(sequence_length, len(scaled)):
        X.append(scaled[i-sequence_length:i, 0])
        y.append(scaled[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Split into train and test (80-20)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Build model
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.3),
        LSTM(100, return_sequences=False),
        Dropout(0.3),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, 
                       validation_data=(X_test, y_test), verbose=0)
    
    # Predict on test data for accuracy metrics
    test_preds = model.predict(X_test)
    test_preds = scaler.inverse_transform(test_preds).flatten()
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    metrics = {
        'MAE': mean_absolute_error(y_test_actual, test_preds),
        'MAPE': mean_absolute_percentage_error(y_test_actual, test_preds) * 100,
        'RMSE': np.sqrt(mean_squared_error(y_test_actual, test_preds)),
        'Training Loss': history.history['loss'][-1],
        'Validation Loss': history.history['val_loss'][-1]
    }
    
    # Predict on historical data
    historical_preds = model.predict(X)
    historical_preds = scaler.inverse_transform(historical_preds).flatten()
    
    # Predict future prices
    future_predictions = []
    last_sequence = scaled[-sequence_length:]
    
    for _ in range(future_days):
        next_pred = model.predict(last_sequence.reshape(1, sequence_length, 1))
        future_predictions.append(next_pred[0,0])
        last_sequence = np.append(last_sequence[1:], next_pred[0,0])
    
    future_predictions = scaler.inverse_transform(
        np.array(future_predictions).reshape(-1, 1)
    ).flatten()
    
    # Generate future dates
    last_date = df.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=future_days
    )
    
    return historical_preds, future_predictions, future_dates, metrics