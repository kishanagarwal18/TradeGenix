import streamlit as st
from lstm_model import predict_future_prices
from rl_agent import get_rl_action
from sentiment import analyze_sentiment
from data_loader import load_stock_data
from stop_loss import get_dynamic_thresholds
import matplotlib.pyplot as plt
import pandas as pd

st.title("📈 TradeGenix: AI-Powered Trading Agent")

# Sidebar options
symbol = st.sidebar.text_input("Enter Stock Symbol (e.g. AAPL, TSLA)", value="AAPL")
days = st.sidebar.slider("Days of Historical Data", 30, 365, 90)
future_days = st.sidebar.slider("Days to Predict", 5, 60, 30)

# Load data
data = load_stock_data(symbol, days)

if data is not None and not data.empty:
    st.subheader(f"Stock Price Data for {symbol}")
    st.line_chart(data['Close'])

    if len(data) < 60:
        st.error("❌ Not enough data to make predictions. Please select more days.")
        st.stop()

    # Future price prediction
    st.subheader(f"🔮 LSTM Future Price Prediction (Next {future_days} Days)")
    
    with st.spinner('Training model and making predictions...'):
        hist_preds, future_preds, future_dates, metrics = predict_future_prices(data, future_days)
        
        # Create DataFrame for plotting
        hist_dates = data.index[-len(hist_preds):]
        future_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted': future_preds
        }).set_index('Date')
        
        # Plot results
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data['Close'], label='Actual Prices', color='blue')
        ax.plot(hist_dates, hist_preds, label='Historical Predictions', color='green')
        ax.plot(future_df.index, future_df['Predicted'], label='Future Predictions', color='red', linestyle='--')
        ax.axvline(x=data.index[-1], color='gray', linestyle=':', label='Prediction Start')
        ax.set_title(f'{symbol} Price Prediction')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        ax.legend()
        st.pyplot(fig)
        
        # Show prediction table
        st.subheader("Predicted Prices")
        st.dataframe(future_df.style.format("{:.2f}"))
        
        # Show accuracy metrics
        st.subheader("📊 Model Accuracy Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Mean Absolute Error (MAE)", f"${metrics['MAE']:.2f}")
        col2.metric("Mean Absolute % Error (MAPE)", f"{metrics['MAPE']:.2f}%")
        col3.metric("Root Mean Squared Error (RMSE)", f"${metrics['RMSE']:.2f}")
        
        st.write("Training Loss:", metrics['Training Loss'])
        st.write("Validation Loss:", metrics['Validation Loss'])
        
        # Add explanation of metrics
        with st.expander("ℹ️ Understanding these metrics"):
            st.write("""
            - **MAE (Mean Absolute Error):** Average absolute difference between predicted and actual values
            - **MAPE (Mean Absolute Percentage Error):** Percentage error relative to actual prices
            - **RMSE (Root Mean Squared Error):** Square root of average squared errors (penalizes large errors more)
            - Lower values indicate better model accuracy
            - Training/Validation Loss: Model's error during training (lower is better)
            """)

    # Sentiment analysis
    st.subheader("🗞️ Sentiment Analysis")
    sentiment = analyze_sentiment(symbol)
    st.metric("Sentiment Score", sentiment)

    # RL agent decision based on future prediction
    st.subheader("🤖 RL Agent Decision")
    action, confidence = get_rl_action(data, sentiment, future_preds)
    
    col1, col2 = st.columns(2)
    col1.metric("Recommended Action", action)
    col2.metric("Confidence Score", f"{confidence*100:.1f}%")
    
    # Visual confidence indicator
    st.progress(int(confidence*100))
    
    if confidence < 0.5:
        st.warning("⚠️ Low confidence recommendation - consider verifying with other indicators")
    elif confidence > 0.7:
        st.success("✅ High confidence recommendation")

    # Stop-Loss and Take-Profit Calculation
    st.subheader("🛡️ Risk Management")
    current_price = data['Close'].iloc[-1]
    stop_loss, take_profit, risk_confidence = get_dynamic_thresholds(
        data=data,
        sentiment_score=sentiment,
        current_price=current_price
    )
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Stop-Loss", f"${stop_loss:.2f}", 
               f"{(stop_loss/current_price-1)*100:.2f}%")
    col2.metric("Take-Profit", f"${take_profit:.2f}", 
                f"{(take_profit/current_price-1)*100:.2f}%")
    col3.metric("Confidence", f"{risk_confidence*100:.0f}%")
    
    # Visual representation
    fig, ax = plt.subplots()
    ax.plot([current_price], [0], 'bo', label="Current Price")
    ax.plot([stop_loss], [0], 'ro', label="Stop-Loss")
    ax.plot([take_profit], [0], 'go', label="Take-Profit")
    ax.set_xlim(stop_loss * 0.98, take_profit * 1.02)
    ax.set_yticks([])
    ax.legend()
    st.pyplot(fig)

else:
    st.warning("⚠️ Failed to load stock data or data is empty. Try a different symbol.")