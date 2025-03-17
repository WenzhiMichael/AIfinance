"""
main.py

Entry point for the datastock project.
It ties everything together:
1) Fetch historical stock data (data_fetch.py)
2) Train (or load) a model for next‑day forecasting (model_train.py)
   - Options: LSTM, GRU (StockGRU), Transformer (StockTransformer)
3) Forecast tomorrow's closing price and print the result
4) Plot the daily Open and Close prices for the last 5 days
"""

import sys
import pandas as pd
import plotly.express as px

from data_fetch import fetch_stock_data
from model_train import (
    train_stock_lstm_model, forecast_stock_next_day_lstm,
    train_stock_gru_model, forecast_stock_next_day_gru,
    train_stock_transformer_model, forecast_stock_next_day_transformer
)

def run_pipeline(ticker: str, model_type: str = "lstm"):
    """
    Orchestrates the workflow:
    - Downloads 5 years of historical data for the given ticker.
    - Trains (or loads) the selected model (LSTM, GRU, or Transformer).
    - Forecasts tomorrow’s closing price.
    - Plots the daily Open and Close for the last 5 days.

    model_type: "lstm", "gru", or "transformer" (case-insensitive)
    """
    try:
        # 1) Download 5 years of data (for training)
        df = fetch_stock_data(ticker, years=5)
    except Exception as e:
        print(f"[run_pipeline] Data fetching error: {e}")
        return

    # 2) Train and forecast using the selected model
    model_type = model_type.lower()
    if model_type == "lstm":
        model, scaler, _, _ = train_stock_lstm_model(df, window_size=60, model_path='lstm_model.h5')
        forecast_price = forecast_stock_next_day_lstm(model, scaler, df, window_size=60)
    elif model_type == "gru":
        model, scaler, _, _ = train_stock_gru_model(df, window_size=60, model_path='gru_model.h5')
        forecast_price = forecast_stock_next_day_gru(model, scaler, df, window_size=60)
    elif model_type == "transformer":
        model, scaler, _, _ = train_stock_transformer_model(df, window_size=60, model_path='transformer_model.h5')
        forecast_price = forecast_stock_next_day_transformer(model, scaler, df, window_size=60)
    else:
        print(f"[run_pipeline] Unknown model type: {model_type}. Choose from 'lstm', 'gru', or 'transformer'.")
        return

    print(f"[run_pipeline] Tomorrow's predicted closing price for {ticker} using {model_type.upper()}: ${forecast_price:.2f}")

    # 4) Plot daily Open and Close for the last 5 days
    last_5_days = df.tail(5).reset_index()

    # Flatten multi-index columns if necessary
    if isinstance(last_5_days.columns, pd.MultiIndex):
        new_cols = []
        for col in last_5_days.columns:
            if isinstance(col, tuple):
                if col[0] == "" or col[0].lower() == "date":
                    new_cols.append("Date")
                else:
                    new_cols.append(col[0])
            else:
                new_cols.append(col if col != "" else "Date")
        last_5_days.columns = new_cols
        last_5_days = last_5_days.loc[:, ~last_5_days.columns.duplicated()]

    print(f"[run_pipeline] Columns in last_5_days after flattening: {last_5_days.columns.tolist()}")
    required_columns = {'Date', 'Open', 'Close'}
    if not required_columns.issubset(last_5_days.columns):
        print(f"[run_pipeline] Missing required columns for plotting. Found columns: {last_5_days.columns.tolist()}")
        return

    print("[run_pipeline] Last 5 days data (post-flatten):")
    print(last_5_days[['Date', 'Open', 'Close']])

    # Transform DataFrame to long format for Plotly
    last_5_days_melted = last_5_days.melt(
        id_vars=['Date'],
        value_vars=['Open', 'Close'],
        var_name='Price Type',
        value_name='Price'
    )
    fig = px.line(
        last_5_days_melted,
        x='Date',
        y='Price',
        color='Price Type',
        title=f'{ticker} - Last 5 Days Daily Open vs Close',
        labels={'Price': 'Price (USD)', 'Date': 'Date'}
    )
    fig.update_layout(xaxis=dict(tickformat="%Y-%m-%d"))
    fig.show()

if __name__ == "__main__":
    ticker_symbol = "AAPL"
    # Default model type is LSTM; you can pass "lstm", "gru", or "transformer" as the first argument.
    model_type = "lstm"
    if len(sys.argv) > 1:
        model_type = sys.argv[1]
    run_pipeline(ticker_symbol, model_type)
