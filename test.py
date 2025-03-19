import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

def fetch_us_stock_data(ticker="AAPL", start_date="2022-01-01", end_date="2023-01-01"):
    data = yf.download(ticker, start=start_date, end=end_date)

    # Flatten multi-index columns if needed
    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

    return data

def compute_moving_averages(df, short_window=20, long_window=50):
    df[f"MA_{short_window}"] = df["Close"].rolling(window=short_window).mean()
    df[f"MA_{long_window}"] = df["Close"].rolling(window=long_window).mean()
    return df

def simple_arima_forecast(df, forecast_days=5):
    close_prices = df["Close"].dropna()
    model = ARIMA(close_prices, order=(1,1,1))
    model_fit = model.fit()
    forecast_result = model_fit.forecast(steps=forecast_days)
    return forecast_result

def plot_results(df, ticker="AAPL", forecast_data=None):
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df["Close"], label="Close Price", color="blue")

    # Now columns should be strings, so this will work:
    for col in df.columns:
        if col.startswith("MA_"):
            plt.plot(df.index, df[col], label=col, linestyle="--")

    if forecast_data is not None:
        last_date = df.index[-1]
        # We'll assume daily frequency for a forecast
        forecast_index = pd.date_range(start=last_date, periods=len(forecast_data)+1, freq='B')[1:]
        plt.plot(forecast_index, forecast_data, label="ARIMA Forecast", marker="o", color="red")

    plt.title(f"US Stock Prediction Test for {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    ticker_symbol = "AAPL"

    data_df = fetch_us_stock_data(ticker=ticker_symbol)
    data_df = compute_moving_averages(data_df)
    forecast_values = simple_arima_forecast(data_df, forecast_days=5)
    plot_results(data_df, ticker=ticker_symbol, forecast_data=forecast_values)
