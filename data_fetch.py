"""
data_fetch.py

Handles downloading historical stock data using yfinance.
"""

import yfinance as yf
import pandas as pd
import datetime as dt
import time
from requests.exceptions import ConnectionError, Timeout

def fetch_stock_data(ticker: str, years: int = 5, retries: int = 3, delay: int = 5) -> pd.DataFrame:
    """
    Downloads 'years' years of daily historical data for 'ticker' from Yahoo Finance.
    Returns a DataFrame sorted in ascending order.
    Retries the download if connection issues occur.
    """
    end_date = dt.datetime.today()
    start_date = end_date - dt.timedelta(days=years * 365)
    attempt = 0
    while attempt < retries:
        try:
            print(f"[fetch_stock_data] Attempt {attempt+1}: Downloading {years} years of data for {ticker} ...")
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if df.empty:
                raise ValueError(f"No data found for ticker {ticker}.")
            df.sort_index(ascending=True, inplace=True)
            return df
        except (ConnectionError, Timeout) as e:
            print(f"[fetch_stock_data] Connection error: {e}. Retrying in {delay} seconds...")
            attempt += 1
            time.sleep(delay)
    raise ConnectionError(f"Failed to download data for {ticker} after {retries} attempts.")
