# pip install python-binance
import pandas as pd
import numpy as np
from binance.client import Client
import plotly.graph_objects as go
import requests
import time
import mplfinance as mpf
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf
import matplotlib.pyplot as plt
import pytz

from scipy.stats import bernoulli
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
import os
import schedule

# Binance API credentials
api_key = 'zMMoF47K9D6u1UrEs0SF5ZgwDtYsLfZYPPd2hDme5XtUJJOd6gogsVw8ibNu7mxM'
api_secret = 'xPqzQ87NvKFawOcYizIC81Ui3s7oQsBxPvXaD4t7LR85AtUhYeJL9XLnvwmoLPLN'

#Telegram Bot
TOKEN = "7440240128:AAHGgBidb-mEjSOfzWXJ2hzY8UupUDlvKEs"
CHAT_ID = "-4629116369"

# Initialize Binance client
client = Client(api_key, api_secret)


def get_historical_klines(symbol, interval, lookback):
    """
    Fetch historical klines (candlestick) data from Binance.
    :param symbol: Trading pair symbol (e.g., 'BTCUSDT')
    :param interval: Timeframe for candlesticks (e.g., '1h', '1d')
    :param lookback: Lookback period (e.g., '1 day ago UTC',2 months ago UTC)
    :return: Pandas DataFrame with OHLCV data
    """
    try:
        klines = client.get_historical_klines(symbol, interval, lookback)
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 
            'close_time', 'quote_asset_volume', 'number_of_trades', 
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        time_to_add = pd.Timedelta(hours=5, minutes=30)
        df['timestamp'] = df['timestamp']+time_to_add
        df['date'] = df['timestamp'].apply(lambda x: datetime.strftime(x, "%Y-%m-%d"))
        #df.set_index('timestamp', inplace=True)
        df['loc_index'] = list(range(len(df)))
        df['open'] = df.open.astype(float)
        df['high'] = df.high.astype(float)
        df['low'] = df.low.astype(float)
        df['close'] = df.close.astype(float)
        df['volume'] = df.volume.astype(float)
        #df = df.astype(float)
        return df
    except Exception as e:
        raise Exception(f"Error fetching data: {e}")
    
def hours_42_high_low (symbol):
    # Get 42 hours high low Data
    symbol = symbol  # Trading pair symbol
    interval = '1d'  # Time interval (e.g., '1h', '1d')
    lookback = '3 day ago UTC'  # Lookback period

    # Fetch data
    df = get_historical_klines(symbol, interval, lookback)

    previous_day_high = df.iloc[-3,2]
    previous_day_low = df.iloc[-3,3]

    day_before_previous_day_high = df.iloc[-2,2]
    day_before_previous_day_low = df.iloc[-2,3]

    symbol = symbol  # Trading pair symbol
    interval = '1m'  # Time interval (e.g., '1h', '1d')
    lookback = '10 m ago UTC'  # Lookback period

    # Fetch data
    df = get_historical_klines(symbol, interval, lookback)
    current_price  = df.iloc[-1,4]

    # print(previous_day_high)
    # print(previous_day_low)
    # print(day_before_previous_day_high)
    # print(day_before_previous_day_low)
    # print(current_price)

    difference = 0
    if (symbol == 'BTCUSDT'):
        difference = 100
    elif(symbol == 'ETHUSDT'):
        difference = 20
    else:
        difference = 2

    if (abs(current_price-previous_day_high))<difference:
        print("close to previous_day_high")
    if (abs(current_price-previous_day_low))<difference:
        print("close to previous_day_low")
    if (abs(current_price-day_before_previous_day_high))<difference:
        print("close to day_before_previous_day_high")
    if (abs(current_price-day_before_previous_day_low))<difference:
        print("close to day_before_previous_day_low")


#get high/low data in given time
def get_high_low(df, start_datetime, end_datetime):
    """
    Get the high and low prices between a given start and end datetime.

    Parameters:
        df (pd.DataFrame): The dataframe containing price data.
        start_datetime (str): The start datetime in 'YYYY-MM-DD HH:MM:SS' format.
        end_datetime (str): The end datetime in 'YYYY-MM-DD HH:MM:SS' format.

    Returns:
        tuple: (highest price, lowest price)
    """

    # Convert timestamps to datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Convert input times to datetime
    start_datetime = pd.to_datetime(start_datetime)
    end_datetime = pd.to_datetime(end_datetime)

    # Filter the DataFrame based on the time range
    filtered_df = df[(df['timestamp'] >= start_datetime) & (df['timestamp'] <= end_datetime)]

    if filtered_df.empty:
        return None, None  # Return None if no data is found in the range

    # Get the highest and lowest prices
    high_price = filtered_df['high'].max()
    low_price = filtered_df['low'].min()

    return high_price, low_price

#sent alert for every session open and close with high/low report data
#get high/low data in given time
def get_high_low(df, start_datetime, end_datetime):
    """
    Get the high and low prices between a given start and end datetime.

    Parameters:
        df (pd.DataFrame): The dataframe containing price data.
        start_datetime (str): The start datetime in 'YYYY-MM-DD HH:MM:SS' format.
        end_datetime (str): The end datetime in 'YYYY-MM-DD HH:MM:SS' format.

    Returns:
        tuple: (highest price, lowest price)
    """

    # Convert timestamps to datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Convert input times to datetime
    start_datetime = pd.to_datetime(start_datetime)
    end_datetime = pd.to_datetime(end_datetime)

    # Filter the DataFrame based on the time range
    filtered_df = df[(df['timestamp'] >= start_datetime) & (df['timestamp'] <= end_datetime)]

    if filtered_df.empty:
        return None, None  # Return None if no data is found in the range

    # Get the highest and lowest prices
    high_price = filtered_df['high'].max()
    low_price = filtered_df['low'].min()

    return high_price, low_price

def send_alert():
    print("Alert")

def open_asia():
    print("Asia session opened")

def open_london():
    print("London session opened")

def open_ny():
    print("New York session opened")

def close_asia():
    # Get Data
    symbol = 'BTCUSDT'  # Trading pair symbol
    interval = '5m'  # Time interval (e.g., '1h', '1d')
    lookback = '2 day ago UTC'  # Lookback period
    # Fetch data
    df = get_historical_klines(symbol, interval, lookback)
    current_date = datetime.today().date()
    start_time = str(current_date)+" "+"05:30:00"
    end_time = str(current_date)+" "+"14:30:00"
    high, low = get_high_low(df, start_time, end_time)
    print(f"High: {high}, Low: {low} of BTC")
    
    # Get Data
    symbol = 'ETHUSDT'  # Trading pair symbol
    interval = '5m'  # Time interval (e.g., '1h', '1d')
    lookback = '2 day ago UTC'  # Lookback period
    # Fetch data
    df = get_historical_klines(symbol, interval, lookback)
    current_date = datetime.today().date()
    start_time = str(current_date)+" "+"05:30:00"
    end_time = str(current_date)+" "+"14:30:00"
    high, low = get_high_low(df, start_time, end_time)
    print(f"High: {high}, Low: {low} of ETH")

    # Get Data
    symbol = 'SOLUSDT'  # Trading pair symbol
    interval = '5m'  # Time interval (e.g., '1h', '1d')
    lookback = '2 day ago UTC'  # Lookback period
    # Fetch data
    df = get_historical_klines(symbol, interval, lookback)
    current_date = datetime.today().date()
    start_time = str(current_date)+" "+"05:30:00"
    end_time = str(current_date)+" "+"14:30:00"
    high, low = get_high_low(df, start_time, end_time)
    print(f"High: {high}, Low: {low} of SOL")

    print("Asia session close")

def close_london():
    # Get Data
    symbol = 'BTCUSDT'  # Trading pair symbol
    interval = '5m'  # Time interval (e.g., '1h', '1d')
    lookback = '2 day ago UTC'  # Lookback period
    # Fetch data
    df = get_historical_klines(symbol, interval, lookback)
    current_date = datetime.today().date()
    start_time = str(current_date)+" "+"12:30:00"
    end_time = str(current_date)+" "+"20:30:00"
    high, low = get_high_low(df, start_time, end_time)
    print(f"High: {high}, Low: {low} of BTC")

    # Get Data
    symbol = 'ETHUSDT'  # Trading pair symbol
    interval = '5m'  # Time interval (e.g., '1h', '1d')
    lookback = '2 day ago UTC'  # Lookback period
    # Fetch data
    df = get_historical_klines(symbol, interval, lookback)
    current_date = datetime.today().date()
    start_time = str(current_date)+" "+"12:30:00"
    end_time = str(current_date)+" "+"20:30:00"
    high, low = get_high_low(df, start_time, end_time)
    print(f"High: {high}, Low: {low} of ETH")

        # Get Data
    symbol = 'SOLUSDT'  # Trading pair symbol
    interval = '5m'  # Time interval (e.g., '1h', '1d')
    lookback = '2 day ago UTC'  # Lookback period
    # Fetch data
    df = get_historical_klines(symbol, interval, lookback)
    current_date = datetime.today().date()
    start_time = str(current_date)+" "+"12:30:00"
    end_time = str(current_date)+" "+"20:30:00"
    high, low = get_high_low(df, start_time, end_time)
    print(f"High: {high}, Low: {low} of SOL")

    print("London session close")

def close_ny():
    # Get Data
    symbol = 'BTCUSDT'  # Trading pair symbol
    interval = '5m'  # Time interval (e.g., '1h', '1d')
    lookback = '2 day ago UTC'  # Lookback period
    # Fetch data
    df = get_historical_klines(symbol, interval, lookback)
    current_date = datetime.today().date()
    start_time = str(current_date)+" "+"18:30:00"
    end_time = str(current_date)+" "+"02:30:00"
    high, low = get_high_low(df, start_time, end_time)
    print(f"High: {high}, Low: {low} of BTC")

    # Get Data
    symbol = 'ETHUSDT'  # Trading pair symbol
    interval = '5m'  # Time interval (e.g., '1h', '1d')
    lookback = '2 day ago UTC'  # Lookback period
    # Fetch data
    df = get_historical_klines(symbol, interval, lookback)
    current_date = datetime.today().date()
    start_time = str(current_date)+" "+"18:30:00"
    end_time = str(current_date)+" "+"02:30:00"
    high, low = get_high_low(df, start_time, end_time)
    print(f"High: {high}, Low: {low} of ETH")

    # Get Data
    symbol = 'SOLUSDT'  # Trading pair symbol
    interval = '5m'  # Time interval (e.g., '1h', '1d')
    lookback = '2 day ago UTC'  # Lookback period
    # Fetch data
    df = get_historical_klines(symbol, interval, lookback)
    current_date = datetime.today().date()
    start_time = str(current_date)+" "+"18:30:00"
    end_time = str(current_date)+" "+"02:30:00"
    high, low = get_high_low(df, start_time, end_time)
    print(f"High: {high}, Low: {low} of SOL")

    print("New York session close")

#Asis
schedule.every().day.at("05:30").do(open_asia)
schedule.every().day.at("14:31").do(close_asia)

#New York
schedule.every().day.at("18:30").do(open_ny)
schedule.every().day.at("02:31").do(close_ny)

#London
schedule.every().day.at("12:30").do(open_london)
schedule.every().day.at("20:31").do(close_london)

#Test
# schedule.every().day.at("10:30").do(alert)
# schedule.every().day.at("10:31").do(alert)

while True:
    schedule.run_pending()
    hours_42_high_low('BTCUSDT')
    hours_42_high_low('ETHUSDT')
    hours_42_high_low('SOLUSDT')
    time.sleep(300)


