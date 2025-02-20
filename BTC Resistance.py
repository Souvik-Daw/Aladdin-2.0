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
from scipy.stats import bernoulli
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
import os

# Binance API credentials
api_key = 'zMMoF47K9D6u1UrEs0SF5ZgwDtYsLfZYPPd2hDme5XtUJJOd6gogsVw8ibNu7mxM'
api_secret = 'xPqzQ87NvKFawOcYizIC81Ui3s7oQsBxPvXaD4t7LR85AtUhYeJL9XLnvwmoLPLN'

#Telegram Bot
TOKEN = "7440240128:AAHGgBidb-mEjSOfzWXJ2hzY8UupUDlvKEs"
CHAT_ID = "5097888685"

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
    


def candlestick_ax(t, o, h, l, c):
    t_index = list(range(len(t)))

    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)

    color = ["green" if close_price > open_price else "red" for close_price, open_price in zip(c, o)]
    ax.bar(x=t_index, height=h-l, bottom=l, width=0.1, color=color)
    ax.bar(x=t_index, height=np.abs(o-c), bottom=np.min((o,c), axis=0), width=0.6, color=color)

    t_loc = t_index[::7200]
    #if type(t[0]) == datetime or type(t[0]) == pd._libs.tslibs.timestamps.Timestamp:
    #    label_loc = [datetime.strftime(x, '%H-%M') for x in t[::12]]
    label_loc = [x[:10] for x in t[::7200]]
    ax.set_xticks(ticks=t_loc, labels=label_loc, rotation=45);

    return ax

def get_init_slope_intercept(df):
    """Calculate initial best-fit resistance line using Linear Regression."""
    X = df[['loc_index']].values  # Independent variable
    y = df['mid'].values  # Dependent variable

    # Create and fit the model
    model = LinearRegression()
    model.fit(X, y)

    # Get the coefficients
    slope = model.coef_[0]
    intercept = model.intercept_

    best_fit = X * slope + intercept

    # Use the highest price points to calculate resistance
    df['candle_top'] = df[['open', 'close']].max(axis=1)

    # Compute offsets to shift trendline down to touch candle tops
    offsets = df['candle_top'].values - best_fit[:, 0]
    max_offset = min(offsets)  # We take min(offsets) to lower the line
    intercept_2 = intercept + max_offset  # Adjust intercept

    return slope, intercept_2

def get_best_fit_slope_intercept(df):
    """Calculate the initial best-fit line without adjustments."""
    X = df[['loc_index']].values  # Independent variable
    y = df['mid'].values  # Dependent variable

    # Create and fit the model
    model = LinearRegression()
    model.fit(X, y)

    # Get the coefficients
    slope = model.coef_[0]
    intercept = model.intercept_

    return slope, intercept

# Objective function to minimize: sum of squared differences for resistance line
def objective(params, X, candle_top):
    m, b = params
    line = m * X + b
    return np.sum((line - candle_top) ** 2)  # Minimize distance from candle tops

# Constraint function to ensure resistance line stays above candle tops
def constraint(params, X, candle_top):
    m, b = params
    line = m * X + b
    return line - candle_top  # Ensure line >= candle_top

def get_final_slope_intercept(df_new, slope_init, intercept_init):
    """Optimize the resistance trendline using scipy minimize."""
    initial_params = [slope_init, intercept_init]
    X = df_new[['loc_index']].values

    # Define highest price levels as resistance
    df_new['candle_top'] = df_new[['open', 'close']].max(axis=1)

    # Define constraints
    constraints = {
        'type': 'ineq',  # Ensures line stays above candle tops
        'fun': constraint,
        'args': (X.flatten(), df_new['candle_top'].values)
    }

    # Perform the optimization
    result = minimize(
        objective,
        initial_params,
        args=(X.flatten(), df_new['candle_top'].values),
        constraints=constraints,
        method='SLSQP'
    )

    optimized_slope, optimized_intercept = result.x
    return optimized_slope, optimized_intercept

def visualize_trendline(df_new, slope, intercept):
    """Plot candlestick chart with resistance trendline."""
    _ax = candlestick_ax(t=df_new['date'], o=df_new['open'], h=df_new['high'], l=df_new['low'], c=df_new['close'])
    X = df_new[['loc_index']].values  # Independent variable
    best_fit_init = X * slope + intercept
    plt.plot(X - min(X), best_fit_init[:, 0], 'r-')  # Red line for resistance

def safegraph(df_new, slope, intercept, name):
    """Save the resistance trendline chart in the alert folder."""
    os.makedirs("alert", exist_ok=True)  # Ensure directory exists
    _ax = candlestick_ax(t=df_new['date'], o=df_new['open'], h=df_new['high'], l=df_new['low'], c=df_new['close'])
    X = df_new[['loc_index']].values  # Independent variable
    best_fit_init = X * slope + intercept
    plt.plot(X - min(X), best_fit_init[:, 0], 'r-')  # Red line for resistance
    file_path = os.path.join("alert", name)
    plt.savefig(file_path)
    plt.close()

def candles_close_to_trendline(df_new, slope, intercept):
    """Find the last value of the resistance trendline."""
    X = df_new[['loc_index']].values  # Independent variable
    best_fit_init = X * slope + intercept
    return best_fit_init[-1, 0]

def dist_from_trendline(df_new, slope, intercept, percentile=50):
    """Calculate percentile-based distance from the resistance line."""
    X = df_new[['loc_index']].values  # Independent variable
    best_fit_init = X * slope + intercept
    trendline_dist = df_new['candle_top'].values - best_fit_init[:, 0]
    return np.nanpercentile(trendline_dist, percentile)


#1 hour analysis
symbol = 'BTCUSDT'  
interval = '1m'  
lookback = '4 hours ago UTC'

while True:
    # Fetch data
    df = get_historical_klines(symbol, interval, lookback)
    # print("From Time")
    # print(df.iloc[0, 0])
    # print("To Time")
    # print(df.iloc[-1, 0])

    #_ax = candlestick_ax(t=df['date'], o=df['open'], h=df['high'], l=df['low'], c=df['close'])
    df['mid'] = (df['open'] + df['close'])/2
    slope_init, intercept_init = get_init_slope_intercept(df)
    slope_final, intercept_final = get_final_slope_intercept(df, slope_init, intercept_init)
    visualize_trendline(df, slope_final, intercept_final)

    support = candles_close_to_trendline(df, slope_final, intercept_final)
    current = df.iloc[-1,4]

    if (abs(current-support))<100:
        print("Sent to telegram")
        name = str(int(current)) +" at "+ str(int(support))+".png"
        safegraph(df, slope_final, intercept_final,name)
        #Send telegram message
        message = "BTC Chart at resistance level at 1 hour tf"
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={CHAT_ID}&text={message}"
        r= requests.get(url)
        print(r.json())

        #Send telegram pictures
        files = {'photo': open('alert/'+ name, 'rb')}
        resp = requests.post('https://api.telegram.org/bot7440240128:AAHGgBidb-mEjSOfzWXJ2hzY8UupUDlvKEs/sendPhoto?chat_id=5097888685', files=files)
        print(resp.json())
