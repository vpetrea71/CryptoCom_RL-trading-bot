import os
from binance.client import Client
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import time

PRICE_PATH = "asset_prices/"
PLOT_PATH = "asset_prices/plots/"

# Initialize Binance client
api_key = os.getenv("CLIENT_KEY")
api_secret = os.getenv("SECRET_KEY")
client = Client(api_key, api_secret)

_columns = [
    'Open time',
    'Open',
    'High',
    'Low',
    'Close',
    'Volume',
    'Close time',
    'Quote asset volume',
    'Number of trades',
    'Taker buy base asset volume',
    'Taker buy quote asset volume',
    'ignore'
]

def create_directories():
    # Create asset_prices directory if it doesn't exist
    if not os.path.exists(PRICE_PATH):
        os.makedirs(PRICE_PATH)
    
    # Create asset_prices/plots directory if it doesn't exist
    if not os.path.exists(PLOT_PATH):
        os.makedirs(PLOT_PATH)

def get_crypto_data(crypto):
    create_directories()
    
    klines = client.get_historical_klines(crypto, Client.KLINE_INTERVAL_1MINUTE, "120 DAY  ago UTC")
    prices = pd.DataFrame(klines, columns=_columns).astype(float)

    # Change timestamp to datetime and format it as desired
    prices['Open time'] = prices['Open time'].apply(lambda x: dt.datetime.fromtimestamp(int(x)/1000).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
    prices = prices.set_index('Open time')
    
    # Save data
    prices.to_csv(PRICE_PATH + crypto + '.csv')

    # Display data
    plt.plot(prices['Close'])
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.savefig(PLOT_PATH + crypto + ".png")

    plt.show()

# Run the function with the desired cryptocurrency pair
get_crypto_data('BTCUSDT')
