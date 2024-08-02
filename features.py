import pandas as pd
import datetime as dt
import numpy as np
import ta

class Features():
    """
    Features will hold time series data relevant to technical analysis

    Requires:
    price_data - A DataFrame of OHLC bars
    feature_length - The lookback period for all the features
    """

    def __init__(self, price_data, feature_length):
        self.price_data = price_data
        self.feature_length = feature_length
        self.rsi = ta.momentum.RSIIndicator(close=price_data['Close'])
        self.macd = ta.trend.MACD(close=price_data['Close'], fillna=True)
        
        # Manually compute MACD with custom parameters
        self.price_data['ema_12'] = self.price_data['Close'].ewm(span=12, adjust=False).mean()
        self.price_data['ema_26'] = self.price_data['Close'].ewm(span=26, adjust=False).mean()
        self.price_data['macd_custom'] = self.price_data['ema_12'] - self.price_data['ema_26']
        self.price_data['macd_custom_signal'] = self.price_data['macd_custom'].ewm(span=9, adjust=False).mean()
        self.price_data['macd_custom_diff'] = self.price_data['macd_custom'] - self.price_data['macd_custom_signal']

        self.features = pd.DataFrame({
            "minute_returns": self.price_data['Close'].pct_change().fillna(0) * 100,
            "rsi": self.rsi.rsi(),
            "macd": self.macd.macd(),
            "macd_signal": self.macd.macd_signal(),
            "macd_diff": self.macd.macd_diff(),
            "macd_custom": self.price_data['macd_custom'],
            "macd_custom_signal": self.price_data['macd_custom_signal'],
            "macd_custom_diff": self.price_data['macd_custom_diff']
        })

        # We will be shifting the start date to the length of feature_length
        # because we want to initialize our features and
        # agent state with nonzero data. This will help when learning
        # on smaller amounts of data
        self.price_data = self.price_data.iloc[feature_length:]

if __name__ == "__main__":
    CSV_PATH = 'scripts/asset_prices/BTCUSDT.csv'  # @param
    # Split the CSV_PATH string into individual file paths
    file_paths = CSV_PATH.split(',')

    # Read each CSV file into a DataFrame and append it to the list
    prices = pd.read_csv(CSV_PATH, parse_dates=True, index_col=0)

    # Split data into training and test set
    date_split = dt.datetime(2024, 3, 30, 1, 0)  # @param
    prices = pd.read_csv(CSV_PATH, parse_dates=True, index_col=0)
    train = prices[:date_split]
    test = prices[date_split:]

    s = Features(prices, 10)
    print(s.features[:date_split])
