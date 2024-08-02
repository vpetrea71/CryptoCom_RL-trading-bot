import tensorflow as tf
import numpy as np
import ta
import os
import pandas as pd
#import alpaca_trade_api as tradeapi
from binance.client import Client
import datetime as dt
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
import tensorflow.python.trackable.base
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import time


tf.compat.v1.enable_v2_behavior()
class TradingEnvironment(py_environment.PyEnvironment):
    """
    Environment to train agent with volume spike included

    Requires:
    initial_balance - starting balance
    features - DataFrame containing price and feature data

    Goal:
    Include relevant features to the state to help the agent make the best actions
    Reward the agent properly for each action through the _step function to optimize P/L
    """

    def __init__(self, initial_balance, features, position_increment=0.001, fees=0.0001):
        self.t = 0
        self.position_increment = position_increment
        self.fees = fees
        self.positions = []
        self.features = features
        self.initial_balance = self.balance = self.cash_balance = initial_balance
        self.volume_history = []  # Placeholder for volume history
        self.target_balance = 1500000  # Target balance to achieve
        self.profits = 0
        self._episode_ended = False

        # Define action and observation specs
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=2, name='action')
        # Include volume spike in observation
        observation_shape = len(self.features.features.columns) + 3  # Add 3 for balance, volume, and volume spike
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(observation_shape,), dtype=np.float32, name="observation")

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec
    
    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        closing_price = self.features.price_data.iloc[self.t]['Close']
        volume = self.features.price_data.iloc[self.t]['Volume']
        features = self.features.features.iloc[self.t]
       
        # Calculate volume spike
        if len(self.volume_history) > 0:
            avg_volume = np.mean(self.volume_history[-30:])  # Use the last 10 periods for average
            volume_spike = volume / avg_volume if avg_volume > 0 else 0
        else:
            volume_spike = 0

        # Immediate rewards
        immediate_reward = 0

        if action == 0:  # Hold
            print("\033[94mAgent chose to hold.\033[0m")
            ma5 = self.features.price_data.iloc[self.t - 4:self.t + 1]['Close'].mean()
            if closing_price > ma5:
                # Positive reward for holding in an uptrend
                immediate_reward += 1
            elif closing_price < ma5:
                # Positive reward for waiting in a downtrend
                immediate_reward += 1
                
           
                
        elif action == 1:  # Buy
            print("\033[92mAgent chose to buy.\033[0m")
            p = closing_price * self.position_increment * (1 + self.fees)
            price_change = closing_price - self.features.price_data.iloc[self.t - 1]['Close']

            if price_change < 0:
                immediate_reward = 2  # Positive reward for buying during a price decrease
            else:
                immediate_reward = 3  # Positive reward for buying during a price increase

            if p > self.cash_balance:
                immediate_reward = -1  # Penalty for insufficient funds
            else:
                self.cash_balance -= p
                self.positions.append(closing_price)
                immediate_reward += features['macd']

                # Check if the agent can buy additional positions based on profit
                if self.profits >= 1:  # Adjust the condition based on your criteria
                    additional_positions = int(self.profits / 1)  # Calculate additional positions
                    self.positions.extend([closing_price] * additional_positions)  # Buy additional positions
                    self.profits -= additional_positions * 1  # Deduct the cost from profits
                    immediate_reward += 1  # Reward for buying additional positions

        elif action == 2:  # Sell
            print("\033[91mAgent chose to sell.\033[0m")
            if len(self.positions) == 0:
                immediate_reward = -1  # Penalty for selling without positions
            else:
                profits = 0
                for p in self.positions:
                    self.positions.pop(0)
                    profits += (closing_price - p) * self.position_increment * (1 - self.fees)
                    self.cash_balance += closing_price * self.position_increment * (1 - self.fees)

                # Check if total profit plus fees is above 0
                if profits > 0:
                    immediate_reward += profits - features['macd']
                    self.profits += profits  # Add profits to the total profits
                    immediate_reward += 5  # Reward for selling at a profit
                else:
                    immediate_reward -= 1  # Penalty for selling at a loss

                immediate_reward -= features['macd']

        # Calculate new balance
        self.balance = self.cash_balance
        for _ in self.positions:
            self.balance += closing_price * self.position_increment

        # Append volume to volume history
        self.volume_history.append(volume)

        # Print current state
        print("Time = {}: #Positions = {}: Balance = {}: Closing Price = {}: Volume = {}: Volume Spike = {:.2f}".format(
            self.t, len(self.positions), self.balance, closing_price, volume, volume_spike))
        
        # Check if the target balance is achieved or exceeded
        if self.balance >= self.target_balance:
            immediate_reward += 10  # Reward for achieving or exceeding target balance

        self.t += 1

        if self.t == len(self.features.price_data) - 1:
            self._episode_ended = True

        # Include volume spike in state
        self._state = [self.balance, volume, volume_spike] + self.features.features.iloc[self.t].values.tolist()
        return ts.transition(
            np.array(self._state, dtype=np.float32), reward=immediate_reward, discount=0.7)

    def _reset(self):
        self.t = 0
        self._episode_ended = False
        self.profits = 0
        self.balance = self.initial_balance
        self.cash_balance = self.initial_balance
        self.positions = []
        self.volume_history = []
        self._state = [self.balance, 0, 0] + self.features.features.iloc[0].values.tolist()
        return ts.restart(np.array(self._state, dtype=np.float32))

    def buy_and_hold(self):
        amount = self.initial_balance / self.features.price_data.iloc[0]['Close']
        return self.features.price_data * amount


class LiveBinanceEnvironment(py_environment.PyEnvironment):
    """
    Environment to trade on Binance.

    Does not include features class to organize time series data.
    """

    def __init__(self, asset1, asset2, position_increment, fees, price_history_t, mean_history_t, macd_t, fast_ema, slow_ema, target_balance):
        super().__init__()
        self.asset1 = asset1
        self.asset2 = asset2
        self.assetpair = asset1 + asset2
        self.position_increment = position_increment
        self.fees = fees
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.price_history_t = price_history_t
        self.mean_history_t = mean_history_t
        self.macd_t = macd_t
        self.trades = []
        self.orders = []
        self.volume_history = []
        self.target_balance = target_balance  # Target balance to achieve ($3000)
        api_key = os.getenv("CLIENT_KEY")
        api_secret = os.getenv("SECRET_KEY")
        self.client = Client(api_key, api_secret)

        # Synchronize server time
        self.time_offset = self.synchronize_server_time()

        self._columns = [
            'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close time', 'Quote asset volume', 'Number of trades',
            'Taker buy base asset volume', 'Taker buy quote asset volume', 'ignore'
        ]

        prices = self.client.get_historical_klines(self.assetpair, self.client.KLINE_INTERVAL_1MINUTE, "3 DAY ago UTC")
        time.sleep(5)
        prices = pd.DataFrame(prices, columns=self._columns).astype(float)
        prices['Open time'] = prices['Open time'].apply(lambda x: dt.datetime.fromtimestamp(int(x) / 1000))
        self.price_data = prices.set_index('Open time')

        self.initial_balance = float(self.client.get_asset_balance(asset='USDT')['free'])
        time.sleep(5)
        self.balance = self.initial_balance
        self.free_balance = self.initial_balance

        self.return_history = [self.price_data.iloc[-k, :]['Close'] - self.price_data.iloc[-k - 1, :]['Close'] for k in reversed(range(self.price_history_t))]
        self.mean_data = self.price_data.rolling(20, min_periods=1).mean()
        self.MACD_trend = ta.trend.ema_indicator(self.price_data['Close'], self.fast_ema) - ta.trend.ema_indicator(self.price_data['Close'], self.slow_ema)
        self.MACD_trend = self.MACD_trend.fillna(self.MACD_trend[self.slow_ema]).tolist()
        self.MACD = [self.MACD_trend[-k] for k in reversed(range(self.macd_t))]

        # Calculate MA5
        self.MA5 = self.price_data['Close'].rolling(window=5).mean().tolist()

        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=2, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(shape=(price_history_t + macd_t + 1,), dtype=np.float32, name="observation")

    def synchronize_server_time(self):
        server_time = self.client.get_server_time()
        server_timestamp = server_time['serverTime'] // 1000
        local_timestamp = int(time.time())
        time_offset = server_timestamp - local_timestamp
        return time_offset

    def fetch_news_data(self):
        url = "https://finance.yahoo.com/topic/crypto"
        response = requests.get(url)
        time.sleep(5)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            news_articles = soup.find_all("div", class_="largeTitle")
            news_data = [article.text.strip() for article in news_articles]
            return news_data
        else:
            print("Failed to fetch news data. Status code:", response.status_code)
            return None

    def analyze_news_sentiment(self, news_data):
        news_sentiments = []
        for article in news_data:
            blob = TextBlob(article)
            polarity = blob.sentiment.polarity
            if polarity > 0:
                news_sentiments.append("positive")
            elif polarity < 0:
                news_sentiments.append("negative")
            else:
                news_sentiments.append("neutral")
        return news_sentiments

    def process_news_sentiment(self, news_sentiments):
        actions = []
        for news_sentiment in news_sentiments:
            if (news_sentiment == 'positive'):
                actions.append(1)
            elif (news_sentiment == 'negative'):
                actions.append(2)
            else:
                actions.append(0)
        return actions

    def get_order_book(self):
        order_book = self.client.get_order_book(symbol=self.assetpair)
        time.sleep(5)
        return order_book

    def process_order_book(self, order_book):
        time.sleep(5)
        bids = order_book['bids']
        asks = order_book['asks']
        total_bid_quantity = sum(float(bid[1]) for bid in bids)
        total_ask_quantity = sum(float(ask[1]) for ask in asks)
        if total_bid_quantity > total_ask_quantity:
            return 1
        elif total_ask_quantity > total_bid_quantity:
            return 2
        else:
            return 0

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    def _reset(self):
        self.balance = self.initial_balance
        self.free_balance = self.initial_balance
        self.trades = []
        self.orders = []
        self.price_data = self.price_data.iloc[:-1]
        self.return_history = [self.price_data.iloc[-k, :]['Close'] - self.price_data.iloc[-k - 1, :]['Close'] for k in reversed(range(self.price_history_t))]
        self.mean_data = self.price_data.rolling(20, min_periods=1).mean()
        self.MACD_trend = ta.trend.ema_indicator(self.price_data['Close'], self.fast_ema) - ta.trend.ema_indicator(self.price_data['Close'], self.slow_ema)
        self.MACD_trend = self.MACD_trend.fillna(self.MACD_trend[self.slow_ema]).tolist()
        self.MACD = [self.MACD_trend[-k] for k in reversed(range(self.macd_t))]
        self.MA5 = self.price_data['Close'].rolling(window=5).mean().tolist()
        self._state = [self.balance] + self.return_history + self.MACD + [self.MA5[-1]]
        return ts.restart(np.array(self._state, dtype=np.float32))

    def _step(self, action):
        reward = 0
        cost_basis = 0
        current_price = self.price_data.iloc[-1]['Close']
        ma5 = self.MA5[-1]

        usd_balance = float(self.client.get_asset_balance(asset='USDT')['free'])
        btc_balance = float(self.client.get_asset_balance(asset=self.asset1)['free'])

        if action == 0:  # Hold
            print("################Agent chose to hold.#################")
            if current_price > ma5:  # Uptrend
                reward = 0.1
            else:  # Downtrend
                reward = -0.1
        elif action == 1:  # Buy
            print("################Agent chose to buy.##################")
            avg_price_info = self.client.get_avg_price(symbol=self.assetpair)
            time.sleep(5)
            average_price = float(avg_price_info['price'])
            buy_quantity = self.position_increment / average_price
            if usd_balance < self.position_increment:
                print("Insufficient balance to buy.")
                reward = -1
            else:
                try:
                    order = self.client.order_market_buy(symbol=self.assetpair, quantity=buy_quantity)
                    time.sleep(5)
                    print("Bought {} of {}".format(buy_quantity, self.asset1))
                    self.trades.append((order['fills'][0]['price'], buy_quantity))
                    self.free_balance -= buy_quantity * average_price  # Adjust free balance
                    reward = 0.5 * (self.MACD[-1])
                except Exception as e:
                    print("Buy failed:", e)
        elif action == 2:  # Sell
            print("################Agent chose to sell.###################")
            avg_price_info = self.client.get_avg_price(symbol=self.assetpair)
            time.sleep(5)
            average_price = float(avg_price_info['price'])
            sell_quantity = self.position_increment / average_price
            if btc_balance < sell_quantity:
                print("Insufficient balance to sell.")
                reward = -1
            else:
                try:
                    order = self.client.order_market_sell(symbol=self.assetpair, quantity=sell_quantity)
                    time.sleep(5)
                    print("Sold {} of {}".format(sell_quantity, self.asset1))
                    self.trades.append((order['fills'][0]['price'], -sell_quantity))
                    self.free_balance += sell_quantity * average_price  # Adjust free balance
                    reward = 0.5 * (self.MACD[-1])
                except Exception as e:
                    print("Sell failed:", e)

        self.balance = float(self.client.get_asset_balance(asset='USDT')['free'])
        time.sleep(5)  # Add delay after the request
        self.free_balance = self.balance
        cur_price = float(self.client.get_avg_price(symbol=self.assetpair)['price'])
        time.sleep(5)  # Add delay after the request

        try:
            order_book = self.get_order_book()
            action_from_order_book = self.process_order_book(order_book)
            print("Action taken based on order book signal:", action_from_order_book)
        except Exception as e:
            print("Error fetching or processing order book:", e)
            action_from_order_book = 0

        if action_from_order_book != action:
            print("Agent changing action based on order book signal.")
            action = action_from_order_book

        if reward > 1:
            additional_positions = int(reward / 1)
            print("Additional positions can be bought:", additional_positions)
            try:
                for _ in range(additional_positions):
                    order = self.client.order_market_buy(symbol=self.assetpair, quantity=self.position_increment)
                    time.sleep(5)  # Add delay after each additional buy order
                    print("Bought additional position of {}".format(self.asset1))
                    reward -= 1
                    self.trades.append((order['fills'][0]['price'], self.position_increment))
            except Exception as e:
                print("Additional buy failed:", e)

        self.return_history.pop(0)
        self.return_history.append(cur_price - self.price_data.iloc[-1]['Close'])
        self.price_data.loc[pd.Timestamp.now()] = {'Close': cur_price}
        self.MACD_trend = ta.trend.ema_indicator(self.price_data['Close'], self.fast_ema)
        self.MACD_trend = self.MACD_trend.fillna(self.MACD_trend[self.slow_ema]).tolist()
        self.MACD.pop(0)
        self.MACD.append(self.MACD_trend[-1])
        self.MA5.append(self.price_data['Close'].rolling(window=5).mean().iloc[-1])
        self._state = [self.balance, self.free_balance] + self.return_history + self.MACD + [self.MA5[-1]]

        asset_balance = float(self.client.get_asset_balance(asset=self.asset1)['free'])
        print(f"{self.asset1} Balance:", asset_balance)
        print("State components:")
        print("Balance (USDT):", self.balance)
        print("Free Balance (USDT):", self.free_balance)
        print(f"{self.asset1} Balance:", asset_balance)
        print("Action taken:", action)
        time.sleep(5)  # Ensure 5 seconds delay before returning

        # Check if target balance achieved
        if self.free_balance >= self.target_balance:
            print("Target balance achieved! Episode completed.")
            return ts.termination(np.array(self._state, dtype=np.float32), reward=10)
        else:
            return ts.transition(np.array(self._state, dtype=np.float32), reward=reward, discount=0.7)

    def buy_and_hold(self):
        amount = self.initial_balance / self.price_data.iloc[0, :]['Close']
        return self.price_data * amount
