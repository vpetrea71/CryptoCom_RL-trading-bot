import numpy as np
import pandas as pd
import time
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from mt5linux import MetaTrader5
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import ta
import tensorflow as tf
tf.compat.v1.enable_v2_behavior()

class LiveMT5Environment(py_environment.PyEnvironment):
    def __init__(self, mt5_login, mt5_password, mt5_server, symbol, lot_size, deviation, sl_points, tp_points, start_date, timeframe, price_history_t, macd_t, fast_ema, slow_ema):
        super().__init__()
        self.mt5_login = mt5_login
        self.mt5_password = mt5_password
        self.mt5_server = mt5_server
        self.symbol = symbol
        self.lot_size = lot_size
        self.deviation = deviation
        self.sl_points = sl_points
        self.tp_points = tp_points
        self.start_date = pd.Timestamp(start_date)
        self.timeframe = timeframe
        self.price_history_t = price_history_t
        self.macd_t = macd_t
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema

        # Initialize MT5 connection
        self.mt5 = MetaTrader5()
        if not self.mt5.initialize(path=None, login=self.mt5_login, password=self.mt5_password, server=self.mt5_server, timeout=30):
            print("initialize() failed, error code =", self.mt5.last_error())
            quit()

        # Initialize account balance
        self.initial_balance = self.get_account_balance()
        self.balance = self.initial_balance
        self.free_balance = self.initial_balance

        # Fetch historical prices
        self.price_data = self.fetch_historical_prices()

        # Initialize indicators
        self.return_history = [self.price_data.iloc[-k]['close'] - self.price_data.iloc[-k - 1]['close'] for k in reversed(range(self.price_history_t))]
        self.MACD_trend = ta.trend.ema_indicator(self.price_data['close'], self.fast_ema) - ta.trend.ema_indicator(self.price_data['close'], self.slow_ema)
        self.MACD_trend = self.MACD_trend.fillna(self.MACD_trend[self.slow_ema]).tolist()
        self.MACD = [self.MACD_trend[-k] for k in reversed(range(self.macd_t))]

        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=2, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(shape=(self.price_history_t + self.macd_t + 1,), dtype=np.float32, name="observation")

    def get_account_balance(self):
        account_info = self.mt5.account_info()
        if account_info is not None:
            return account_info.balance
        else:
            print("Failed to get account balance, error code =", self.mt5.last_error())
            return 0

    def fetch_historical_prices(self):
        rates = self.mt5.copy_rates_range(self.symbol, self.timeframe, self.start_date, pd.Timestamp.now())
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        return df

    def fetch_news_data(self):
        url = "https://finance.yahoo.com/topic/crypto"
        response = requests.get(url)
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
            if news_sentiment == 'positive':
                actions.append(1)
            elif news_sentiment == 'negative':
                actions.append(2)
            else:
                actions.append(0)
        return actions

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    def _step(self, action):
        reward = 0
        if action == 0:
            print("################ Agent chose to hold. #################")
        elif action == 1:
            print("################ Agent chose to buy. ##################")
            print("Account Balance before action:", float(self.get_account_balance()))
            try:
                request = {
                    "action": self.mt5.TRADE_ACTION_DEAL,
                    "symbol": self.symbol,
                    "volume": self.lot_size,
                    "type": self.mt5.ORDER_TYPE_BUY,
                    "price": self.mt5.symbol_info_tick(self.symbol).ask,
                    "deviation": self.deviation,
                    "magic": 234000,
                    "comment": "Python script buy order",
                    "type_time": self.mt5.ORDER_TIME_GTC,
                    "type_filling": self.mt5.ORDER_FILLING_RETURN,
                }
                result = self.mt5.order_send(request)
                if result.retcode != self.mt5.TRADE_RETCODE_DONE:
                    print("Order failed, retcode={}".format(result.retcode))
                    reward -= 1
                else:
                    print("Bought {} lot of {}".format(self.lot_size, self.symbol))
                    self.balance = self.get_account_balance()
                    reward += 0.5 * (self.MACD[-1])
            except Exception as e:
                print("Buy failed:", e)
                reward -= 1
        elif action == 2:
            print("################ Agent chose to sell. ###################")
            print("Account Balance before action:", float(self.get_account_balance()))
            try:
                request = {
                    "action": self.mt5.TRADE_ACTION_DEAL,
                    "symbol": self.symbol,
                    "volume": self.lot_size,
                    "type": self.mt5.ORDER_TYPE_SELL,
                    "price": self.mt5.symbol_info_tick(self.symbol).bid,
                    "deviation": self.deviation,
                    "magic": 234000,
                    "comment": "Python script sell order",
                    "type_time": self.mt5.ORDER_TIME_GTC,
                    "type_filling": self.mt5.ORDER_FILLING_RETURN,
                }
                result = self.mt5.order_send(request)
                if result.retcode != self.mt5.TRADE_RETCODE_DONE:
                    print("Order failed, retcode={}".format(result.retcode))
                    reward -= 1
                else:
                    print("Sold {} lot of {}".format(self.lot_size, self.symbol))
                    self.balance = self.get_account_balance()
                    reward += 0.5 * (self.MACD[-1])
            except Exception as e:
                print("Sell failed:", e)
                reward -= 1

        self.balance = self.get_account_balance()
        self.free_balance = self.balance
        cur_price = self.mt5.symbol_info_tick(self.symbol).last
        
        self.return_history.pop(0)
        self.return_history.append(cur_price - self.price_data.iloc[-1]['close'])
        self.price_data.loc[pd.Timestamp.now()] = {'close': cur_price}
        self.MACD_trend = ta.trend.ema_indicator(self.price_data['close'], self.fast_ema)
        self.MACD_trend = self.MACD_trend.fillna(self.MACD_trend[self.slow_ema]).tolist()
        self.MACD.pop(0)
        self.MACD.append(self.MACD_trend[-1])
        self._state = [self.balance, self.free_balance] + self.return_history + self.MACD

        print("State components:")
        print("Balance:", self.balance)
        print("Free Balance:", self.free_balance)
        print("Action taken:", action)
        time.sleep(1)

        return ts.transition(np.array(self._state, dtype=np.float32), reward)

    def _reset(self):
        self._state = [self.balance] + self.return_history + self.MACD
        return ts.restart(np.array(self._state, dtype=np.float32))

    def buy_and_hold(self):
        amount = self.initial_balance / self.price_data.iloc[0]['close']
        return self.price_data * amount

    def shutdown(self):
        self.mt5.shutdown()

    def __del__(self):
        self.shutdown()
