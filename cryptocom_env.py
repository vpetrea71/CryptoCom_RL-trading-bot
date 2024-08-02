import pandas as pd
import numpy as np
import ccxt
import os
import time
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.environments import py_environment
import tensorflow as tf
import tensorflow.python.trackable.base
tf.compat.v1.enable_v2_behavior()

class LiveCryptoComEnvironment(py_environment.PyEnvironment):
    def __init__(self, asset, position_size=0.00007, fees=0.00001, price_history_t=15, macd_t=9, fast_ema=12, slow_ema=26, target_balance=1500, profit_threshold=1.0, sleep_duration=5):
        super().__init__()

        self.asset = asset
        self.position_size = position_size
        self.fees = fees
        self.price_history_t = price_history_t
        self.macd_t = macd_t
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.target_balance = target_balance
        self.profit_threshold = profit_threshold
        self.positions = []
        self.profits = 0
        self.volume_history = []
        self.trades = []
        self.orders = []
        self.sleep_duration = sleep_duration

        self.api_key = os.getenv("API_KEY")
        self.api_secret = os.getenv("API_SECRET")
        self.api_url = os.getenv("API_URL")
        self.client = self.create_client()

        self.accounts = {}
        self.update_user_accounts()

        self.price_data = self.fetch_price_data()
        self.volume_history = self.price_data['volume'].tolist()

        self.macd_trend = self.calculate_macd()
        self.update_state()

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=2, name='action')
        self._observation_spec = array_spec.ArraySpec(
            shape=(len(self._state),), dtype=np.float32, name='observation')

        self._episode_ended = False

    def create_client(self):
        return ccxt.cryptocom({
            'apiKey': self.api_key,
            'secret': self.api_secret,
        })

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self.__init__(self.asset, self.position_size, self.fees, self.price_history_t,
                      self.macd_t, self.fast_ema, self.slow_ema, self.target_balance, self.profit_threshold, self.sleep_duration)
        return ts.restart(np.array(self._state, dtype=np.float32))

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        reward = 0
        try:
            closing_price = self.client.fetch_ticker(self.asset)['last']
            self.sleep()

            new_data = pd.DataFrame([self.client.fetch_ohlcv(self.asset, timeframe='1m', limit=1)[-1]], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            new_data['timestamp'] = pd.to_datetime(new_data['timestamp'], unit='ms')
            new_data.set_index('timestamp', inplace=True)

            self.price_data = pd.concat([self.price_data, new_data]).tail(self.price_history_t)
            self.volume_history.append(new_data['volume'].values[-1])
            if len(self.volume_history) > self.price_history_t:
                self.volume_history = self.volume_history[-self.price_history_t:]

            self.update_state()

            current_volume = self.volume_history[-1]
            avg_volume = np.mean(self.volume_history[-self.price_history_t:]) if len(self.volume_history) >= self.price_history_t else 0
            volume_spike = current_volume / avg_volume if avg_volume > 0 else 0

            if action == 0:  # Hold
                ma5 = self.price_data['close'].tail(5).mean()
                if closing_price > ma5:
                    reward += 1  # Positive reward for holding in an uptrend
                elif closing_price < ma5:
                    reward += 0.3  # Positive reward for waiting in a downtrend

            elif action == 1:  # Buy
                buy_price = closing_price * (1 + self.fees)
                available_balance = self.get_balance()
                if buy_price * self.position_size > available_balance:
                    self.sleep()
                    available_balance = self.get_balance()
                    if buy_price * self.position_size > available_balance:
                        reward = -1
                    else:
                        self.execute_buy(closing_price, volume_spike, reward)
                else:
                    self.execute_buy(closing_price, volume_spike, reward)

            elif action == 2:  # Sell
                btc_balance = self.get_btc_balance()
                self.sleep()
                if btc_balance <= 0:
                    reward = -1
                else:
                    self.execute_sell(closing_price, volume_spike, reward)

            self.reinvest_profits(closing_price)

            if self.get_balance() + self.get_btc_balance() * closing_price >= self.target_balance:
                reward += 5
        except Exception as e:
            print(f"Failed to fetch latest price data: {e}")
            self.reconnect_client()
            self._episode_ended = True
            return ts.termination(np.array(self._state, dtype=np.float32), reward)

        self._state = np.array(self._state, dtype=np.float32)
        return ts.transition(self._state, reward=reward)

    def execute_buy(self, closing_price, volume_spike, reward):
        try:
            order = self.client.create_market_order(self.asset, 'buy', self.position_size)
            self.orders.append(order)
            self.positions.append(closing_price)
            self.accounts['USD']['available'] -= closing_price * self.position_size * (1 + self.fees)
            self.accounts['BTC']['available'] += self.position_size
            reward = 0.2 + 0.5 * self.macd_trend.iloc[-1]
            if volume_spike > 1.2:
                reward += 0.3
            self.trades.append({
                'action': 'buy',
                'price': closing_price,
                'size': self.position_size,
                'timestamp': pd.Timestamp.now()
            })
        except Exception as e:
            print(f"Failed to place buy order: {e}")
            reward = -1

    def execute_sell(self, closing_price, volume_spike, reward):
        try:
            sell_price = closing_price * (1 - self.fees)
            sell_quantity = self.position_size
            order = self.client.create_market_order(self.asset, 'sell', sell_quantity)
            self.orders.append(order)
            profits = sum((closing_price - p) * self.position_size * (1 - self.fees) for p in self.positions)
            self.profits += profits
            self.accounts['USD']['available'] += sell_price * sell_quantity
            self.accounts['BTC']['available'] -= sell_quantity
            self.positions = []
            reward = 0.5 + (profits / 1 if profits > 0 else -0.5) - self.macd_trend.iloc[-1]
            if profits > 0:
                reward += profits  # Reward based on profit
            else:
                reward += profits  # Negative reward based on loss

            if volume_spike > 1.2:
                reward += 0.3
            self.trades.append({
                'action': 'sell',
                'price': closing_price,
                'size': sell_quantity,
                'timestamp': pd.Timestamp.now(),
                'profit': profits
            })
        except Exception as e:
            print(f"Failed to place sell order: {e}")
            reward = -1

    def reinvest_profits(self, closing_price):
        profit_amount = self.profits
        if profit_amount > self.profit_threshold:
            reinvest_amount = profit_amount / closing_price
            if reinvest_amount > 0:
                try:
                    order = self.client.create_market_order(self.asset, 'buy', reinvest_amount)
                    self.orders.append(order)
                    self.positions.append(closing_price)
                    self.accounts['USD']['available'] -= profit_amount
                    self.accounts['BTC']['available'] += reinvest_amount
                    self.profits = 0
                except Exception as e:
                    print(f"Failed to reinvest profits: {e}")

    def reconnect_client(self):
        print("Reconnecting to Crypto.com API...")
        try:
            self.client = self.create_client()
            self.update_user_accounts()
        except Exception as e:
            print(f"Reconnection failed: {e}")
            self.sleep()

    def fetch_price_data(self):
        try:
            since = int((pd.Timestamp.now() - pd.Timedelta(days=1)).timestamp() * 1000)
            prices = self.client.fetch_ohlcv(self.asset, timeframe='1m', since=since)
            columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            df = pd.DataFrame(prices, columns=columns)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            print(f"Failed to fetch price data: {e}")
            self.reconnect_client()
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    def calculate_macd(self):
        try:
            self.price_data['fast_ema'] = self.price_data['close'].ewm(span=self.fast_ema, adjust=False).mean()
            self.price_data['slow_ema'] = self.price_data['close'].ewm(span=self.slow_ema, adjust=False).mean()
            self.price_data['macd'] = self.price_data['fast_ema'] - self.price_data['slow_ema']
            self.price_data['signal'] = self.price_data['macd'].ewm(span=self.macd_t, adjust=False).mean()
            self.price_data['macd_trend'] = self.price_data['macd'] - self.price_data['signal']
            return self.price_data['macd_trend']
        except Exception as e:
            print(f"Failed to calculate MACD: {e}")
            self.reconnect_client()
            return pd.Series(dtype=np.float64)

    def update_user_accounts(self):
        """
        Fetch and update user account balances and staking details.
        """
        try:
            self.sleep()
            raw_accounts = self.client.fetch_balance()
            self.accounts = {
                'USD': {
                    'balance': raw_accounts['total'].get('USD', 0.0),
                    'available': raw_accounts['free'].get('USD', 0.0),
                    'stake': 0,
                    'order': 0
                },
                'BTC': {
                    'balance': raw_accounts['total'].get('BTC', 0.0),
                    'available': raw_accounts['free'].get('BTC', 0.0),
                    'stake': 0,
                    'order': 0
                }
            }
        except Exception as e:
            print(f"Failed to update user accounts: {e}")
            self.reconnect_client()

    def get_balance(self):
        return self.accounts.get('USD', {}).get('available', 0)

    def get_btc_balance(self):
        return self.accounts.get('BTC', {}).get('available', 0)

    def update_state(self):
        if self.price_data is not None and not self.price_data.empty:
            closing_price = self.price_data['close'].iloc[-1]
            macd_trend = self.macd_trend.iloc[-1] if not self.macd_trend.empty else 0
            num_positions = len(self.positions)
            usd_balance = self.get_balance()
            btc_balance = self.get_btc_balance()
            current_volume = self.volume_history[-1] if self.volume_history else 0
            avg_volume = np.mean(self.volume_history[-self.price_history_t:]) if len(self.volume_history) >= self.price_history_t else 0
            volume_spike = current_volume / avg_volume if avg_volume > 0 else 0
            self._state = [
                closing_price,
                macd_trend,
                num_positions,
                usd_balance,
                btc_balance,
                volume_spike
            ]
        else:
            self._state = [0, 0, 0, 0, 0, 0]

    def sleep(self):
        time.sleep(self.sleep_duration)
