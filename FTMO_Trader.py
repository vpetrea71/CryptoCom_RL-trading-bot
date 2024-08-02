

import os
import sched
import time
import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from ftmo_env import LiveMT5Environment  # Ensure this is correctly imported
from dotenv import load_dotenv
from mt5linux import MetaTrader5

load_dotenv()

# Set environment variables
os.environ["FTMO_ACCOUNT_ID"] = "51859944"
os.environ["FTMO_SERVER"] = "ICMarketsSC-Demo"
os.environ["FTMO_PASSWORD"] = "=efD&61dTFWY6!i"

AGENT_MODEL_PATH = "policy_1min_1000"

# Check for GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Initialize MetaTrader 5 environment
mt5 = MetaTrader5()
if not mt5.initialize():
    raise Exception("Failed to initialize MetaTrader 5")

account_id = os.getenv("FTMO_ACCOUNT_ID")
server = os.getenv("FTMO_SERVER")
password = os.getenv("FTMO_PASSWORD")

# Login to your account
if not mt5.login(account_id, password, server):
    raise Exception(f"Failed to login to account #{account_id}")

instrument = "EURUSD"
position_increment = 0.01
fees = 0.0001
price_history_t = 15
macd_t = 15
fast_ema = 12
slow_ema = 26

# Initialize LiveMT5Environment
live_env = LiveMT5Environment(account_id, password, server, instrument, position_increment, fees, price_history_t, macd_t, fast_ema, slow_ema)
live_env = tf_py_environment.TFPyEnvironment(live_env)

# Initialize Q Network
fc_layer_params = (100, 50, 25)
q_net = q_network.QNetwork(
    live_env.observation_spec(),
    live_env.action_spec(),
    fc_layer_params=fc_layer_params
)

# Initialize DQN Agent
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4)
train_step_counter = tf.compat.v2.Variable(0)
tf_agent = dqn_agent.DqnAgent(
    live_env.time_step_spec(),
    live_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    train_step_counter=train_step_counter
)

tf_agent.initialize()

# Load the agent policy
try:
    policy = tf_agent.policy
    policy_state = policy.get_initial_state(batch_size=1)
    tf_agent.initialize()
    tf_agent.load(AGENT_MODEL_PATH)
    print("Policy loaded from: {}".format(AGENT_MODEL_PATH))
except Exception as e:
    raise Exception(f"Model loading failed: {e}")

# Schedule the agent to run at every T time interval
# The agent will make a step depending on the current state
s = sched.scheduler(time.time, time.sleep)

def run_step(sc):
    global time_step
    time_step = live_env.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = live_env.step(action_step.action)
    print(f"Action: {action_step.action}")
    print(f"Observation: {time_step.observation}")
    print(f"Reward: {time_step.reward}")
    s.enter(60, 1, run_step, (sc,))

s.enter(60, 1, run_step, (s,))
s.run()

# Shutdown MetaTrader 5 connection
mt5.shutdown()
