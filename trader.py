import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import wrappers
from tf_agents.environments import tf_py_environment
from environment import LiveBinanceEnvironment
import pandas as pd
from tf_agents.networks import q_network
import os 
from dotenv import load_dotenv
from binance.client import Client
import sched, time
import tensorflow.python.trackable.base

load_dotenv()

AGENT_MODEL_PATH = "policy_1min_1000"

# Check for GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Initialize Binance client
api_key = os.getenv("CLIENT_KEY")
api_secret = os.getenv("SECRET_KEY")
client = Client(api_key, api_secret)

live_env = LiveBinanceEnvironment("ETH", "USDT", 0.005 , 0.0001, 15, 15, 10, 12, 26 , target_balance=10000)
live_env = tf_py_environment.TFPyEnvironment(live_env)

# Initialize Q Network
fc_layer_params = (25, 50, 100)
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

# Load the agent policy
try:
    policy = tf_agent.policy
    policy_state = policy.get_initial_state(batch_size=3)
    print("Policy loaded from: {}".format(AGENT_MODEL_PATH))
except:
    raise Exception("Model needed")

# Schedule the agent to run at every T time interval
# The agent will make a step depending on the current state
s = sched.scheduler(time.time, time.sleep)

time_step = live_env.current_time_step()

def run_step(step, sc):
    action_step = policy.action(step)
    next_time_step = live_env.step(action_step.action)
    time_step = next_time_step
    print(action_step.action)
    print(time_step.observation)
    print(time_step.reward)
    s.enter(60, 1, run_step, (step, sc,))

s.enter(60, 1, run_step, (time_step, s, ))
s.run()
