import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from cryptocom_env import LiveCryptoComEnvironment  # Adjust the import to match your file structure
from tf_agents.networks import q_network
import os
from dotenv import load_dotenv
import sched, time
import tensorflow.python.trackable.base
# Load environment variables
load_dotenv()

AGENT_MODEL_PATH = "policy_1min_1000"

# Check for GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Initialize API keys
api_key = os.getenv("API_KEY")
api_secret = os.getenv("API_SECRET")
print("API_KEY:", api_key)
print("API_SECRET:", api_secret)

# Initialize the live environment
live_env = LiveCryptoComEnvironment("BTC/USD", 0.00007, 0.00001, 15, 9, 12, 26, target_balance=1500, profit_threshold=1.0)

# Wrap the live environment in a TFPyEnvironment
tf_live_env = tf_py_environment.TFPyEnvironment(live_env)

# Initialize Q Network
fc_layer_params = (25, 50, 100)
q_net = q_network.QNetwork(
    tf_live_env.observation_spec(),
    tf_live_env.action_spec(),
    fc_layer_params=fc_layer_params
)

# Initialize DQN Agent
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4)
train_step_counter = tf.compat.v2.Variable(0)
tf_agent = dqn_agent.DqnAgent(
    tf_live_env.time_step_spec(),
    tf_live_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    train_step_counter=train_step_counter
)

# Initialize the agent
tf_agent.initialize()

# Load the agent policy
try:
    policy = tf_agent.policy
    checkpoint_dir = AGENT_MODEL_PATH
    checkpoint = tf.train.Checkpoint(policy=policy)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
    print("Policy loaded from: {}".format(AGENT_MODEL_PATH))
except Exception as e:
    raise Exception(f"Model loading failed: {e}")

# Schedule the agent to run at every T time interval
s = sched.scheduler(time.time, time.sleep)
time_step = tf_live_env.reset()

def run_step(sc, time_step):
    global policy, tf_live_env
    action_step = policy.action(time_step)
    next_time_step = tf_live_env.step(action_step.action)

    print("\033[94m########################Agent chose to {}.#########################\033[0m".format(['hold', 'buy', 'sell'][action_step.action.numpy()[0]]))
    print("Action:", action_step.action.numpy())
    print("Observation:", next_time_step.observation.numpy())
    print("Reward:", next_time_step.reward.numpy())

    observation = next_time_step.observation.numpy()

    # Check if the observation has the expected length before printing its components
    if observation.ndim == 2 and observation.shape[0] == 1:
        observation = observation[0]  # Extract the inner array

    print(f"Observation Length: {len(observation)}")
    print(f"Observation Content: {observation}")

    expected_length = 5  # Adjust this based on the actual length of your observation array
    if len(observation) == expected_length:
        print(f"Closing Price: {observation[0]}")
        print(f"MACD Trend: {observation[1]}")
        print(f"Number of Open Positions: {observation[2]}")
        print(f"Available USD Balance: {observation[3]}")
        print(f"Available BTC Balance: {observation[4]}")
    else:
        print(f"Unexpected observation length. Expected {expected_length}, but got {len(observation)}.")

    s.enter(60, 5, run_step, (sc, next_time_step))


# Start the scheduling
s.enter(60, 5, run_step, (s, time_step))
s.run()
