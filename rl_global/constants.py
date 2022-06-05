import os

state_amount = 6

action_type_amount = 3
action_duration_lower_limit = .1
action_duration_upper_limit = 1

simulator_time_factor = 2

max_distance = 10

std_dev = 0.2

batch_size = 50
total_episodes = 1000
total_epochs = 50
max_episode_steps = 100
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.005

epsilon_min = 0.1  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (
    epsilon_max - epsilon_min
)
# Number of frames for exploration
epsilon_greedy_frames = 1000.0

RL_REPLAY_MEMORY_FOLDER_NAME = "rl_replay_memory"
RL_REPLAY_MEMORY_FOLDER_PATH = os.path.join(os.path.dirname(__file__), RL_REPLAY_MEMORY_FOLDER_NAME)

RL_FORWARD_PATHS = os.path.join(os.path.dirname(__file__), "rl_forward_paths")

