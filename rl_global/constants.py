import os

state_amount = 5

action_type_amount = 5
action_duration_lower_limit = .2
action_duration_upper_limit = 3

simulator_time_factor = 2

max_distance = 10

std_dev = 0.2

batch_size = 1000
total_episodes = 100
total_epochs = 500
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.005

RL_REPLAY_MEMORY_FOLDER_NAME = "rl_replay_memory"
RL_REPLAY_MEMORY_FOLDER_PATH = os.path.join(os.path.dirname(__file__), RL_REPLAY_MEMORY_FOLDER_NAME)
