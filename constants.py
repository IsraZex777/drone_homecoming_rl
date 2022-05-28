from drone_controller import DroneActions
from ou_action_noice import OUActionNoise

state_amount = 4

action_type_amount = 6
action_duration_lower_limit = .5
action_duration_upper_limit = 5

max_distance = 10

std_dev = 0.2


batch_size = 30
total_episodes = 100
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.005

