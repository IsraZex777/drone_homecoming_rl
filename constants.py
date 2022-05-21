from drone_controller import DroneActions
from ou_action_noice import OUActionNoise

state_amount = 4

action_type_amount = len(DroneActions)
action_duration_lower_limit = .5
action_duration_upper_limit = 10

std_dev = 0.2
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))


batch_size = 20
total_episodes = 100
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.005

buffer = Buffer(50000, 64)