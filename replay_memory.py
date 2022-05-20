import random
import tensorflow as tf
from collections import namedtuple, deque

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def extend(self, replay_memory):
        """Save a transition"""
        self.memory.extend(replay_memory.memory)

    def sample(self, batch_size):
        random_sample = random.sample(self.memory, batch_size)
        transition_batch = Transition(*zip(*random_sample))

        state_tensor = tf.convert_to_tensor(transition_batch.state)
        action_tensor = tf.convert_to_tensor(transition_batch.action)
        reward_tensor = tf.convert_to_tensor(transition_batch.reward)
        next_state_tensor = tf.convert_to_tensor(transition_batch.next_state)

        return state_tensor, action_tensor, reward_tensor, next_state_tensor

    def __len__(self):
        return len(self.memory)