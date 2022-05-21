import random
import tensorflow as tf
from collections import namedtuple, deque
from constants import (
    action_type_amount,
    state_amount
)

Transition = namedtuple('Transition',
                        ('state', 'action_type', "action_duration", 'reward', 'next_state'))


class ReplayMemory(object):

    def __init__(self, capacity=10000):
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

        state_tensor = tf.convert_to_tensor(transition_batch.state, dtype="float32")
        state_tensor = tf.reshape(state_tensor, (-1, state_amount))

        action_type_tensor = tf.convert_to_tensor(transition_batch.action_type, dtype="float32")
        action_type_tensor = tf.reshape(action_type_tensor, (-1, action_type_amount))

        action_duration_tensor = tf.convert_to_tensor(transition_batch.action_duration, dtype="float32")
        action_duration_tensor = tf.reshape(action_duration_tensor, (-1, 1))

        reward_tensor = tf.convert_to_tensor(transition_batch.reward, dtype="float32")
        action_duration_tensor = tf.reshape(action_duration_tensor, (-1, 1))

        next_state_tensor = tf.convert_to_tensor(transition_batch.next_state, dtype="float32")
        next_state_tensor = tf.reshape(next_state_tensor, (-1, state_amount))

        return state_tensor, action_type_tensor, action_duration_tensor, reward_tensor, next_state_tensor

    def __len__(self):
        return len(self.memory)
