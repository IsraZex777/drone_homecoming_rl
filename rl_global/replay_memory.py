import random
import tensorflow as tf
from collections import namedtuple, deque
from rl_global.constants import (
    action_type_amount,
    state_amount
)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


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

    def _sample_to_train_data(self, input_sample):
        transition_batch = Transition(*zip(*input_sample))

        state_tensor = tf.convert_to_tensor(transition_batch.state, dtype="float32")
        state_tensor = tf.reshape(state_tensor, (-1, state_amount))

        action_type_tensor = tf.convert_to_tensor(transition_batch.action_type, dtype="float32")
        action_type_tensor = tf.reshape(action_type_tensor, (-1, action_type_amount))

        action_duration_tensor = tf.convert_to_tensor(transition_batch.action_duration, dtype="float32")
        action_duration_tensor = tf.reshape(action_duration_tensor, (-1, 1))

        reward_tensor = tf.convert_to_tensor(transition_batch.reward, dtype="float32")
        reward_tensor = tf.reshape(reward_tensor, (-1, 1))

        next_state_tensor = tf.convert_to_tensor(transition_batch.next_state, dtype="float32")
        next_state_tensor = tf.reshape(next_state_tensor, (-1, state_amount))

        return state_tensor, action_type_tensor, action_duration_tensor, reward_tensor, next_state_tensor

    def sample(self, batch_size):
        random_sample = random.sample(self.memory, min(len(self.memory), batch_size))
        return self._sample_to_train_data(random_sample)

    def get_batches(self, batch_size: int, shuffle: bool = True):
        """
        Returns all the memory data in form of batches
        @param shuffle: Should memory be shuffled
        @param batch_size: Should memory be shuffled
        @return:
        """
        memory = list(self.memory)
        random.shuffle(memory)
        batches = [self._sample_to_train_data(batch) for batch in chunks(memory, batch_size)]
        return batches

    def __len__(self):
        return len(self.memory)
