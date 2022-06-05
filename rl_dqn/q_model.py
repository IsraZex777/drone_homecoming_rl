import logging
import random
import numpy as np
import tensorflow as tf

from typing import Tuple
from tensorflow.keras import (
    layers,
)

from rl_global.constants import (
    action_type_amount,
    state_amount,
)


def create_q_model() -> tf.keras.Model:
    """
    creates the dqn model that
        [] receives - observed state
        [] outputs  -
            [] chosen action

    @return: tensorflow keras model
    """

    input_layer = layers.Input(shape=state_amount)
    hidden_layer = layers.Dense(128)(input_layer)
    hidden_layer = layers.Dense(128)(hidden_layer)
    action_type_output = layers.Dense(action_type_amount, activation="linear")(hidden_layer)

    model = tf.keras.Model(input_layer, action_type_output)

    return model
