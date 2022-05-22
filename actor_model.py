import logging
import random
import numpy as np
import tensorflow as tf

from typing import Tuple
from tensorflow.keras import (
    layers,
)

from ou_action_noice import OUActionNoise

from constants import (
    action_type_amount,
    action_duration_upper_limit,
    action_duration_lower_limit,
    state_amount,
)

from drone_controller import DroneActions


def create_actor_model() -> tf.keras.Model:
    """
    creates the actor model that
        [] receives - observed state
        [] outputs  -
            [] chosen action (as softmax output)
            [] duration time for the action to take place

    @return: tensorflow keras model
    """
    # Initialize weights between -3e-3 and 3-e3
    input_layer = layers.Input(shape=state_amount)
    hidden_layer = layers.Dense(256, activation="relu")(input_layer)
    hidden_layer = layers.Dense(256, activation="relu")(hidden_layer)

    action_type_hidden_layer = layers.Dense(16, activation="relu")(hidden_layer)
    action_type_output = layers.Dense(action_type_amount, activation="softmax")(action_type_hidden_layer)

    action_duration_hidden_layer = layers.Dense(16, activation="relu")(hidden_layer)
    action_duration_output = layers.Dense(1, activation="sigmoid")(action_duration_hidden_layer)

    model = tf.keras.Model(input_layer, [action_type_output, action_duration_output])

    return model


def make_actor_action(actor_model: tf.keras.Model,
                      state,
                      noise_object: OUActionNoise,
                      epsilon: float = 0.35,
                      logger: logging.Logger = logging.getLogger("dummy")) -> Tuple[DroneActions, float]:
    """
    Chooses the based action using the input actor model

    Uses two methods to ensure exploration:
        [] epsilon greedy - for the action types
        [] noice - for the action duration (which is continuous)

    @param actor_model:  Input actor model that chooses the actions
    @param state: Input observed state
    @param noise_object: Object that creates noice in order to make exploration
    @param epsilon: percent for action exploration
    @param logger: Operation logger
    @return: Tuple of - action type index, action duration in seconds
    """
    action_type_vector, action_duration = actor_model(state)

    # shuffles action types - in epsilon percent
    if random.random() < epsilon:
        logger.debug(f"Exploration: applies action type shuffle (following: epsilon greedy method[{epsilon}%])")
        shuffled_data = tf.random.shuffle(tf.squeeze(action_type_vector))
        action_type_vector = tf.expand_dims(shuffled_data, 0)

    noise = noise_object()
    # Adding noise to action
    sampled_actions = action_duration.numpy()[0] * 10 + noise
    # print(sampled_actions)
    # We make sure action is within bounds

    return action_type_vector, sampled_actions
