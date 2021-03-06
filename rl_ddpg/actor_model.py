import logging
import random
import numpy as np
import tensorflow as tf

from typing import Tuple
from tensorflow.keras import (
    layers,
)

from .ou_action_noice import OUActionNoise

from rl_global.constants import (
    action_type_amount,
    action_duration_upper_limit,
    action_duration_lower_limit,
    state_amount,
)

from drone_interface.drone_controller import DroneActions


def create_actor_model() -> tf.keras.Model:
    """
    creates the actor model that
        [] receives - observed state
        [] outputs  -
            [] chosen action (as softmax output)
            [] duration time for the action to take place

    @return: tensorflow keras model
    """
    last_init = tf.random_uniform_initializer(minval=-0.001, maxval=0.001)

    input_layer = layers.Input(shape=state_amount)
    hidden_layer = layers.Dense(16, activation="relu")(input_layer)
    hidden_layer = layers.Dense(32, activation="relu")(hidden_layer)

    action_type_hidden_layer = layers.Dense(16, activation="relu")(hidden_layer)
    action_type_output = layers.Dense(action_type_amount, activation="softmax")(action_type_hidden_layer)

    action_duration_hidden_layer = layers.Dense(16, activation="relu", kernel_initializer=last_init)(hidden_layer)
    action_duration_output = layers.Dense(1, activation="sigmoid", kernel_initializer=last_init)(
        action_duration_hidden_layer)

    model = tf.keras.Model(input_layer, [action_type_output, action_duration_output])

    return model


def make_actor_action(actor_model: tf.keras.Model,
                      state,
                      noise_object: OUActionNoise,
                      logger: logging.Logger = logging.getLogger("dummy")) -> Tuple[tf.Tensor, np.array]:
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
    action_type_vector = tf.add(action_type_vector,
                                np.array([noise_object(), noise_object(), noise_object()]).reshape((1, 3)))

    # Adding noise to action
    action_duration = abs(action_duration.numpy()[0] + noise_object())

    # return action_type_vector, np.array([random.random()])
    return action_type_vector, action_duration


def action_duration_to_real(output_duration: np.array):
    action_duration = abs(output_duration[0] * action_duration_upper_limit)
    action_duration = max(action_duration, action_duration_lower_limit)
    return action_duration


def action_type_to_real(action_type: tf.Tensor) -> DroneActions:
    action_type_index = tf.math.argmax(action_type[0]).numpy()
    action_type = DroneActions(action_type_index)
    return action_type
