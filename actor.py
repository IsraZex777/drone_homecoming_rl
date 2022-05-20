import random
import numpy as np
import tensorflow as tf

from typing import Tuple
from tensorflow.keras import (
    layers,
)

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
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    input_layer = layers.Input(shape=state_amount)
    hidden_layer = layers.Dense(256, activation="relu")(input_layer)
    hidden_layer = layers.Dense(256, activation="relu")(hidden_layer)

    action_type_hidden_layer = layers.Dense(16, activation="relu")(hidden_layer)
    action_type_output = layers.Dense(len(action_type_amount), activation="softmax")(action_type_hidden_layer)

    action_duration_hidden_layer = layers.Dense(16, activation="relu")(hidden_layer)
    action_duration_output = layers.Dense(1,
                                          activation="tanh",
                                          kernel_initializer=last_init)(action_duration_hidden_layer)
    action_duration_output = action_duration_output * action_duration_upper_limit

    model = tf.keras.Model(input_layer, [action_type_output, action_duration_output])

    return model


def make_actor_action(actor_model: tf.keras.Model,
                      state,
                      noise_object,
                      epsilon) -> Tuple[DroneActions, float]:
    """
    Chooses the based action using the input actor model

    Uses two methods to ensure exploration:
        [] epsilon greedy - for the action types (which are discrete)
        [] noice - for the action duration (which is continuous)

    @param actor_model:  Input actor model that chooses the actions
    @param state: Input observed state
    @param noise_object: Object that creates noice in order to make exploration
    @param epsilon: percent for action exploration
    @return: Tuple of - action type index, action duration in seconds
    """
    action_type_vector, action_duration = actor_model(state)

    # chooses random action type - in epsilon percent
    if random.random() < epsilon:
        action_type_index = random.randint(0, action_type_amount - 1)
    else:
        action_type_index = tf.math.argmax(action_type_vector).numpy()

    chosen_action_type = DroneActions(action_type_index)

    noise = noise_object()
    # Adding noise to action
    sampled_actions = tf.squeeze(action_duration).numpy() + noise

    # We make sure action is within bounds
    legal_action_duration = np.clip(sampled_actions,
                                    action_duration_lower_limit,
                                    action_duration_upper_limit)

    return chosen_action_type, legal_action_duration[0]
