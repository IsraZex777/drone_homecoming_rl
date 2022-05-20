import tensorflow as tf
from tensorflow.keras import (
    layers,
)

from constants import (
    action_type_amount,
    action_duration_limit,
    state_amount,
)


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
    action_duration_output = action_duration_output * action_duration_limit

    model = tf.keras.Model(input_layer, [action_type_output, action_duration_output])

    return model
