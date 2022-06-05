import tensorflow as tf
from tensorflow.keras import (
    layers,
)

from rl_global.constants import (
    action_type_amount,
    state_amount,
)


def create_critic_model() -> tf.keras.Model:
    """
    creates the actor model that
        [] receives -
            [] Observed state
            [] chosen actions:
                [] Action type
                [] Action duration
        [] outputs the reward

    @return: tensorflow keras model
    """

    # State as input
    state_input = layers.Input(shape=state_amount)
    state_hidden = layers.Dense(16, activation="relu")(state_input)
    state_hidden = layers.Dense(32, activation="relu")(state_hidden)

    # Action type as input
    action_type_input = layers.Input(shape=action_type_amount)
    action_type_hidden = layers.Dense(32, activation="relu")(action_type_input)

    # Action duration as input
    action_duration_input = layers.Input(shape=1)
    action_duration_hidden = layers.Dense(16, activation="relu")(action_duration_input)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_hidden, action_type_hidden, action_duration_hidden])

    hidden = layers.Dense(256, activation="relu")(concat)
    hidden = layers.Dense(256, activation="relu")(hidden)

    reward_output = layers.Dense(1)(hidden)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_type_input, action_duration_input], reward_output)

    return model
