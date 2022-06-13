import logging
import random
import tensorflow as tf

from typing import Tuple

from drone_interface import DroneActions
from rl_global.constants import action_type_amount


def make_actor_action(q_model: tf.keras.Model,
                      state,
                      epsilon: float = 0.2,
                      logger: logging.Logger = logging.getLogger("dummy")) -> Tuple[DroneActions, float]:
    """
    Chooses the based action using the input actor model

    Uses two methods to ensure exploration:
        [] epsilon greedy - for the action types
        [] noice - for the action duration (which is continuous)

    @param q_model:  Input q model that is used to choose the actions
    @param state: Input observed state
    @param epsilon: percent for action exploration
    @param logger: Operation logger
    @return: Tuple of - action type index, action duration in seconds
    """



    # score_per_action = q_model(state)

    # epsilon greedy for exploration
    if random.random() < epsilon:
        logger.debug(f"Exploration: applies action type shuffle (following: epsilon greedy method[{epsilon}%])")
        random_action = random.randrange(0, action_type_amount)
        action_type = DroneActions(random_action)
    else:
        if state[0][2] >= .15:
            if int(state[0][3].numpy()):
                action_type = DroneActions.TURN_RIGHT
            else:
                action_type = DroneActions.TURN_LEFT
        else:
            action_type = DroneActions.FORWARD

        # # Action with the highest reward
        # action_type_index = tf.math.argmax(score_per_action[0]).numpy()
        # action_type = DroneActions(action_type_index)

    action_duration = 0

    if action_type in [DroneActions.FORWARD, DroneActions.UP, DroneActions.DOWN]:
        action_duration = 1
    elif action_type in [DroneActions.TURN_LEFT, DroneActions.TURN_RIGHT]:
        action_duration = .2

    # return action_type_vector, np.array([random.random()])
    return action_type, action_duration
