import logging
import math

import gym
import airsim
import pandas as pd

import tensorflow as tf

from flight_recording.actor_observer import ActorObserver
from drone_controller import (
    DroneActions,
)
from agent_drone_controller import AgentDroneController


class AirSimDroneEnvironment(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 drone_name,
                 forward_path_csv_path: str,
                 max_distance_ratio: float = 1.5,
                 logger: logging.Logger = logging.getLogger("dummy")):
        super(AirSimDroneEnvironment, self).__init__()
        self.logger = logger

        self.drone_name = drone_name
        self.max_distance_ratio = max_distance_ratio
        self.forward_path_data = pd.read_csv(forward_path_csv_path)
        self.init_position_x = self.forward_path_data.iloc[0, "position_x"]
        self.init_position_y = self.forward_path_data.iloc[0, "position_y"]
        self.init_position_z = self.forward_path_data.iloc[0, "position_z"]

        curr_position_x = self.forward_path_data.iloc[-1, "x_position"]
        curr_position_y = self.forward_path_data.iloc[-1, "y_position"]
        curr_position_z = self.forward_path_data.iloc[-1, "z_position"]

        self.initial_distance = math.sqrt(
            (curr_position_x - self.init_position_x) ** 2 +
            (curr_position_y - self.init_position_y) ** 2 +
            (curr_position_z - self.init_position_z)
        )

        self.observer = ActorObserver(drone_name=drone_name)
        self.observer.start_flight_recording()

        self.controller = AgentDroneController(drone_name=drone_name)

    def reset(self):
        self.controller.reset(self.init_position_x,
                              self.init_position_y,
                              self.init_position_z,)
        self.observer.reset_recording_data()

        return self.forward_path_data

    def step(self, action):

        action_type_vector, action_duration = action

        action_type_index = tf.math.argmax(action_type_vector).numpy()
        action_type = DroneActions(action_type_index)

        # takes action
        self.controller.handle_action(action_type, action_duration, stop_duration=min(6, 1.5 * action_duration))

        obs_state = self.observer.get_recording_data()
        self.observer.reset_recording_data()

        has_collied = obs_state["has_collided"]
        obs_state = obs_state.drop(columns=["has_collided"])

        # cannot collied
        if has_collied.any():
            return obs_state, None, True, {"reason": "Has collied"}

        # Calculates reword
        curr_position_x = obs_state.iloc[-1, "x_position"]
        curr_position_y = obs_state.iloc[-1, "y_position"]
        curr_position_z = obs_state.iloc[-1, "z_position"]
        pos_distance = math.sqrt(
            (curr_position_x - self.init_position_x) ** 2 +
            (curr_position_y - self.init_position_y) ** 2 +
            (curr_position_z - self.init_position_z)
        )

        distance_ratio = pos_distance / self.initial_distance

        # Cannot go more far than 1.5 of the initial distance
        if distance_ratio > self.max_distance_ratio:
            return obs_state, None, True, {"reason": f"gone too far {distance_ratio}"}

        reward = self.initial_distance / pos_distance

        # Too close to border isn't good - to avoid collision
        # lowers reward if is too close to object
        curr_distance = obs_state.iloc[-1, "distance"]
        if curr_distance < 1:
            reward = reward * curr_distance

        return obs_state, reward, False, {}
