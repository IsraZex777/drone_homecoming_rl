import logging
import math
import numpy as np

import gym
import airsim
import pandas as pd

import tensorflow as tf

from flight_recording.actor_observer import ActorObserver
from drone_controller import (
    DroneActions,
)
from agent_drone_controller import AgentDroneController
from utils import calculate_yaw_diff


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

        self.init_position_x = self.forward_path_data.iloc[0]["position_x"]
        self.init_position_y = self.forward_path_data.iloc[0]["position_y"]
        self.init_position_z = self.forward_path_data.iloc[0]["position_z"]

        self.last_position_x = self.forward_path_data.iloc[-1]["position_x"]
        self.last_position_y = self.forward_path_data.iloc[-1]["position_y"]
        self.last_position_z = self.forward_path_data.iloc[-1]["position_z"]

        self.initial_distance = math.sqrt(
            (self.last_position_x - self.init_position_x) ** 2 +
            (self.last_position_y - self.init_position_y) ** 2 +
            abs(self.last_position_z - self.init_position_z)
        )

        self.observer = ActorObserver(drone_name=drone_name)
        self.observer.start_flight_recording()

        self.controller = AgentDroneController(drone_name=drone_name)

    def reset(self):
        self.controller.reset(self.last_position_x,
                              self.last_position_y,
                              self.last_position_z, )
        self.observer.reset_recording_data()

        return self.forward_path_data

    def step(self, action):

        action_type_vector, action_duration = action
        action_duration = action_duration[0]

        action_type_index = tf.math.argmax(action_type_vector[0]).numpy()
        action_type = DroneActions(action_type_index)

        # takes action
        self.controller.handle_action(action_type, action_duration, stop_duration=min(6, max(2, 1.5 * action_duration)))
        # self.controller.handle_action(action_type, action_duration, stop_duration=0)

        obs_state = self.observer.get_recording_data()
        self.observer.reset_recording_data()

        has_collied = obs_state["has_collided"]
        distance = obs_state["distance"]
        obs_state = obs_state.drop(columns=["has_collided"])

        # cannot collied
        if has_collied.any() or distance.any() < .5:
            return obs_state, None, True, {"reason": "Has collied"}

        # Calculates reword
        curr_position_x = obs_state.iloc[-1]["position_x"]
        curr_position_y = obs_state.iloc[-1]["position_y"]
        curr_position_z = obs_state.iloc[-1]["position_z"]
        pos_distance = math.sqrt(
            (curr_position_x - self.init_position_x) ** 2 +
            (curr_position_y - self.init_position_y) ** 2 +
            abs(curr_position_z - self.init_position_z)
        )

        distance_ratio = pos_distance / self.initial_distance

        # cannot collied
        if curr_position_z > -1:
            return obs_state, None, True, {"reason": "Gone too low"}

        # Cannot go more far than 1.5 of the initial distance
        if distance_ratio > self.max_distance_ratio:
            return obs_state, None, True, {"reason": f"gone too far {distance_ratio}"}

        reward = self.initial_distance / pos_distance

        # Too close to border isn't good - to avoid collision
        # lowers reward if is too close to object
        curr_distance = obs_state.iloc[-1]["distance"]
        if curr_distance < 3:
            reward = reward * (curr_distance / 3) ** 2

        yaw_diff = calculate_yaw_diff(np.array([obs_state.iloc[-1]["orientation_x"],
                                                obs_state.iloc[-1]["orientation_y"],
                                                obs_state.iloc[-1]["orientation_z"],
                                                obs_state.iloc[-1]["orientation_w"]]),
                                      np.array((curr_position_x, curr_position_y, curr_position_z)),
                                      np.array((self.init_position_x, self.init_position_y, self.init_position_z)))
        yaw_reward_factor = max(0.5, 1 - abs(yaw_diff))
        print(yaw_reward_factor)
        print(yaw_diff)
        reward *= yaw_reward_factor

        return obs_state, reward, False, {}
