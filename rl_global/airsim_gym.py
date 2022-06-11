import logging
import math
import numpy as np
import gym
import pandas as pd

from flight_recording.actor_observer import ActorObserver
from drone_interface.agent_drone_controller import AgentDroneController
from rl_global.utils import calculate_yaw_diff
from rl_global.constants import (
    simulator_time_factor,
    max_position_distance
)

from drone_interface import DroneActions


class AirSimDroneEnvironment(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 drone_name,
                 forward_path_csv_path: str,
                 allow_same_action: bool = True,
                 logger: logging.Logger = logging.getLogger("dummy")):
        super(AirSimDroneEnvironment, self).__init__()
        self.logger = logger

        # position variables
        self.forward_path_csv_path = ""
        self.forward_path_data = None
        self.init_position_x = None
        self.init_position_y = None
        self.init_position_z = None
        self.last_position_x = None
        self.last_position_y = None
        self.last_position_z = None
        self.last_quat_x = None
        self.last_quat_y = None
        self.last_quat_z = None
        self.last_quat_w = None
        self.initial_distance = None

        self.drone_name = drone_name

        # initializes data collector
        self.observer = ActorObserver(drone_name=drone_name)
        self.observer.start_flight_recording()

        # initializes drone controller
        self.controller = AgentDroneController(drone_name=drone_name)

        # same action reward factor
        self.allow_same_action = allow_same_action
        self.prev_action = None
        self.same_action_factor = 1
        self.same_action_scaler = .90

        # resets the drone to its initial location
        self.reset(forward_path_csv_path)

    def reset(self, forward_path_csv_path: str = ""):

        if forward_path_csv_path != self.forward_path_csv_path:
            self.forward_path_data = pd.read_csv(forward_path_csv_path)
            self.forward_path_csv_path = forward_path_csv_path

            self.init_position_x = self.forward_path_data.iloc[0]["position_x"]
            self.init_position_y = self.forward_path_data.iloc[0]["position_y"]
            self.init_position_z = self.forward_path_data.iloc[0]["position_z"]

            self.last_position_x = self.forward_path_data.iloc[-1]["position_x"]
            self.last_position_y = self.forward_path_data.iloc[-1]["position_y"]
            self.last_position_z = self.forward_path_data.iloc[-1]["position_z"]

            # self.initial_distance = math.sqrt(
            #     (self.last_position_x - self.init_position_x) ** 2 +
            #     (self.last_position_y - self.init_position_y) ** 2 +
            #     (self.last_position_z - self.init_position_z) ** 2
            # )

            self.last_quat_x = self.forward_path_data.iloc[-1]["orientation_x"]
            self.last_quat_y = self.forward_path_data.iloc[-1]["orientation_y"]
            self.last_quat_z = self.forward_path_data.iloc[-1]["orientation_z"]
            self.last_quat_w = self.forward_path_data.iloc[-1]["orientation_w"]

        self.controller.reset(self.last_position_x,
                              self.last_position_y,
                              self.last_position_z,
                              self.last_quat_x,
                              self.last_quat_y,
                              self.last_quat_z,
                              self.last_quat_w)
        self.observer.reset_recording_data()

        return self.forward_path_data

    def step(self, action):
        action_type, action_duration = action

        # takes action
        stop_duration = min(6, max(3, 1.5 * action_duration))
        self.controller.handle_action(action_type, action_duration / simulator_time_factor,
                                      stop_duration=stop_duration / simulator_time_factor)
        # self.controller.handle_action(action_type, action_duration, stop_duration=0)

        obs_state = self.observer.get_recording_data()
        self.observer.reset_recording_data()

        has_collied = obs_state["has_collided"]
        distance = obs_state["distance"]
        obs_state = obs_state.drop(columns=["has_collided"])

        # Calculates reword
        curr_position_x = obs_state.iloc[-1]["position_x"]
        curr_position_y = obs_state.iloc[-1]["position_y"]
        curr_position_z = obs_state.iloc[-1]["position_z"]
        pos_distance = math.sqrt(
            (curr_position_x - self.init_position_x) ** 2 +
            (curr_position_y - self.init_position_y) ** 2 +
            abs(curr_position_z - self.init_position_z)
        )

        pos_distance_normalized = pos_distance / max_position_distance
        # distance_ratio = pos_distance / self.initial_distance

        reward = 1 - pos_distance_normalized

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

        yaw_reward_factor = ((1 - abs(yaw_diff / 180)) * .6) + .4
        reward *= yaw_reward_factor ** 2

        # cannot collied
        if curr_position_z > -1:
            return obs_state, reward * abs(curr_position_z), True, {"reason": "Gone too low"}

        # cannot collied
        # print(distance.any() < 1)
        # print(distance)
        if has_collied.any() or np.less(distance, .7).any():
            return obs_state, reward, True, {"reason": "Has collied"}

        # Can't get more far then max_position_distance
        if pos_distance > max_position_distance:
            return obs_state, reward, True, {"reason": f"gone too far {pos_distance}"}

        if self.prev_action == action_type:
            self.same_action_factor *= self.same_action_scaler
        else:
            self.same_action_factor = 1

        if action_type.name in [DroneActions.TURN_LEFT.name, DroneActions.TURN_RIGHT.name] and \
                not self.allow_same_action:
            reward *= self.same_action_factor

        self.prev_action = action_type

        return obs_state, reward, False, {}
